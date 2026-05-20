#!/usr/bin/env python3
"""
z_visual corruption ablation for Active Perception.

Runs inference under three modes:
  normal   — standard retrieval
  zero     — z_visual replaced with zeros
  random   — z_visual replaced with Gaussian noise (same norm)

If the model truly uses z_visual, CE loss should degrade under zero/random.
Token accuracy is measured on all supervised positions (labels != -100).

Usage:
    CUDA_VISIBLE_DEVICES=1 /cortex/users/rabanof/conda_envs/qwen49/bin/python \\
        scripts/eval_z_ablation.py \\
        --config configs/exp1_ce_only.yaml \\
        --checkpoint runs/exp1_ce_only/checkpoint-best \\
        --output_dir outputs/diagnostics \\
        --num_samples 50
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator


# ── Config loading (identical to train.py) ────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_names = cfg.pop("defaults")
        for base_name in base_names:
            base_path = Path(config_path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as fb:
                    base_cfg = yaml.safe_load(fb)
                cfg = _deep_merge(base_cfg, cfg)
    return cfg


# ── z-mode context manager ────────────────────────────────────────────────────

@contextlib.contextmanager
def z_override_mode(model: ActivePerceptionModel, mode: str):
    """
    Monkey-patch PerceptionModule.forward to override z_visual.

    Modes:
      "normal" — no change
      "zero"   — replace z with zeros (ablates ALL visual signal)
      "random" — replace z with Gaussian noise scaled to the same norm as z
                 (retains magnitude but destroys content)
    """
    if mode == "normal":
        yield
        return

    original_forward = model.perception_module.forward

    if mode == "zero":
        def patched(h_perception, visual_memory, **kwargs):
            z, attn = original_forward(h_perception, visual_memory, **kwargs)
            return torch.zeros_like(z), attn
    elif mode == "random":
        def patched(h_perception, visual_memory, **kwargs):
            z, attn = original_forward(h_perception, visual_memory, **kwargs)
            noise = torch.randn_like(z)
            # Scale noise to match z's norm per vector
            z_norm = z.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            n_norm = noise.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            noise = (noise / n_norm * z_norm).to(z.dtype)
            return noise, attn
    else:
        raise ValueError(f"Unknown z_mode: {mode}")

    model.perception_module.forward = patched
    try:
        yield
    finally:
        model.perception_module.forward = original_forward


# ── Token accuracy computation ────────────────────────────────────────────────

def compute_token_accuracy(
    logits: torch.Tensor,          # [B, T, V]
    modified_labels: torch.Tensor, # [B, T]
    modified_input_ids: torch.Tensor,  # [B, T]
    perc_out_id: int,
) -> Dict[str, float]:
    """
    Compute token accuracy on supervised positions.

    Returns:
      all_acc   — accuracy on all labels != -100
      post_perc_acc — accuracy on tokens immediately after PERC_OUT
    """
    B, T, V = logits.shape

    # Causal LM: logit[t] predicts token[t+1]
    pred = logits[:, :-1, :].argmax(-1)   # [B, T-1]
    targets = modified_labels[:, 1:]      # [B, T-1]
    ids_shift = modified_input_ids[:, :-1]  # [B, T-1]

    sup_mask = (targets != -100)
    if sup_mask.sum() == 0:
        return {"all_acc": None, "post_perc_acc": None, "n_supervised": 0}

    all_correct = (pred[sup_mask] == targets[sup_mask]).float()
    all_acc = all_correct.mean().item()

    # Post-PERC_OUT: positions where ids_shift[t] == perc_out_id (meaning
    # we just emitted PERC_OUT at position t, and now predict t+1)
    post_mask = (ids_shift == perc_out_id) & sup_mask
    if post_mask.sum() == 0:
        post_acc = None
    else:
        post_correct = (pred[post_mask] == targets[post_mask]).float()
        post_acc = post_correct.mean().item()

    return {
        "all_acc": all_acc,
        "post_perc_acc": post_acc,
        "n_supervised": int(sup_mask.sum()),
        "n_post_perc": int(post_mask.sum()),
    }


# ── Per-mode evaluation ───────────────────────────────────────────────────────

def run_eval_mode(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
    mode: str,
    num_samples: int,
) -> Dict:
    """Run inference for `num_samples` under the given z mode."""
    perc_out_id = model.special_tokens.PERC_OUT

    losses, all_accs, post_accs = [], [], []
    n = min(num_samples, len(dataset))

    for i in range(n):
        item = dataset[i]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad(), z_override_mode(model, mode):
            out = model.training_forward(**batch)

        loss_ce = out["loss_ce"].item()
        losses.append(loss_ce)

        if out["attn_weights_list"][0] is not None:
            acc_info = compute_token_accuracy(
                out["logits"],
                out["modified_labels"],
                out["modified_input_ids"],
                perc_out_id,
            )
            if acc_info["all_acc"] is not None:
                all_accs.append(acc_info["all_acc"])
            if acc_info["post_perc_acc"] is not None:
                post_accs.append(acc_info["post_perc_acc"])

    return {
        "mode": mode,
        "n_samples": n,
        "loss_ce_mean": float(np.mean(losses)),
        "loss_ce_std":  float(np.std(losses)),
        "token_acc_all_mean":       float(np.mean(all_accs)) if all_accs else None,
        "token_acc_post_perc_mean": float(np.mean(post_accs)) if post_accs else None,
        "n_with_perception": len(all_accs),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_cfg = ActivePerceptionConfig(**{
        k: v for k, v in cfg.get("model", {}).items()
        if k in ActivePerceptionConfig.__dataclass_fields__
    })
    model = ActivePerceptionModel(model_cfg)
    model.load_perception_module(args.checkpoint)
    model = model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded: {args.checkpoint}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    eval_path = data_cfg.get("eval_data_path") or data_cfg.get("data_path")
    special_ids = model.get_special_token_ids()

    dataset = ActivePerceptionDataset(
        data_path=eval_path,
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        supervision_mode=data_cfg.get("supervision_mode", "full"),
    )
    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )
    logger.info(f"Eval dataset: {len(dataset)} samples, using {args.num_samples}")

    # ── Run all three modes ───────────────────────────────────────────────────
    modes = ["normal", "zero", "random"]
    results = []

    for mode in modes:
        logger.info(f"\n  Running mode: {mode} ...")
        r = run_eval_mode(model, dataset, collator, device, mode, args.num_samples)
        results.append(r)
        acc_a = f"{r['token_acc_all_mean']:.4f}" if r['token_acc_all_mean'] is not None else "N/A"
        acc_p = f"{r['token_acc_post_perc_mean']:.4f}" if r['token_acc_post_perc_mean'] is not None else "N/A"
        logger.info(
            f"  [{mode:6s}] loss={r['loss_ce_mean']:.4f} ± {r['loss_ce_std']:.4f}  "
            f"acc_all={acc_a}  acc_post_perc={acc_p}"
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    normal = results[0]
    logger.info("\n" + "=" * 72)
    logger.info("z_visual CORRUPTION ABLATION RESULTS")
    logger.info("=" * 72)
    logger.info(
        f"{'Mode':<8} {'CE Loss':>12} {'Delta Loss':>12} {'Acc (all)':>12} {'Acc (post)':>12}"
    )
    logger.info("-" * 72)
    for r in results:
        delta = r["loss_ce_mean"] - normal["loss_ce_mean"]
        acc_all  = f"{r['token_acc_all_mean']:.4f}" if r['token_acc_all_mean'] is not None else "  N/A  "
        acc_post = f"{r['token_acc_post_perc_mean']:.4f}" if r['token_acc_post_perc_mean'] is not None else "  N/A  "
        logger.info(
            f"{r['mode']:<8} {r['loss_ce_mean']:>12.4f} {delta:>+12.4f} {acc_all:>12} {acc_post:>12}"
        )
    logger.info("=" * 72)

    # Causal interpretation
    zero_delta  = results[1]["loss_ce_mean"] - normal["loss_ce_mean"]
    rand_delta  = results[2]["loss_ce_mean"] - normal["loss_ce_mean"]

    logger.info("\nInterpretation:")
    if zero_delta > 0.05 or rand_delta > 0.05:
        logger.info("  >> Loss degrades under z=zero/random → model IS causally using z_visual ✓")
    elif zero_delta > 0.01 or rand_delta > 0.01:
        logger.info("  >> Small loss increase → model uses z_visual weakly at this checkpoint.")
    else:
        logger.info("  >> No loss change → model may NOT be using z_visual yet.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "results": results,
        "delta_zero":   zero_delta,
        "delta_random": rand_delta,
    }
    out_path = out_dir / "z_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
