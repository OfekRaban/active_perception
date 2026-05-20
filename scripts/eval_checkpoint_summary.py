#!/usr/bin/env python3
"""
Unified per-checkpoint evaluation summary for ablation runs.

For a given run directory + checkpoint tag, reports:
  1. Training metrics from train_metrics.jsonl
       - best eval loss, final train loss, final entropy, overfitting gap
  2. z-ablation (normal / zero / random z_visual, 50 eval samples)
       - CE loss delta, post-PERC_OUT accuracy
  3. Attention grounding (50 eval samples)
       - grounding_ratio, attn_entropy
  4. Verdict: is the model optimization-limited or learning something?

Usage:
    CUDA_VISIBLE_DEVICES=0 /cortex/users/rabanof/conda_envs/qwen49/bin/python \\
        scripts/eval_checkpoint_summary.py \\
        --run_dir runs/phase1_latent \\
        --config configs/phase1_latent.yaml \\
        --checkpoint_tag best \\
        --num_samples 50 \\
        --output_dir outputs/ablation
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from active_perception.data.utils import resolve_image_path
from PIL import Image


# ── Config loading ─────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        for base_name in cfg.pop("defaults"):
            base_path = Path(config_path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as fb:
                    cfg = _deep_merge(yaml.safe_load(fb), cfg)
    return cfg


# ── 1. Training log parsing ────────────────────────────────────────────────────

def summarize_training_log(run_dir: Path) -> Dict:
    """
    Parse train_metrics.jsonl.  Returns:
      best_eval_loss, final_train_loss, final_step,
      entropy_start, entropy_end, entropy_delta, eval_curve, loss_curve
    """
    jsonl = run_dir / "train_metrics.jsonl"
    if not jsonl.exists():
        return {"error": "train_metrics.jsonl not found"}

    train_losses, eval_losses, entropies = [], [], []

    with open(jsonl) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            step = d.get("step")
            if "loss/ce" in d and step is not None:
                train_losses.append((step, d["loss/ce"]))
            if "eval/loss" in d and step is not None:
                eval_losses.append((step, d["eval/loss"]))
            if "diag/attn_entropy_mean" in d and step is not None:
                entropies.append((step, d["diag/attn_entropy_mean"]))

    best_eval = min((v for _, v in eval_losses), default=None)
    final_train = train_losses[-1][1] if train_losses else None
    final_step = train_losses[-1][0] if train_losses else None
    ent_start = entropies[0][1] if entropies else None
    ent_end   = entropies[-1][1] if entropies else None
    ent_delta = (ent_end - ent_start) if (ent_start and ent_end) else None
    overfit_gap = (final_train - best_eval) if (final_train and best_eval) else None

    return {
        "best_eval_loss": best_eval,
        "final_train_loss": final_train,
        "final_step": final_step,
        "entropy_start": ent_start,
        "entropy_end": ent_end,
        "entropy_delta": ent_delta,
        "overfit_gap": overfit_gap,  # negative = train < eval = overfitting
        "eval_curve": eval_losses,
        "entropy_curve": entropies,
    }


# ── 2. z-visual ablation ───────────────────────────────────────────────────────

@contextlib.contextmanager
def z_override_mode(model: ActivePerceptionModel, mode: str):
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
            z_norm = z.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            n_norm = noise.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return (noise / n_norm * z_norm).to(z.dtype), attn
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.perception_module.forward = patched
    try:
        yield
    finally:
        model.perception_module.forward = original_forward


def _post_perc_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    perc_out_id: int,
) -> Optional[float]:
    pred    = logits[:, :-1, :].argmax(-1)
    targets = labels[:, 1:]
    ids_shift = input_ids[:, :-1]
    post_mask = (ids_shift == perc_out_id) & (targets != -100)
    if post_mask.sum() == 0:
        return None
    return (pred[post_mask] == targets[post_mask]).float().mean().item()


def run_z_ablation(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
    num_samples: int,
) -> Dict:
    perc_out_id = model.special_tokens.PERC_OUT
    n = min(num_samples, len(dataset))
    results = {}

    for mode in ("normal", "zero", "random"):
        losses, post_accs = [], []
        for i in range(n):
            item  = dataset[i]
            batch = collator([item])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.no_grad(), z_override_mode(model, mode):
                out = model.training_forward(**batch)
            losses.append(out["loss_ce"].item())
            if out["attn_weights_list"][0] is not None:
                acc = _post_perc_accuracy(
                    out["logits"], out["modified_labels"],
                    out["modified_input_ids"], perc_out_id,
                )
                if acc is not None:
                    post_accs.append(acc)

        results[mode] = {
            "loss_ce_mean": float(np.mean(losses)),
            "loss_ce_std":  float(np.std(losses)),
            "post_perc_acc_mean": float(np.mean(post_accs)) if post_accs else None,
            "n_samples": n,
            "n_with_post_perc": len(post_accs),
        }
        acc_s = f"{results[mode]['post_perc_acc_mean']:.4f}" if results[mode]['post_perc_acc_mean'] is not None else "N/A"
        logger.info(
            f"  z-ablation [{mode:6s}] "
            f"loss={results[mode]['loss_ce_mean']:.4f} ± {results[mode]['loss_ce_std']:.4f}  "
            f"acc_post={acc_s}"
        )

    normal_loss = results["normal"]["loss_ce_mean"]
    results["delta_zero"]   = results["zero"]["loss_ce_mean"]   - normal_loss
    results["delta_random"] = results["random"]["loss_ce_mean"] - normal_loss
    return results


# ── 3. Grounding ratio ─────────────────────────────────────────────────────────

def _bbox_to_patch_mask(bbox: List[float], H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cy = (np.arange(H) + 0.5) / H
    cx = (np.arange(W) + 0.5) / W
    return np.outer((cy >= y1) & (cy <= y2), (cx >= x1) & (cx <= x2)).flatten().astype(np.float32)


def run_grounding(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
    num_samples: int,
) -> Dict:
    merge_size = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)
    n = min(num_samples, len(dataset))

    entropies, bbox_masses, bbox_areas = [], [], []

    for i in range(n):
        item  = dataset[i]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            out = model.training_forward(**batch)

        attn_w = out["attn_weights_list"][0]
        if attn_w is None:
            continue

        grid_thw = batch.get("image_grid_thw")
        if grid_thw is None:
            continue
        _, H_pre, W_pre = [int(x) for x in grid_thw[0].cpu().tolist()]
        H, W = H_pre // merge_size, W_pre // merge_size
        if attn_w.shape[-1] != H * W:
            continue

        attn_np = attn_w.float().cpu().numpy()  # [K, N]
        K = attn_np.shape[0]

        raw = dataset.samples[i]
        bboxes = [s.bbox for s in raw.perception_steps if s.has_bbox()]
        bboxes = (bboxes + [None] * K)[:K]

        for k in range(K):
            vec = attn_np[k]
            entropies.append(float(-np.sum(vec * np.log(vec + 1e-9))))
            if bboxes[k] is not None:
                mask = _bbox_to_patch_mask(bboxes[k], H, W)
                bbox_masses.append(float((vec * mask).sum()))
                bbox_areas.append(float(mask.sum() / len(mask)))

    grounding_ratio = None
    if bbox_masses and bbox_areas:
        ratios = [m / max(a, 1e-6) for m, a in zip(bbox_masses, bbox_areas)]
        grounding_ratio = float(np.mean(ratios))

    result = {
        "n_samples": n,
        "n_with_attn": len(entropies),
        "attn_entropy_mean": float(np.mean(entropies)) if entropies else None,
        "attn_entropy_std":  float(np.std(entropies))  if entropies else None,
        "bbox_attn_mass_mean": float(np.mean(bbox_masses)) if bbox_masses else None,
        "bbox_area_mean":      float(np.mean(bbox_areas))  if bbox_areas  else None,
        "grounding_ratio": grounding_ratio,
    }
    gr_s  = f"{grounding_ratio:.3f}" if grounding_ratio is not None else "N/A"
    ent_s = f"{result['attn_entropy_mean']:.4f}" if result['attn_entropy_mean'] is not None else "N/A"
    logger.info(f"  Grounding: ratio={gr_s}  entropy={ent_s}")
    return result


# ── 4. Generation accuracy ────────────────────────────────────────────────────

def run_generation_eval(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    device: torch.device,
    num_samples: int,
) -> Dict:
    """
    Autoregressive generation accuracy on final answer (no teacher forcing).

    Builds prompt-only inputs, runs generate_with_perception, decodes the
    result, and checks whether the GT converted_answer appears in the output.
    Reports substring-match accuracy (lenient but fast).
    """
    n = min(num_samples, len(dataset))
    n_correct = 0
    n_valid = 0
    n_skipped = 0

    for i in range(n):
        sample = dataset.samples[i]
        gt = (sample.converted_answer or "").strip()
        if not gt:
            n_skipped += 1
            continue

        # Build prompt-only inputs with generation template
        prompt_msgs = dataset._build_prompt_messages(sample)
        prompt_text = model.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        try:
            image = dataset._load_image(sample.image)
            if image is None:
                n_skipped += 1
                continue
            enc = model.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
        except Exception as e:
            logger.warning(f"  [GenEval] sample {i} tokenization failed: {e}")
            n_skipped += 1
            continue

        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        pv = enc.get("pixel_values")
        if pv is not None:
            pv = pv.to(device)
        gthw = enc.get("image_grid_thw")
        if gthw is not None:
            gthw = gthw.to(device)

        try:
            with torch.no_grad():
                gen_ids = model.generate_with_perception(
                    input_ids=input_ids,
                    pixel_values=pv,
                    image_grid_thw=gthw,
                    attention_mask=attn_mask,
                    max_new_tokens=256,
                    do_sample=False,
                )
        except Exception as e:
            logger.warning(f"  [GenEval] sample {i} generation failed: {e}")
            n_skipped += 1
            continue

        gen_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if gt.lower() in gen_text.lower():
            n_correct += 1
        n_valid += 1

    acc = n_correct / max(n_valid, 1)
    logger.info(
        f"  Generation eval: n={n}  valid={n_valid}  skipped={n_skipped}  "
        f"correct={n_correct}  acc={acc:.4f}"
    )
    return {
        "n_samples": n,
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "n_correct": n_correct,
        "accuracy": acc,
    }


# ── 5. Qualitative examples ────────────────────────────────────────────────────

def run_qualitative_examples(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    device: torch.device,
    n: int = 10,
) -> List[Dict]:
    """
    For n samples, generate under normal / zero / random z and return examples.

    Each example dict has:
      sample_id, question, gt_answer, normal_gen, zero_gen, random_gen
    """
    examples = []
    n = min(n, len(dataset))

    for i in range(n):
        sample = dataset.samples[i]
        prompt_msgs = dataset._build_prompt_messages(sample)
        prompt_text = model.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        try:
            image = dataset._load_image(sample.image)
            if image is None:
                continue
            enc = model.processor(
                text=[prompt_text], images=[image],
                return_tensors="pt", padding=False,
            )
        except Exception:
            continue

        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        pv = enc.get("pixel_values")
        if pv is not None:
            pv = pv.to(device)
        gthw = enc.get("image_grid_thw")
        if gthw is not None:
            gthw = gthw.to(device)

        gens = {}
        for mode in ("normal", "zero", "random"):
            try:
                with torch.no_grad(), z_override_mode(model, mode):
                    gen_ids = model.generate_with_perception(
                        input_ids=input_ids,
                        pixel_values=pv,
                        image_grid_thw=gthw,
                        attention_mask=attn_mask,
                        max_new_tokens=256,
                        do_sample=False,
                    )
                gens[mode] = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            except Exception as e:
                gens[mode] = f"[ERROR: {e}]"

        ex = {
            "sample_id": sample.id,
            "question": sample.question[:200],
            "gt_answer": (sample.converted_answer or "")[:200],
            "normal_gen": gens.get("normal", "")[:400],
            "zero_gen":   gens.get("zero",   "")[:400],
            "random_gen": gens.get("random", "")[:400],
        }
        examples.append(ex)

        logger.info(f"  [Qualitative {i+1}/{n}] sample={sample.id}")
        logger.info(f"    Q:      {ex['question'][:100]}")
        logger.info(f"    GT:     {ex['gt_answer'][:100]}")
        logger.info(f"    Normal: {ex['normal_gen'][:100]}")
        logger.info(f"    Zero:   {ex['zero_gen'][:100]}")
        logger.info(f"    Random: {ex['random_gen'][:100]}")

    return examples


# ── 6. Verdict ─────────────────────────────────────────────────────────────────

def build_verdict(log_summary: Dict, z_abl: Dict, grounding: Dict) -> str:
    lines = []

    # Overfitting
    gap = log_summary.get("overfit_gap")
    if gap is not None:
        if gap < -0.3:
            lines.append(f"OVERFITS: train-eval gap={gap:+.3f} (train much lower than eval)")
        elif gap < -0.05:
            lines.append(f"WEAK OVERFIT: train-eval gap={gap:+.3f}")
        else:
            lines.append(f"NO OVERFIT: train-eval gap={gap:+.3f} — optimization-limited or plateau")

    # z-ablation
    dz = z_abl.get("delta_zero", 0)
    dr = z_abl.get("delta_random", 0)
    if dz > 0.1 or dr > 0.05:
        lines.append(f"z CAUSAL: loss degrades under zero ({dz:+.3f}) / random ({dr:+.3f})")
    else:
        lines.append(f"z WEAK/NONE: delta_zero={dz:+.3f}  delta_random={dr:+.3f}")

    # Grounding
    gr = grounding.get("grounding_ratio")
    if gr is not None:
        if gr > 1.5:
            lines.append(f"GROUNDING: moderate-strong (ratio={gr:.3f})")
        elif gr > 1.1:
            lines.append(f"GROUNDING: weak (ratio={gr:.3f})")
        else:
            lines.append(f"GROUNDING: random (ratio={gr:.3f})")

    # Entropy
    ed = log_summary.get("entropy_delta")
    if ed is not None:
        lines.append(f"ENTROPY: {'focusing ↓' if ed < 0 else 'spreading ↑'} delta={ed:+.3f}")

    return " | ".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",         required=True)
    parser.add_argument("--config",          required=True)
    parser.add_argument("--checkpoint_tag",  default="best")
    parser.add_argument("--num_samples",     type=int, default=50)
    parser.add_argument("--gen_samples",     type=int, default=100,
                        help="Number of samples for autoregressive generation eval (0 to skip)")
    parser.add_argument("--qualitative_n",   type=int, default=10,
                        help="Number of qualitative normal/zero/random examples (0 to skip)")
    parser.add_argument("--output_dir",      default="outputs/ablation")
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")

    checkpoint = str(run_dir / f"checkpoint-{args.checkpoint_tag}")

    # ── 1. Training log ───────────────────────────────────────────────────────
    logger.info(f"\n[1/3] Parsing training log: {run_dir}/train_metrics.jsonl")
    log_summary = summarize_training_log(run_dir)
    logger.info(
        f"  best_eval={log_summary.get('best_eval_loss'):.4f}  "
        f"final_train={log_summary.get('final_train_loss'):.4f}  "
        f"step={log_summary.get('final_step')}  "
        f"gap={log_summary.get('overfit_gap'):+.4f}"
    )

    # ── Load model (once) ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_cfg = ActivePerceptionConfig(**{
        k: v for k, v in cfg.get("model", {}).items()
        if k in ActivePerceptionConfig.__dataclass_fields__
    })
    logger.info(f"\nLoading model (mode={model_cfg.initial_perception_mode}) ...")
    model = ActivePerceptionModel(model_cfg)
    model.load_perception_module(checkpoint)
    model = model.to(device)
    model.eval()
    logger.info(f"  Checkpoint: {checkpoint}")

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
    logger.info(f"  Eval dataset: {len(dataset)} samples (using {args.num_samples})")

    # ── 2. z-ablation ─────────────────────────────────────────────────────────
    logger.info(f"\n[2/4] z-visual ablation ({args.num_samples} samples) ...")
    z_abl = run_z_ablation(model, dataset, collator, device, args.num_samples)

    # ── 3. Grounding ──────────────────────────────────────────────────────────
    logger.info(f"\n[3/4] Grounding + entropy ({args.num_samples} samples) ...")
    grounding = run_grounding(model, dataset, collator, device, args.num_samples)

    # ── 4. Generation accuracy ────────────────────────────────────────────────
    gen_eval = {}
    if args.gen_samples > 0:
        logger.info(f"\n[4a/4] Generation accuracy ({args.gen_samples} samples, autoregressive) ...")
        gen_eval = run_generation_eval(model, dataset, device, args.gen_samples)
    else:
        logger.info(f"\n[4a/4] Generation eval skipped (--gen_samples 0)")

    qualitative = []
    if args.qualitative_n > 0:
        logger.info(f"\n[4b/4] Qualitative examples ({args.qualitative_n} samples × 3 modes) ...")
        qualitative = run_qualitative_examples(model, dataset, device, args.qualitative_n)
    else:
        logger.info(f"\n[4b/4] Qualitative examples skipped (--qualitative_n 0)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = build_verdict(log_summary, z_abl, grounding)

    # ── Summary table ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info(f"SUMMARY: {run_name}  [{args.checkpoint_tag}]")
    logger.info("=" * 72)

    be  = log_summary.get("best_eval_loss")
    ft  = log_summary.get("final_train_loss")
    gap = log_summary.get("overfit_gap")
    ee  = log_summary.get("entropy_end")
    ed  = log_summary.get("entropy_delta")
    gr  = grounding.get("grounding_ratio")
    dz  = z_abl.get("delta_zero")
    dr  = z_abl.get("delta_random")
    pa_n = z_abl.get("normal", {}).get("post_perc_acc_mean")
    pa_z = z_abl.get("zero",   {}).get("post_perc_acc_mean")

    def fmt(v, spec=".4f"):
        return f"{v:{spec}}" if v is not None else "N/A"

    logger.info(f"  best_eval_loss   : {fmt(be)}")
    logger.info(f"  final_train_loss : {fmt(ft)}")
    logger.info(f"  overfit_gap      : {fmt(gap, '+.4f')}  (neg = train << eval)")
    logger.info(f"  entropy_end      : {fmt(ee)}  delta={fmt(ed, '+.4f')}")
    logger.info(f"  grounding_ratio  : {fmt(gr)}")
    logger.info(f"  z_delta_zero     : {fmt(dz, '+.4f')}")
    logger.info(f"  z_delta_random   : {fmt(dr, '+.4f')}")
    logger.info(f"  post_perc_acc    : normal={fmt(pa_n)}  zero={fmt(pa_z)}")
    gen_acc = gen_eval.get("accuracy")
    logger.info(f"  gen_accuracy     : {fmt(gen_acc)}")
    logger.info(f"\n  VERDICT: {verdict}")
    logger.info("=" * 72)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "checkpoint_tag": args.checkpoint_tag,
        "initial_perception_mode": model_cfg.initial_perception_mode,
        "num_samples": args.num_samples,
        "training_log": {k: v for k, v in log_summary.items()
                         if k not in ("eval_curve", "entropy_curve")},
        "z_ablation": z_abl,
        "grounding": grounding,
        "generation_eval": gen_eval,
        "qualitative_examples": qualitative,
        "verdict": verdict,
    }
    out_path = out_dir / f"{run_name}_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
