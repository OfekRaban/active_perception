#!/usr/bin/env python3
"""
Sanity check 2: Grounding loss end-to-end verification.

Checks:
  1. Patch mask from bbox is non-empty and covers correct patches
  2. Attention grid shape matches patch mask (same N = H*W)
  3. Grounding loss is finite (not NaN/inf)
  4. Gradients flow: perception_module params have non-zero grad after backward
  5. PERC_OUT embedding source: special_token_embeddings[2] (trainable Parameter),
     NOT frozen embedding table — verified by perturbation test

Usage:
    CUDA_VISIBLE_DEVICES=0 /cortex/users/rabanof/conda_envs/qwen49/bin/python \
        scripts/smoke_test_grounding.py \
        --config configs/exp1_ce_only.yaml \
        --checkpoint runs/exp1_ce_only/checkpoint-best
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator
from active_perception.training.losses import PerceptionLosses, LossConfig


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        for base_name in cfg.pop("defaults"):
            base_path = Path(path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as fb:
                    cfg = _deep_merge(yaml.safe_load(fb), cfg)
    return cfg


def _build_patch_mask_tensor(bbox, H, W, device):
    """Build [1, H*W] patch mask tensor from a normalized bbox."""
    x1, y1, x2, y2 = bbox
    cy = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    cx = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    row_mask = (cy >= y1) & (cy <= y2)   # [H]
    col_mask = (cx >= x1) & (cx <= x2)   # [W]
    grid = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)  # [H, W]
    return grid.flatten().float().unsqueeze(0)  # [1, H*W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp1_ce_only.yaml")
    parser.add_argument("--checkpoint", default="runs/exp1_ce_only/checkpoint-best")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)
    model_cfg = ActivePerceptionConfig(**{k: v for k, v in cfg.get("model", {}).items()
                                          if k in ActivePerceptionConfig.__dataclass_fields__})
    model = ActivePerceptionModel(model_cfg).to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_perception_module(str(ckpt))
        logger.info(f"Loaded checkpoint: {ckpt}")
    else:
        logger.warning(f"Checkpoint not found; using fresh weights")

    data_cfg = cfg.get("data", {})
    eval_path = data_cfg.get("eval_data_path") or data_cfg.get("data_path")
    special_ids = model.get_special_token_ids()
    dataset = ActivePerceptionDataset(
        data_path=eval_path,
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
    )
    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )

    merge_size = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)

    loss_cfg = LossConfig(
        use_grounding=True,
        lambda_ground=0.05,
        grounding_temperature=0.5,
    )
    d_model = model.base_model.config.hidden_size
    loss_fn = PerceptionLosses(d_model, loss_cfg).to(device)

    failures = []

    # ── Check 1–4: grounding loss end-to-end ─────────────────────────────────
    print(f"\n{'='*72}")
    print("CHECK 1-4: Grounding loss computation")
    print(f"{'='*72}")

    samples_with_bbox = 0
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        bboxes = [s.bbox for s in sample.perception_steps if s.has_bbox()]
        if bboxes:
            samples_with_bbox += 1
            if samples_with_bbox > args.n_samples:
                break

            item = dataset[i]
            batch = collator([item])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass (grad enabled to check backward)
            model.train()
            out = model.training_forward(**batch)

            attn_w = out["attn_weights_list"][0]  # [K, N] or None
            if attn_w is None:
                print(f"  Sample {i}: no attn_weights (no PERCEPTION tokens), skip")
                continue

            K, N = attn_w.shape
            grid_thw = batch["image_grid_thw"]
            _, H_pre, W_pre = [int(x) for x in grid_thw[0].cpu().tolist()]
            H, W = H_pre // merge_size, W_pre // merge_size

            print(f"\n  Sample {i}: K={K} K_bboxes={len(bboxes)} H={H} W={W} N={N}")

            # Check 1: patch mask non-empty
            mask_list = []
            for bbox in bboxes[:K]:
                m = _build_patch_mask_tensor(bbox, H, W, device)
                mask_list.append(m)
            if len(mask_list) < K:
                mask_list += [torch.zeros(1, H * W, device=device)] * (K - len(mask_list))
            patch_mask = torch.cat(mask_list, dim=0)  # [K, H*W]

            bbox_area = patch_mask.sum().item() / (K * N)
            if patch_mask.sum() == 0:
                failures.append(f"Sample {i}: patch_mask is all zeros (bbox outside grid?)")
                print(f"  FAIL: patch_mask is all zeros")
            else:
                print(f"  PASS: patch_mask non-empty  area={bbox_area:.4f}  shape={patch_mask.shape}")

            # Check 2: attention grid shape matches patch mask
            if attn_w.shape[-1] != patch_mask.shape[-1]:
                failures.append(f"Sample {i}: attn shape {attn_w.shape} != patch_mask shape {patch_mask.shape}")
                print(f"  FAIL: shape mismatch: attn={attn_w.shape}  mask={patch_mask.shape}")
            else:
                print(f"  PASS: attention grid shape matches patch mask: {attn_w.shape}")

            # Check 3: grounding loss is finite
            patch_masks_batch = [patch_mask]
            loss_out = loss_fn.compute(
                loss_ce=out["loss_ce"],
                z_perceptions=out["z_perceptions"],
                attn_weights_list=out["attn_weights_list"],
                patch_masks=patch_masks_batch,
            )
            ground_loss = loss_out.ground
            total_loss = loss_out.total
            if ground_loss is None:
                failures.append(f"Sample {i}: grounding loss is None (no valid pairs?)")
                print(f"  FAIL: grounding loss=None")
            elif not torch.isfinite(ground_loss):
                failures.append(f"Sample {i}: grounding loss={ground_loss.item()} is NaN/inf")
                print(f"  FAIL: grounding loss={ground_loss.item()} NaN/inf")
            else:
                print(f"  PASS: grounding loss={ground_loss.item():.6f} (finite)")

            # Check 4: gradients flow to perception_module
            model.zero_grad()
            total_loss.backward()
            perc_grads = [p.grad for p in model.perception_module.parameters()
                          if p.grad is not None]
            if not perc_grads:
                failures.append(f"Sample {i}: no gradients on perception_module params")
                print(f"  FAIL: no gradients on perception_module")
            else:
                max_grad = max(g.abs().max().item() for g in perc_grads)
                if max_grad == 0:
                    failures.append(f"Sample {i}: all perception_module gradients are zero")
                    print(f"  FAIL: all perception_module gradients are zero")
                else:
                    print(f"  PASS: perception_module grad max_abs={max_grad:.6e}")

            # Also check special_token_embeddings grad
            ste_grad = model.special_token_embeddings.grad
            if ste_grad is not None:
                print(f"  INFO: special_token_embeddings.grad max_abs={ste_grad.abs().max().item():.4e}")
            else:
                print(f"  INFO: special_token_embeddings.grad=None (may be fine if no PERC_OUT in labels)")

    if samples_with_bbox == 0:
        failures.append("No samples with bboxes found in eval set — grounding checks skipped")
        print("  WARN: No samples with bboxes found")

    # ── Check 5: PERC_OUT embedding source ────────────────────────────────────
    print(f"\n{'='*72}")
    print("CHECK 5: PERC_OUT embedding source (special_token_embeddings[2] vs frozen table)")
    print(f"{'='*72}")

    model.eval()
    perc_out_id = model.special_tokens.PERC_OUT
    embed_table = model.base_model.get_input_embeddings()
    frozen_row = embed_table.weight.data[perc_out_id].clone()  # [D]
    ste_row = model.special_token_embeddings[2].data.clone()   # [D]

    # After training, these should differ (z_init was trained; table was frozen)
    cosine = F.cosine_similarity(
        frozen_row.float().unsqueeze(0),
        ste_row.float().unsqueeze(0),
    ).item()
    print(f"  cosine(frozen_table[PERC_OUT], special_token_emb[2]) = {cosine:.4f}")

    # Perturbation test: modify special_token_embeddings[2] and check output changes
    ids = torch.tensor([[perc_out_id]], dtype=torch.long, device=device)
    emb_before = model._embed(ids).detach().clone()

    with torch.no_grad():
        model.special_token_embeddings.data[2] += 1.0

    emb_after = model._embed(ids).detach().clone()

    with torch.no_grad():
        model.special_token_embeddings.data[2] -= 1.0  # restore

    diff = (emb_after - emb_before).abs().max().item()
    if diff < 1e-6:
        failures.append("PERC_OUT embedding NOT coming from special_token_embeddings[2] (perturbation had no effect)")
        print(f"  FAIL: perturbation of special_token_embeddings[2] had no effect (diff={diff:.2e})")
    else:
        print(f"  PASS: perturbation of special_token_embeddings[2] changes output (diff={diff:.4f})")
        print(f"        => PERC_OUT embedding IS sourced from trainable special_token_embeddings[2]")

    # Verify frozen table was NOT changed by perturbation
    frozen_row_after = embed_table.weight.data[perc_out_id].clone()
    table_diff = (frozen_row_after - frozen_row).abs().max().item()
    if table_diff > 1e-6:
        failures.append(f"Embedding table row for PERC_OUT changed (table_diff={table_diff:.2e}) — table not frozen!")
        print(f"  FAIL: embedding table row changed (diff={table_diff:.2e}) — table should be frozen!")
    else:
        print(f"  PASS: frozen embedding table unchanged (diff={table_diff:.2e})")

    # Print norms
    print(f"  frozen_table[PERC_OUT] norm = {frozen_row.float().norm().item():.4f}")
    print(f"  special_token_emb[2]   norm = {ste_row.float().norm().item():.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    if failures:
        print(f"SMOKE TEST 2 FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST 2 PASSED — all grounding + embedding checks OK")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
