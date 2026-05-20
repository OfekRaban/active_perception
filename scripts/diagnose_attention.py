#!/usr/bin/env python3
"""
Diagnostic pipeline for a trained Active Perception checkpoint.

Evaluates:
  [1] Attention grounding: does attention align with GT bbox regions?
  [2] Gradient verification: does PerceptionModule receive non-zero gradients?
  [3] Entropy curve: is attention sharpening over training?
  [4] Architectural check: is injected z truly different from the static PERC_OUT embedding?

Usage:
    CUDA_VISIBLE_DEVICES=1 /cortex/users/rabanof/conda_envs/qwen49/bin/python \\
        scripts/diagnose_attention.py \\
        --config configs/exp1_ce_only.yaml \\
        --checkpoint runs/exp1_ce_only/checkpoint-best \\
        --train_log runs/exp1_ce_only/train_metrics.jsonl \\
        --output_dir outputs/diagnostics \\
        --num_samples 100
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

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


# ── Spatial utilities ─────────────────────────────────────────────────────────

def bbox_to_patch_mask(
    bbox: List[float],
    H: int,
    W: int,
) -> np.ndarray:
    """
    Convert normalized bbox [x1,y1,x2,y2] to binary patch mask of shape [H*W].
    Patch center (r,c) → (cx=(c+.5)/W, cy=(r+.5)/H).
    """
    x1, y1, x2, y2 = bbox
    r_idx = np.arange(H)
    c_idx = np.arange(W)
    cy = (r_idx + 0.5) / H   # [H]
    cx = (c_idx + 0.5) / W   # [W]
    mask_r = (cy >= y1) & (cy <= y2)
    mask_c = (cx >= x1) & (cx <= x2)
    return np.outer(mask_r, mask_c).flatten().astype(np.float32)


def attn_entropy(weights: np.ndarray) -> float:
    return float(-np.sum(weights * np.log(weights + 1e-9)))


def overlay_and_draw(
    img: Image.Image,
    heatmap_hw: np.ndarray,
    bbox_normalized: Optional[List[float]] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay normalized [H,W] heatmap on image, optionally draw GT bbox."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    hm_rgba = cmap(heatmap_hw)
    hm_rgb = (hm_rgba[:, :, :3] * 255).astype(np.uint8)
    hm_pil = Image.fromarray(hm_rgb, "RGB").resize(img.size, Image.BILINEAR)
    overlay = Image.blend(img.convert("RGB"), hm_pil, alpha)

    if bbox_normalized is not None:
        x1, y1, x2, y2 = bbox_normalized
        W_img, H_img = overlay.size
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [x1 * W_img, y1 * H_img, x2 * W_img, y2 * H_img],
            outline=(0, 255, 0),
            width=max(2, W_img // 200),
        )
    return overlay


# ── [1] Attention diagnostics ─────────────────────────────────────────────────

def run_attention_diagnostics(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
    num_samples: int,
    heatmap_dir: Path,
) -> List[Dict]:
    """
    For each eval sample:
      - Run training_forward (no_grad)
      - Compute per-query: entropy, bbox_attn_mass, top10_overlap
      - Save heatmap PNG with GT bbox drawn
    Returns list of per-sample metric dicts.
    """
    merge_size = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    n = min(num_samples, len(dataset))

    for sample_idx in range(n):
        if sample_idx % 10 == 0:
            logger.info(f"  Diagnostics: sample {sample_idx}/{n}")

        item = dataset[sample_idx]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model.training_forward(**batch)

        loss_ce = out["loss_ce"].item()
        attn_w = out["attn_weights_list"][0]  # [K, N] or None

        raw_sample = dataset.samples[sample_idx]
        img_path = resolve_image_path(raw_sample.image)
        orig_image = None
        if img_path is not None:
            try:
                orig_image = Image.open(img_path).convert("RGB")
            except Exception:
                pass

        image_grid_thw = batch.get("image_grid_thw")
        grid_thw = image_grid_thw[0].cpu().tolist() if image_grid_thw is not None else None

        if attn_w is None or grid_thw is None:
            all_metrics.append({
                "sample_idx": sample_idx,
                "sample_id": raw_sample.id,
                "loss_ce": loss_ce,
                "has_attn": False,
                "skip_reason": "no_attn_or_grid",
            })
            continue

        _, H_pre, W_pre = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        H = H_pre // merge_size
        W = W_pre // merge_size
        N_expected = H * W
        N_actual = attn_w.shape[-1]

        grid_ok = (N_actual == N_expected)
        if not grid_ok:
            all_metrics.append({
                "sample_idx": sample_idx,
                "sample_id": raw_sample.id,
                "loss_ce": loss_ce,
                "has_attn": True,
                "skip_reason": f"grid_mismatch_N{N_actual}_expected{N_expected}",
            })
            continue

        attn_w_np = attn_w.float().cpu().numpy()  # [K, N]
        K = attn_w_np.shape[0]

        # GT bboxes: one per perception step
        bboxes = [s.bbox for s in raw_sample.perception_steps if s.has_bbox()]
        # Extend or truncate to match K
        bboxes_aligned = (bboxes + [None] * K)[:K]

        per_query = []
        for k in range(K):
            vec = attn_w_np[k]
            ent = attn_entropy(vec)
            hmap = vec.reshape(H, W)
            mn, mx = hmap.min(), hmap.max()
            hmap_norm = (hmap - mn) / (mx - mn) if mx > mn else np.zeros_like(hmap)

            top10_idx = np.argsort(vec)[::-1][:10].tolist()
            bbox = bboxes_aligned[k]

            bbox_attn_mass = None
            top10_overlap_count = 0
            bbox_area = None

            if bbox is not None:
                patch_mask = bbox_to_patch_mask(bbox, H, W)
                bbox_attn_mass = float((vec * patch_mask).sum())
                bbox_area = float(patch_mask.sum() / len(patch_mask))
                top10_overlap_count = int(patch_mask[top10_idx].sum())

            # Save heatmap
            png_name = f"sample_{sample_idx:04d}_q{k}.png"
            if orig_image is not None:
                vis = overlay_and_draw(orig_image, hmap_norm, bbox)
                vis.save(heatmap_dir / png_name)

            per_query.append({
                "query_idx": k,
                "entropy": ent,
                "bbox_attn_mass": bbox_attn_mass,
                "bbox_area": bbox_area,
                "top10_overlap_count": top10_overlap_count,
                "top10_indices": top10_idx,
                "top10_weights": vec[top10_idx].tolist(),
                "heatmap_shape": [H, W],
                "N_patches": int(N_actual),
            })

        all_metrics.append({
            "sample_idx": sample_idx,
            "sample_id": raw_sample.id,
            "loss_ce": loss_ce,
            "has_attn": True,
            "num_queries": K,
            "per_query": per_query,
        })

    return all_metrics


# ── [2] Gradient verification ─────────────────────────────────────────────────

def run_gradient_check(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
) -> Dict:
    """
    Run one forward+backward pass on a single sample and record
    PerceptionModule parameter gradient norms.
    """
    logger.info("  Running gradient verification...")
    item = dataset[0]
    batch = collator([item])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model.train()
    model.zero_grad()

    import torch.cuda.amp as amp
    with amp.autocast(dtype=torch.bfloat16):
        out = model.training_forward(**batch)

    out["loss_ce"].backward()

    results = {}

    # Total perception_module grad norm
    pm_norms = []
    for name, param in model.perception_module.named_parameters():
        if param.grad is not None:
            g = param.grad.float().norm().item()
            results[f"pm/{name}"] = g
            pm_norms.append(g)
    results["pm/total_norm"] = float(np.sqrt(sum(g**2 for g in pm_norms))) if pm_norms else 0.0

    # Special token embeddings grad
    if model.special_token_embeddings.grad is not None:
        results["special_token_embeddings_norm"] = model.special_token_embeddings.grad.float().norm().item()
    else:
        results["special_token_embeddings_norm"] = 0.0

    results["any_nonzero"] = any(v > 0 for k, v in results.items() if isinstance(v, float))
    results["loss_ce"] = out["loss_ce"].item()

    model.eval()
    model.zero_grad()

    logger.info(
        f"  Grad check: pm_total_norm={results['pm/total_norm']:.6f}  "
        f"spe_emb_norm={results['special_token_embeddings_norm']:.6f}  "
        f"any_nonzero={results['any_nonzero']}"
    )
    return results


# ── [4] Architectural check: static vs injected z ─────────────────────────────

def run_arch_check(
    model: ActivePerceptionModel,
    dataset: ActivePerceptionDataset,
    collator: ActivePerceptionCollator,
    device: torch.device,
    num_samples: int = 8,
) -> List[Dict]:
    """
    Compare the static PERC_OUT embedding vs the z_perception injected at runtime.
    High cosine similarity → model barely changes the token; low → true visual latent.
    """
    logger.info("  Running architectural check (static vs injected z)...")

    # Static PERC_OUT embedding: index 2 in [IMAGE, PERCEPTION, PERC_OUT, INIT_PERC_OUT]
    static_emb = model.special_token_embeddings[2].float().detach()  # [D]

    results = []
    n = min(num_samples, len(dataset))

    for i in range(n):
        item = dataset[i]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model.training_forward(**batch)

        z_list = out["z_perceptions"]
        if z_list[0] is None:
            continue

        z = z_list[0].float()  # [K, D]

        for k in range(z.shape[0]):
            z_k = z[k]  # [D]
            cos = F.cosine_similarity(z_k.unsqueeze(0), static_emb.unsqueeze(0)).item()
            z_norm = z_k.norm().item()
            static_norm = static_emb.norm().item()
            results.append({
                "sample_idx": i,
                "query_idx": k,
                "cosine_static_vs_z": cos,
                "z_norm": z_norm,
                "static_emb_norm": static_norm,
            })

    if results:
        cos_vals = [r["cosine_static_vs_z"] for r in results]
        z_norms = [r["z_norm"] for r in results]
        logger.info(
            f"  Arch check: cosine(static, z): mean={np.mean(cos_vals):.4f} "
            f"std={np.std(cos_vals):.4f}  z_norm: mean={np.mean(z_norms):.2f}"
        )

    return results


# ── [3] Entropy curve ─────────────────────────────────────────────────────────

def build_entropy_curve(train_log_path: str, output_dir: Path) -> Dict:
    """
    Parse train_metrics.jsonl, extract entropy entries, plot entropy vs step.
    Returns dict with steps and values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, entropies, losses = [], [], []

    with open(train_log_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "diag/attn_entropy_mean" in d and "step" in d:
                steps.append(d["step"])
                entropies.append(d["diag/attn_entropy_mean"])
            if "loss/ce" in d and "step" in d:
                losses.append((d["step"], d["loss/ce"]))

    if not steps:
        logger.warning("  No entropy entries found in train log.")
        return {"steps": [], "entropy": []}

    # Smooth loss with 20-step rolling mean for overlay
    loss_steps = [x[0] for x in losses]
    loss_vals  = [x[1] for x in losses]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    color1 = "#e06c75"
    color2 = "#61afef"

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Attn entropy (nats)", color=color1)
    ax1.plot(steps, entropies, "o-", color=color1, markersize=5, label="attn entropy")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("CE loss (train)", color=color2)
    if loss_vals:
        window = 20
        smoothed = np.convolve(loss_vals, np.ones(window) / window, mode="valid")
        ax2.plot(
            loss_steps[window - 1:], smoothed,
            color=color2, alpha=0.6, linewidth=1.2, label="CE loss (smooth)"
        )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Attention entropy vs training step (exp1_ce_only)")
    fig.tight_layout()
    out_path = output_dir / "entropy_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved entropy curve: {out_path}")

    # Save entropy history as JSON
    entropy_data = {"steps": steps, "entropy": entropies}
    json_path = output_dir / "entropy_history.json"
    with open(json_path, "w") as f:
        json.dump(entropy_data, f, indent=2)

    logger.info(
        f"  Entropy: start={entropies[0]:.3f} (step {steps[0]})  "
        f"end={entropies[-1]:.3f} (step {steps[-1]})  "
        f"delta={entropies[-1] - entropies[0]:+.3f}"
    )
    return entropy_data


# ── Aggregate metrics helpers ─────────────────────────────────────────────────

def aggregate_attn_metrics(all_metrics: List[Dict]) -> Dict:
    entropies, bbox_masses, bbox_areas, top10_overlaps = [], [], [], []
    losses = []

    for m in all_metrics:
        if not m.get("has_attn"):
            continue
        losses.append(m["loss_ce"])
        for q in m.get("per_query", []):
            entropies.append(q["entropy"])
            if q["bbox_attn_mass"] is not None:
                bbox_masses.append(q["bbox_attn_mass"])
                bbox_areas.append(q["bbox_area"])
                top10_overlaps.append(q["top10_overlap_count"])

    def stats(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    grounding_ratio = None
    if bbox_masses and bbox_areas:
        ratios = [m / max(a, 1e-6) for m, a in zip(bbox_masses, bbox_areas)]
        grounding_ratio = float(np.mean(ratios))

    return {
        "n_samples_with_attn": sum(1 for m in all_metrics if m.get("has_attn")),
        "n_skipped": sum(1 for m in all_metrics if not m.get("has_attn")),
        "ce_loss": stats(losses),
        "attn_entropy": stats(entropies),
        "bbox_attn_mass": stats(bbox_masses),
        "bbox_area": stats(bbox_areas),
        "grounding_ratio": grounding_ratio,
        "top10_overlap_count": stats(top10_overlaps),
        "top10_overlap_rate": float(np.mean([c / 10.0 for c in top10_overlaps])) if top10_overlaps else None,
    }


# ── [7] Report generation ─────────────────────────────────────────────────────

def generate_report(
    agg: Dict,
    arch_results: List[Dict],
    grad_results: Dict,
    entropy_data: Dict,
    checkpoint: str,
    output_dir: Path,
    num_samples: int,
) -> None:
    cos_vals = [r["cosine_static_vs_z"] for r in arch_results]
    z_norms  = [r["z_norm"] for r in arch_results]

    ent = agg["attn_entropy"]
    mass = agg["bbox_attn_mass"]
    area = agg["bbox_area"]
    gr = agg.get("grounding_ratio")
    ol = agg.get("top10_overlap_rate")

    steps = entropy_data.get("steps", [])
    entropies = entropy_data.get("entropy", [])
    ent_start = f"{entropies[0]:.3f} @ step {steps[0]}" if entropies else "N/A"
    ent_end   = f"{entropies[-1]:.3f} @ step {steps[-1]}" if entropies else "N/A"
    ent_delta = f"{entropies[-1] - entropies[0]:+.3f}" if entropies else "N/A"

    # Simple grounding verdict
    if gr is not None:
        if gr > 1.5:
            grounding_verdict = "MODERATE-STRONG (model attends inside bbox more than random)"
        elif gr > 1.1:
            grounding_verdict = "WEAK-MODERATE (slight preference for bbox region)"
        else:
            grounding_verdict = "RANDOM / NO GROUNDING (attention not concentrated in bbox)"
    else:
        grounding_verdict = "UNKNOWN (no bbox data)"

    # Injection verdict
    if cos_vals:
        mean_cos = float(np.mean(cos_vals))
        if mean_cos < 0.5:
            injection_verdict = "CONFIRMED: injected z differs strongly from static embedding"
        elif mean_cos < 0.8:
            injection_verdict = "PARTIAL: injected z is somewhat different from static embedding"
        else:
            injection_verdict = "WEAK: injected z is close to static embedding — check injection logic"
    else:
        injection_verdict = "UNKNOWN"
        mean_cos = None

    grad_ok = grad_results.get("any_nonzero", False)
    pm_norm = grad_results.get("pm/total_norm", 0.0)

    report = f"""# Active Perception Diagnostic Report

**Checkpoint**: `{checkpoint}`
**Evaluated samples**: {num_samples} (eval set)
**Date**: 2026-05-18

---

## [1] Attention Grounding

| Metric | Value |
|--------|-------|
| Samples with attention | {agg['n_samples_with_attn']} |
| Skipped (no attn / grid mismatch) | {agg['n_skipped']} |
| CE loss (mean ± std) | {(mass['mean'] and f"{agg['ce_loss']['mean']:.4f} ± {agg['ce_loss']['std']:.4f}") or 'N/A'} |
| Attn entropy (mean ± std) | {(ent['mean'] and f"{ent['mean']:.4f} ± {ent['std']:.4f}") or 'N/A'} |
| Bbox attn mass (mean) | {(mass['mean'] and f"{mass['mean']:.4f}") or 'N/A'} |
| Expected random mass (bbox area mean) | {(area['mean'] and f"{area['mean']:.4f}") or 'N/A'} |
| **Grounding ratio** (observed/expected) | {f"{gr:.3f}" if gr is not None else 'N/A'} |
| Top-10 overlap rate | {f"{ol:.3f}" if ol is not None else 'N/A'} |

**Verdict**: {grounding_verdict}

Notes:
- Grounding ratio = mean(bbox_attn_mass) / mean(bbox_area). >1.0 = better than random.
- Random baseline: grounding ratio ≈ 1.0 (uniform attention).
- Heatmaps with GT bbox (green rectangle) saved to `{output_dir}/heatmaps/`.

---

## [2] Gradient Verification

| Metric | Value |
|--------|-------|
| PerceptionModule total grad norm | {pm_norm:.6f} |
| Special token embeddings grad norm | {grad_results.get('special_token_embeddings_norm', 0.0):.6f} |
| Any non-zero gradient | {"YES ✓" if grad_ok else "NO ✗"} |

{"Gradients flow correctly into the PerceptionModule." if grad_ok else "WARNING: zero gradients detected."}

Key parameter norms (top 5 by magnitude):
"""
    pm_items = [(k, v) for k, v in grad_results.items()
                if k.startswith("pm/") and k != "pm/total_norm"]
    pm_items.sort(key=lambda x: -x[1])
    for name, val in pm_items[:5]:
        report += f"- `{name}`: {val:.6f}\n"

    report += f"""
---

## [3] Entropy Trend

| Step | Entropy (nats) |
|------|---------------|
"""
    for s, e in zip(steps, entropies):
        report += f"| {s} | {e:.4f} |\n"

    report += f"""
- **Start**: {ent_start}
- **End**: {ent_end}
- **Change**: {ent_delta}

Reference: uniform over N patches → entropy = ln(N).
For N=234 (most common in VGR): ln(234) ≈ 5.46 nats.
Observed range {min(entropies):.2f}–{max(entropies):.2f} suggests
effective concentration on ≈ {int(np.exp(min(entropies)))}–{int(np.exp(max(entropies)))} patches.

{"Entropy is **decreasing** → attention is becoming more focused (positive signal)." if entropies and entropies[-1] < entropies[0] else "Entropy is not consistently decreasing."}

---

## [4] Architectural Injection Check

| Metric | Value |
|--------|-------|
| Samples checked | {len(arch_results)} |
| cosine(static_PERC_OUT, z_visual) mean | {f"{mean_cos:.4f}" if mean_cos is not None else 'N/A'} |
| cosine std | {f"{float(np.std(cos_vals)):.4f}" if cos_vals else 'N/A'} |
| z_visual norm mean | {f"{float(np.mean(z_norms)):.2f}" if z_norms else 'N/A'} |
| Static PERC_OUT norm | {f"{arch_results[0]['static_emb_norm']:.2f}" if arch_results else 'N/A'} |

**Verdict**: {injection_verdict}

A cosine similarity << 1.0 confirms z_visual is carrying novel visual information,
not just copying the static special token embedding.

---

## Summary

| Question | Answer |
|----------|--------|
| Does the model attend to meaningful regions? | {grounding_verdict.split('(')[0].strip()} |
| Are gradients flowing to PerceptionModule? | {"YES" if grad_ok else "NO"} |
| Is attention sharpening over training? | {"YES" if entropies and entropies[-1] < entropies[0] else "UNCLEAR"} |
| Is z_visual truly injected (not static)? | {injection_verdict.split(':')[0]} |

Generated by `scripts/diagnose_attention.py`.
"""

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"  Report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_log",  required=True,
                        help="Path to train_metrics.jsonl from training run")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")
        import os
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

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
    logger.info(f"Eval dataset: {len(dataset)} samples")

    # ── [1] Attention diagnostics ─────────────────────────────────────────────
    logger.info("\n=== [1] Attention Diagnostics ===")
    all_metrics = run_attention_diagnostics(
        model, dataset, collator, device,
        num_samples=args.num_samples,
        heatmap_dir=out_dir / "heatmaps",
    )
    agg = aggregate_attn_metrics(all_metrics)
    logger.info(
        f"  entropy mean={agg['attn_entropy']['mean']:.4f}  "
        f"bbox_mass mean={agg['bbox_attn_mass']['mean']:.4f}  "
        f"bbox_area mean={agg['bbox_area']['mean']:.4f}  "
        f"grounding_ratio={agg['grounding_ratio']:.3f}  "
        f"top10_overlap={agg['top10_overlap_rate']:.3f}"
    )

    # Save per-sample metrics
    metrics_path = out_dir / "attention_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"summary": agg, "per_sample": all_metrics}, f, indent=2)
    logger.info(f"  Saved: {metrics_path}")

    # ── [2] Gradient verification ─────────────────────────────────────────────
    logger.info("\n=== [2] Gradient Verification ===")
    grad_results = run_gradient_check(model, dataset, collator, device)
    with open(out_dir / "grad_check.json", "w") as f:
        json.dump(grad_results, f, indent=2)

    # ── [4] Architectural check ───────────────────────────────────────────────
    logger.info("\n=== [4] Architectural Check ===")
    arch_results = run_arch_check(model, dataset, collator, device, num_samples=10)
    with open(out_dir / "arch_check.json", "w") as f:
        json.dump(arch_results, f, indent=2)

    # ── [3] Entropy curve ─────────────────────────────────────────────────────
    logger.info("\n=== [3] Entropy Curve ===")
    entropy_data = build_entropy_curve(args.train_log, out_dir)

    # ── [7] Report ────────────────────────────────────────────────────────────
    logger.info("\n=== [7] Generating Report ===")
    generate_report(
        agg, arch_results, grad_results, entropy_data,
        checkpoint=args.checkpoint,
        output_dir=out_dir,
        num_samples=args.num_samples,
    )

    logger.info("\n=== DONE ===")
    logger.info(f"  All outputs in: {out_dir}")
    if agg["grounding_ratio"] is not None:
        logger.info(f"  Grounding ratio: {agg['grounding_ratio']:.3f}  (>1 = better than random)")
    logger.info(f"  Entropy {entropy_data['entropy'][0]:.3f} → {entropy_data['entropy'][-1]:.3f}" if entropy_data.get('entropy') else "")


if __name__ == "__main__":
    main()
