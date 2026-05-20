#!/usr/bin/env python3
"""
Visualize perception attention maps from a trained ActivePerceptionModel checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=0 /cortex/users/rabanof/conda_envs/qwen49/bin/python \\
        scripts/visualize_perception.py \\
        --config configs/exp1_ce_only.yaml \\
        --checkpoint runs/exp1_ce_only/checkpoint-best \\
        --output_dir outputs/perception_viz/exp1_step500 \\
        --num_samples 5
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

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


# ── Attention utilities ───────────────────────────────────────────────────────

def attn_entropy(weights: np.ndarray) -> float:
    return float(-np.sum(weights * np.log(weights + 1e-9)))


def overlay_heatmap_on_image(
    img: Image.Image,
    heatmap_hw: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    hm_rgba = cmap(heatmap_hw)                          # [H, W, 4]
    hm_rgb = (hm_rgba[:, :, :3] * 255).astype(np.uint8)
    hm_pil = Image.fromarray(hm_rgb, mode="RGB").resize(img.size, Image.BILINEAR)
    return Image.blend(img.convert("RGB"), hm_pil, alpha)


def reshape_attn_to_grid(
    attn_vec: np.ndarray,
    grid_thw: list,
    merge_size: int,
    sample_idx: int,
    query_idx: int,
) -> tuple:
    """
    Reshape attn_vec [N] → normalized [H, W] heatmap.

    Returns (heatmap_hw or None, meta_dict).
    """
    _, H_pre, W_pre = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    H = H_pre // merge_size
    W = W_pre // merge_size
    N_expected = H * W
    N_actual = len(attn_vec)

    fallback_used = False
    heatmap_hw = None

    if N_actual == N_expected:
        heatmap_hw = attn_vec.reshape(H, W)
    else:
        logger.warning(
            f"  [SHAPE MISMATCH] sample={sample_idx} query={query_idx}: "
            f"N={N_actual}, expected H*W={N_expected} (H={H}, W={W})"
        )
        sqrt_n = int(np.sqrt(N_actual))
        for h_try, w_try in [(sqrt_n, sqrt_n), (sqrt_n, sqrt_n + 1), (sqrt_n + 1, sqrt_n)]:
            if h_try * w_try == N_actual:
                H, W = h_try, w_try
                heatmap_hw = attn_vec.reshape(H, W)
                fallback_used = True
                logger.warning(f"  [FALLBACK] Using inferred grid {H}x{W}")
                break
        if heatmap_hw is None:
            logger.warning(f"  [NO FALLBACK] Cannot infer grid for N={N_actual}. Heatmap skipped.")

    if heatmap_hw is not None:
        mn, mx = heatmap_hw.min(), heatmap_hw.max()
        heatmap_hw = (heatmap_hw - mn) / (mx - mn) if mx > mn else np.zeros_like(heatmap_hw)

    meta = {
        "heatmap_shape": [H, W] if heatmap_hw is not None else None,
        "attn_N_actual": N_actual,
        "attn_N_expected": N_expected,
        "fallback_used": fallback_used,
        "grid_h_pre": H_pre,
        "grid_w_pre": W_pre,
        "merge_size": merge_size,
    }
    return heatmap_hw, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--device", default=None,
        help="Device string, e.g. 'cuda' or 'cuda:0'. "
             "Prefer setting CUDA_VISIBLE_DEVICES instead of this flag.",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    cfg = load_config(args.config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = ActivePerceptionConfig(**{
        k: v for k, v in cfg.get("model", {}).items()
        if k in ActivePerceptionConfig.__dataclass_fields__
    })
    logger.info(f"Model config: {model_cfg}")
    model = ActivePerceptionModel(model_cfg)
    model.load_perception_module(args.checkpoint)
    model = model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded: {args.checkpoint}")

    merge_size = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)
    logger.info(f"spatial_merge_size: {merge_size}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    eval_data_path = data_cfg.get("eval_data_path") or data_cfg.get("data_path")
    special_ids = model.get_special_token_ids()

    dataset = ActivePerceptionDataset(
        data_path=eval_data_path,
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        supervision_mode=data_cfg.get("supervision_mode", "full"),
    )
    logger.info(f"Eval dataset: {len(dataset)} samples")

    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )

    num_samples = min(args.num_samples, len(dataset))
    summary_rows = []

    for sample_idx in range(num_samples):
        logger.info(f"\n=== Sample {sample_idx} ===")

        item = dataset[sample_idx]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model.training_forward(**batch)

        loss_ce = out["loss_ce"].item()
        attn_weights_list = out["attn_weights_list"]    # list[K, N] or None per batch item
        modified_input_ids = out["modified_input_ids"]  # [1, T_new]

        try:
            decoded_seq = model.tokenizer.decode(
                modified_input_ids[0].cpu().tolist(), skip_special_tokens=False
            )
            decoded_truncated = decoded_seq[:400]
        except Exception:
            decoded_truncated = "<decode error>"

        image_grid_thw = batch.get("image_grid_thw")
        grid_thw = image_grid_thw[0].cpu().tolist() if image_grid_thw is not None else None

        attn_w = attn_weights_list[0]  # [K, N] or None

        # ── Load original image ────────────────────────────────────────────────
        raw_sample = dataset.samples[sample_idx]
        img_path = resolve_image_path(raw_sample.image)
        orig_image = None
        missing_image = False
        if img_path is not None:
            try:
                orig_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"  Could not open image {img_path}: {e}")
                missing_image = True
        else:
            logger.warning(f"  Image path not resolved: {raw_sample.image!r}")
            missing_image = True

        if attn_w is None:
            logger.warning(f"  No <PERCEPTION> tokens in this sample — saving metadata only.")
            meta_out = {
                "sample_idx": sample_idx,
                "sample_id": raw_sample.id,
                "loss_ce": loss_ce,
                "question": raw_sample.question,
                "answer": raw_sample.converted_answer,
                "decoded_modified_sequence": decoded_truncated,
                "image_grid_thw": grid_thw,
                "spatial_merge_size": merge_size,
                "checkpoint_path": str(args.checkpoint),
                "no_perception_tokens": True,
                "missing_image": missing_image,
            }
            meta_path = out_dir / f"sample_{sample_idx}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(meta_out, f, indent=2)
            logger.info(f"  Saved: {meta_path}")
            summary_rows.append({
                "sample_idx": sample_idx,
                "loss_ce": loss_ce,
                "num_queries": 0,
                "entropy": [],
                "valid_heatmaps": 0,
                "missing_image": missing_image,
            })
            continue

        attn_w_np = attn_w.float().cpu().numpy()  # [K, N]
        K, N_actual = attn_w_np.shape
        logger.info(f"  attn_weights: [{K}, {N_actual}]  loss_ce={loss_ce:.4f}")

        per_query_meta = []
        sample_entropies = []
        valid_heatmaps = 0

        for query_idx in range(K):
            attn_vec = attn_w_np[query_idx]  # [N]
            entropy = attn_entropy(attn_vec)
            sample_entropies.append(entropy)

            top10_idx = np.argsort(attn_vec)[::-1][:10].tolist()
            top10_weights = attn_vec[top10_idx].tolist()
            logger.info(
                f"  query {query_idx}: entropy={entropy:.4f}  "
                f"top1_weight={top10_weights[0]:.4f}  top1_idx={top10_idx[0]}"
            )

            heatmap_hw = None
            shape_meta = {"fallback_used": False}
            if grid_thw is not None:
                heatmap_hw, shape_meta = reshape_attn_to_grid(
                    attn_vec, grid_thw, merge_size, sample_idx, query_idx
                )

            # Save raw attention vector
            npy_path = out_dir / f"sample_{sample_idx}_query_{query_idx}_heatmap_raw.npy"
            np.save(npy_path, attn_vec)

            # Save heatmap PNG
            png_path = out_dir / f"sample_{sample_idx}_query_{query_idx}_heatmap.png"
            if heatmap_hw is not None:
                valid_heatmaps += 1
                if orig_image is not None:
                    vis_img = overlay_heatmap_on_image(orig_image, heatmap_hw)
                else:
                    hm_uint8 = (heatmap_hw * 255).astype(np.uint8)
                    vis_img = Image.fromarray(hm_uint8, mode="L").convert("RGB")
                vis_img.save(png_path)
                logger.info(f"  query {query_idx}: saved {png_path}")
            else:
                logger.warning(f"  query {query_idx}: heatmap PNG skipped (no valid grid)")

            per_query_meta.append({
                "query_idx": query_idx,
                "entropy": entropy,
                "top10_visual_indices": top10_idx,
                "top10_weights": top10_weights,
                **shape_meta,
            })

        meta_out = {
            "sample_idx": sample_idx,
            "sample_id": raw_sample.id,
            "loss_ce": loss_ce,
            "question": raw_sample.question,
            "answer": raw_sample.converted_answer,
            "decoded_modified_sequence": decoded_truncated,
            "image_grid_thw": grid_thw,
            "spatial_merge_size": merge_size,
            "checkpoint_path": str(args.checkpoint),
            "num_queries": K,
            "missing_image": missing_image,
            "per_query": per_query_meta,
        }
        meta_path = out_dir / f"sample_{sample_idx}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)
        logger.info(f"  Saved: {meta_path}")

        summary_rows.append({
            "sample_idx": sample_idx,
            "loss_ce": loss_ce,
            "num_queries": K,
            "entropy": sample_entropies,
            "valid_heatmaps": valid_heatmaps,
            "missing_image": missing_image,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== SUMMARY ===")
    for row in summary_rows:
        ent_str = (
            "[" + ", ".join(f"{e:.3f}" for e in row["entropy"]) + "]"
            if row["entropy"] else "N/A"
        )
        logger.info(
            f"  sample {row['sample_idx']}: "
            f"loss={row['loss_ce']:.4f}  "
            f"queries={row['num_queries']}  "
            f"entropy={ent_str}  "
            f"valid_heatmaps={row['valid_heatmaps']}  "
            f"missing_image={row['missing_image']}"
        )


if __name__ == "__main__":
    main()
