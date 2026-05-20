#!/usr/bin/env python3
"""
Quick qualitative visualization for exp12b (best checkpoint).

For 3 eval samples:
  - Runs autoregressive generation → extracts final answer
  - Runs training_forward → grabs attention weights
  - Saves one figure per sample: image + GT bbox + attention heatmap
"""
import json
import sys
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator
from active_perception.data.utils import resolve_image_path

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT  = "runs/exp12b_lora_full22k_none_lr5e5/checkpoint-best"
EVAL_DATA   = "data/vgr_existing_eval.jsonl"
MODEL_PATH  = "/cortex/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
OUTPUT_DIR  = Path("outputs/quick_vis")
SAMPLE_IDXS = [2, 3, 4]   # giraffe/left, blue-parrot/yes, black-hat/yes

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    """Pull 'Final answer: X' or last non-empty line from generated text."""
    m = re.search(r'[Ff]inal\s+answer\s*[:\-]?\s*(.+)', text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def make_figure(
    orig_img:    Image.Image,
    attn_hw:     np.ndarray,      # [H, W] normalised
    bbox:        list,             # [x1,y1,x2,y2] normalised
    question:    str,
    gt_answer:   str,
    gen_answer:  str,
    save_path:   Path,
):
    import matplotlib.cm as cmaps

    W_img, H_img = orig_img.size
    aspect = W_img / H_img
    # Fix panel height at 5 inches; width scales with aspect ratio
    panel_h = 5.0
    panel_w = panel_h * aspect
    fig_w   = panel_w * 2 + 0.4          # 2 panels + small gap
    fig_h   = panel_h + 1.2              # extra room for text header

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e2e")
    # 2 image axes in the lower portion; 1 text axis at the top
    ax_txt = fig.add_axes([0.0, 0.82, 1.0, 0.18])   # top 18% for text
    ax_l   = fig.add_axes([0.01, 0.02, 0.47, 0.78])  # left image
    ax_r   = fig.add_axes([0.52, 0.02, 0.47, 0.78])  # right image

    for ax in (ax_l, ax_r):
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
    ax_txt.set_facecolor("#1e1e2e")
    ax_txt.axis("off")

    # ── Text header ──────────────────────────────────────────────────────────
    correct = gt_answer.strip().lower() in gen_answer.strip().lower()
    verdict = "✓  CORRECT" if correct else "✗  WRONG"
    v_color = "#00ff88" if correct else "#ff6b6b"
    q_short  = question if len(question) <= 80 else question[:77] + "…"
    ax_txt.text(0.5, 0.80, f"Q: {q_short}", color="white",   fontsize=10.5,
                ha="center", va="top", transform=ax_txt.transAxes, wrap=True)
    ax_txt.text(0.5, 0.40, f"GT: {gt_answer}",               color="#aaaaff", fontsize=10,
                ha="center", va="top", transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.02, f"Generated: {gen_answer[:120]}   {verdict}",
                color=v_color, fontsize=10,
                ha="center", va="top", transform=ax_txt.transAxes)

    x1, y1, x2, y2 = bbox

    # ── Left: original image + GT bbox ──────────────────────────────────────
    ax_l.imshow(orig_img)
    ax_l.add_patch(patches.Rectangle(
        (x1 * W_img, y1 * H_img), (x2 - x1) * W_img, (y2 - y1) * H_img,
        linewidth=2.5, edgecolor="#00ff88", facecolor="none",
    ))
    ax_l.set_title("Image + GT bbox", color="white", fontsize=10, pad=4)

    # ── Right: attention heatmap blended on image ────────────────────────────
    cmap    = cmaps.get_cmap("inferno")  # noqa: deprecated but works on this mpl version
    hm_rgba = cmap(attn_hw)
    hm_rgb  = (hm_rgba[:, :, :3] * 255).astype(np.uint8)
    hm_pil  = Image.fromarray(hm_rgb).resize(orig_img.size, Image.BILINEAR)
    blended = Image.blend(orig_img.convert("RGB"), hm_pil, alpha=0.55)

    ax_r.imshow(blended)
    ax_r.add_patch(patches.Rectangle(
        (x1 * W_img, y1 * H_img), (x2 - x1) * W_img, (y2 - y1) * H_img,
        linewidth=2.5, edgecolor="#00ff88", facecolor="none",
    ))
    ax_r.set_title("Attention heatmap + GT bbox", color="white", fontsize=10, pad=4)

    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    # Load model
    print("Loading model …")
    cfg = ActivePerceptionConfig(
        model_path=MODEL_PATH,
        attn_implementation="sdpa",
        initial_perception_mode="none",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        freeze_vit=True,
        freeze_projector=True,
        freeze_llm=True,
        torch_dtype="bfloat16",
    )
    model = ActivePerceptionModel(cfg)
    model.load_perception_module(CHECKPOINT)
    model = model.to(device)
    model.eval()
    merge_size = getattr(model.base_model.config.vision_config, "spatial_merge_size", 2)
    print(f"Checkpoint loaded: {CHECKPOINT}  |  merge_size={merge_size}")

    # Load dataset / collator
    special_ids = model.get_special_token_ids()
    dataset = ActivePerceptionDataset(
        data_path=EVAL_DATA,
        processor=model.processor,
        special_token_ids=special_ids,
        max_seq_len=2048,
    )
    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )
    print(f"Eval dataset: {len(dataset)} samples")

    for idx in SAMPLE_IDXS:
        print(f"\n{'='*60}")
        raw = dataset.samples[idx]
        print(f"Sample {idx}  |  Q: {raw.question}")
        print(f"  GT answer: {raw.converted_answer}")

        item  = dataset[idx]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # ── 1. Autoregressive generation ──────────────────────────────────
        print("  Running generation …")
        # Build prompt-only inputs
        prompt_msgs  = dataset._build_prompt_messages(raw)
        prompt_text  = model.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        orig_image = dataset._load_image(raw.image)
        enc = model.processor(
            text=[prompt_text], images=[orig_image],
            return_tensors="pt", padding=False,
        )
        enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}

        with torch.no_grad():
            gen_ids = model.generate_with_perception(
                input_ids=enc["input_ids"],
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=256,
                do_sample=False,
            )
        gen_text   = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        gen_answer = extract_final_answer(gen_text)
        gt_answer  = extract_final_answer(raw.converted_answer or "")
        print(f"  Generated text (last 200 chars): …{gen_text[-200:]}")
        print(f"  Extracted answer: [{gen_answer}]   GT: [{gt_answer}]")

        # ── 2. Attention weights via training_forward ──────────────────────
        print("  Getting attention weights …")
        with torch.no_grad():
            out = model.training_forward(**batch)

        attn_w = out["attn_weights_list"][0]   # [K, N] or None
        grid_thw = batch.get("image_grid_thw")

        if attn_w is None or grid_thw is None:
            print("  WARNING: no attention weights, skipping visualisation")
            continue

        _, H_pre, W_pre = [int(x) for x in grid_thw[0].cpu().tolist()]
        H = H_pre // merge_size
        W = W_pre // merge_size

        if attn_w.shape[-1] != H * W:
            print(f"  WARNING: attn shape {attn_w.shape} != expected [{H}×{W}={H*W}]")
            continue

        # Use first query's attention (most VGR samples have K=1)
        vec    = attn_w[0].float().cpu().numpy()   # [N]
        hm     = vec.reshape(H, W)
        mn, mx = hm.min(), hm.max()
        hm_norm = (hm - mn) / (mx - mn + 1e-8)

        # ── 3. Bbox ────────────────────────────────────────────────────────
        bboxes = [s.bbox for s in raw.perception_steps if s.has_bbox()]
        bbox   = bboxes[0] if bboxes else [0, 0, 1, 1]

        # ── 4. Figure ──────────────────────────────────────────────────────
        if orig_image is None:
            img_path = resolve_image_path(raw.image)
            orig_image = Image.open(img_path).convert("RGB") if img_path else None
        if orig_image is None:
            print("  WARNING: could not load image")
            continue

        save_path = OUTPUT_DIR / f"sample_{idx:03d}.png"
        make_figure(
            orig_img=orig_image,
            attn_hw=hm_norm,
            bbox=bbox,
            question=raw.question,
            gt_answer=gt_answer,
            gen_answer=gen_answer,
            save_path=save_path,
        )

    print(f"\nDone. Figures in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
