"""Shared data utilities: image loading, bbox operations, text helpers."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: str):
    """Load PIL image from path. Returns None on failure."""
    try:
        from PIL import Image
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"[load_image] Failed to load {path}: {e}")
        return None


def crop_image(image, bbox: List[float], normalized: bool = False):
    """
    Crop a PIL image to the given bbox.
    bbox = [x1, y1, x2, y2] in pixel coords (or normalized if normalized=True).
    """
    w, h = image.size
    if normalized:
        x1, y1, x2, y2 = [
            int(bbox[0] * w), int(bbox[1] * h),
            int(bbox[2] * w), int(bbox[3] * h),
        ]
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Bbox → patch mask
# ---------------------------------------------------------------------------

def bbox_to_patch_mask(
    bbox: List[float],
    image_hw: Tuple[int, int],
    patch_grid_hw: Tuple[int, int],
    normalized: bool = False,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Convert a bbox to a soft patch-level mask over the visual memory grid.

    Args:
        bbox: [x1, y1, x2, y2] in pixel or normalized coords
        image_hw: (H, W) of the original image in pixels
        patch_grid_hw: (grid_H, grid_W) — post-merger patch grid dimensions
        normalized: True if bbox is in [0,1]
        blur_sigma: if > 0, apply Gaussian blur to soften hard edges

    Returns:
        mask: FloatTensor [grid_H * grid_W] with values in [0, 1]
    """
    H_img, W_img = image_hw
    gH, gW = patch_grid_hw

    if normalized:
        x1, y1, x2, y2 = bbox[0] * W_img, bbox[1] * H_img, bbox[2] * W_img, bbox[3] * H_img
    else:
        x1, y1, x2, y2 = bbox

    r0 = max(0, int(y1 / H_img * gH))
    r1 = min(gH, int(y2 / H_img * gH) + 1)
    c0 = max(0, int(x1 / W_img * gW))
    c1 = min(gW, int(x2 / W_img * gW) + 1)

    mask = torch.zeros(gH, gW, dtype=torch.float32)
    mask[r0:r1, c0:c1] = 1.0

    if blur_sigma > 0:
        from torchvision.transforms.functional import gaussian_blur
        k = max(3, int(blur_sigma * 3) * 2 + 1)
        mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0), kernel_size=k, sigma=blur_sigma)
        mask = mask.squeeze(0).squeeze(0).clamp(0, 1)

    return mask.flatten()


def soft_patch_mask(
    bbox: List[float],
    image_hw: Tuple[int, int],
    patch_grid_hw: Tuple[int, int],
    normalized: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Normalize the patch mask into a probability distribution."""
    mask = bbox_to_patch_mask(bbox, image_hw, patch_grid_hw, normalized)
    if mask.sum() == 0:
        return torch.ones_like(mask) / mask.numel()
    normalized_mask = mask / (mask.sum() + 1e-8)
    log_mask = torch.log(normalized_mask + 1e-8) / temperature
    return torch.softmax(log_mask, dim=0)


# ---------------------------------------------------------------------------
# Grid utilities for Qwen2.5-VL
# ---------------------------------------------------------------------------

def grid_thw_to_hw(grid_thw: torch.Tensor) -> Tuple[int, int]:
    """
    Extract (H, W) from Qwen2.5-VL grid_thw tensor.
    grid_thw shape: [1, 3] or [3] with values (T, H, W).
    These are the PRE-merger ViT patch grid dimensions.
    """
    if grid_thw.dim() == 2:
        t, h, w = grid_thw[0].tolist()
    else:
        t, h, w = grid_thw.tolist()
    return int(h), int(w)


def compute_patch_grid_hw(grid_thw: torch.Tensor, merge_size: int = 2) -> Tuple[int, int]:
    """
    Compute the post-merger patch grid (H_actual, W_actual).

    Qwen2.5-VL's Vision MLP Merger fuses merge_size×merge_size ViT tokens into
    one LLM token. grid_thw stores the PRE-merger ViT grid dimensions (T, H, W).
    After the merger the spatial grid shrinks by a factor of merge_size in each
    dimension, so the actual token count is:

        N_tokens = T * (H // merge_size) * (W // merge_size)

    This must match visual_memory.shape[-2] (output of encode_image_to_memory).
    Read merge_size from model.config.vision_config.spatial_merge_size (default 2).
    """
    h, w = grid_thw_to_hw(grid_thw)
    return h // merge_size, w // merge_size
