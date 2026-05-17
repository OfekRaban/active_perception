"""
2D Positional Encoding for External Visual Memory.

When visual tokens are moved OUTSIDE the native Qwen2.5-VL sequence,
they lose the 2D RoPE positioning they had inside the ViT.

The ViT's internal 2D RoPE is applied inside the ViT itself, so projected
tokens DO carry implicit spatial structure from the ViT representations.
However, the perception cross-attention has no positional context unless
we add it explicitly.

Three modes (all configurable):
  none              — no additional PE; rely on ViT's implicit spatial encoding
  additive_sincos2d — add sine/cosine 2D PE to visual memory before cross-attn
  concat_sincos2d   — concatenate 2D PE and project back to D_llm

Grid math (Issue 2 fix):
  Qwen2.5-VL applies a Vision MLP Merger with spatial_merge_size=2, fusing
  2×2 ViT patches into one LLM token. `grid_thw` from the processor stores the
  PRE-merger (T, H, W) ViT grid. The actual visual token count after the merger is:
      N = T * (H // merge_size) * (W // merge_size)
  `SpatialEncoding2D` takes `merge_size` as a constructor argument (default 2)
  and divides H and W internally before constructing the PE grid. The `merge_size`
  should be read from `model.config.vision_config.spatial_merge_size` at init.

Research note:
  We do NOT know which mode is best. Keep this modular for ablation.
  Start with "none" as the baseline; add PE if retrieval fails spatially.
"""
from __future__ import annotations
import math
import logging
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpatialEncodingMode(str, Enum):
    NONE = "none"
    ADDITIVE_SINCOS2D = "additive_sincos2d"
    CONCAT_SINCOS2D = "concat_sincos2d"


class SpatialEncoding2D(nn.Module):
    """
    Generates and applies 2D positional encodings for a [N, D] visual memory tensor.

    N = (H // merge_size) * (W // merge_size) post-merger patches, row-major order.
    grid_thw = (T, H, W) from Qwen2.5-VL processor (PRE-merger ViT grid).
    merge_size: Vision MLP Merger downsampling factor (2 for Qwen2.5-VL).
    """

    def __init__(
        self,
        d_model: int,
        mode: SpatialEncodingMode = SpatialEncodingMode.NONE,
        merge_size: int = 2,
        max_grid_size: int = 64,
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = SpatialEncodingMode(mode)
        self.merge_size = merge_size
        self.max_grid_size = max_grid_size
        self.temperature = temperature

        if self.mode == SpatialEncodingMode.CONCAT_SINCOS2D:
            self.proj = nn.Linear(d_model + d_model, d_model, bias=False)
            nn.init.eye_(self.proj.weight[:d_model, :d_model])
            logger.info(
                f"[SpatialEncoding2D] mode=concat_sincos2d, "
                f"proj: {d_model+d_model}→{d_model}"
            )
        else:
            self.proj = None

        if self.mode != SpatialEncodingMode.NONE:
            logger.info(
                f"[SpatialEncoding2D] mode={self.mode}, d_model={d_model}, "
                f"merge_size={merge_size}"
            )

    def forward(
        self,
        visual_memory: torch.Tensor,   # [N, D] or [B, N, D]
        grid_thw: torch.Tensor,        # [1, 3] or [3]: (T, H, W) pre-merger
    ) -> torch.Tensor:
        if self.mode == SpatialEncodingMode.NONE:
            return visual_memory

        pe = self._build_pe(visual_memory, grid_thw)

        if self.mode == SpatialEncodingMode.ADDITIVE_SINCOS2D:
            return visual_memory + pe

        elif self.mode == SpatialEncodingMode.CONCAT_SINCOS2D:
            batched = visual_memory.dim() == 3
            if batched:
                pe = pe.unsqueeze(0).expand(visual_memory.shape[0], -1, -1)
            concatenated = torch.cat([visual_memory, pe], dim=-1)
            return self.proj(concatenated)

        return visual_memory

    def _build_pe(
        self,
        visual_memory: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build 2D sine/cosine PE of shape [N_actual, D_model].

        grid_thw stores the pre-merger ViT grid (T, H, W). We divide H and W
        by merge_size to get the actual post-merger token grid dimensions,
        matching visual_memory.shape[-2].
        """
        if grid_thw.dim() == 2:
            _, H_pre, W_pre = grid_thw[0].int().tolist()
        else:
            _, H_pre, W_pre = grid_thw.int().tolist()

        # Apply merger downsampling to get actual token grid (Issue 2 fix)
        H = int(H_pre) // self.merge_size
        W = int(W_pre) // self.merge_size
        N = H * W

        assert N == visual_memory.shape[-2], (
            f"[SpatialEncoding2D] Post-merger grid H*W={N} "
            f"(grid_thw={grid_thw.tolist()}, merge_size={self.merge_size}) "
            f"!= visual_memory N={visual_memory.shape[-2]}. "
            f"Check that merge_size matches model.config.vision_config.spatial_merge_size."
        )

        D = self.d_model
        half_D = D // 2

        pe = torch.zeros(H, W, D, device=visual_memory.device, dtype=visual_memory.dtype)

        # Row (height) encoding in first half_D dims
        dim_h = torch.arange(half_D // 2, device=visual_memory.device, dtype=torch.float32)
        freq_h = 1.0 / (self.temperature ** (2 * dim_h / half_D))
        rows = torch.arange(H, device=visual_memory.device, dtype=torch.float32)
        angle_h = torch.outer(rows, freq_h)  # [H, half_D//2]
        pe[:, :, 0:half_D:2] = angle_h.sin().unsqueeze(1)
        pe[:, :, 1:half_D:2] = angle_h.cos().unsqueeze(1)

        # Column (width) encoding in second half_D dims
        half_rest = (D - half_D) // 2
        dim_w = torch.arange(half_rest, device=visual_memory.device, dtype=torch.float32)
        freq_w = 1.0 / (self.temperature ** (2 * dim_w / (D - half_D)))
        cols = torch.arange(W, device=visual_memory.device, dtype=torch.float32)
        angle_w = torch.outer(cols, freq_w)  # [W, half_rest]
        pe[:, :, half_D::2] = angle_w.sin().unsqueeze(0)
        pe[:, :, half_D+1::2] = angle_w.cos().unsqueeze(0)

        return pe.reshape(N, D).to(dtype=visual_memory.dtype)
