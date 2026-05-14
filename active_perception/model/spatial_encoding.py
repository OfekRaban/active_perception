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

    N = H_grid * W_grid patches, laid out in row-major order.
    grid_thw = (T, H, W) from Qwen2.5-VL; we use H and W for the 2D grid.
    """

    def __init__(
        self,
        d_model: int,
        mode: SpatialEncodingMode = SpatialEncodingMode.NONE,
        max_grid_size: int = 64,
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = SpatialEncodingMode(mode)
        self.max_grid_size = max_grid_size
        self.temperature = temperature

        if self.mode == SpatialEncodingMode.CONCAT_SINCOS2D:
            # Project concatenated [D + D_pe] back to D
            self.proj = nn.Linear(d_model + d_model, d_model, bias=False)
            nn.init.eye_(self.proj.weight[:d_model, :d_model])  # identity for original dims
            logger.info(f"[SpatialEncoding2D] mode=concat_sincos2d, proj: {d_model+d_model}→{d_model}")
        else:
            self.proj = None

        if self.mode != SpatialEncodingMode.NONE:
            logger.info(f"[SpatialEncoding2D] mode={self.mode}, d_model={d_model}")

    def forward(
        self,
        visual_memory: torch.Tensor,   # [N, D] or [B, N, D]
        grid_thw: torch.Tensor,        # [1, 3] or [3]: (T, H, W)
    ) -> torch.Tensor:
        """
        Returns positional encoding of shape matching visual_memory, or None if mode=none.
        """
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
        """Build 2D sine/cosine PE of shape [N, D_model]."""
        # Extract H, W from grid_thw
        if grid_thw.dim() == 2:
            _, H, W = grid_thw[0].int().tolist()
        else:
            _, H, W = grid_thw.int().tolist()
        H, W = int(H), int(W)
        N = H * W

        assert N == visual_memory.shape[-2], (
            f"[SpatialEncoding2D] grid H*W={N} != visual_memory N={visual_memory.shape[-2]}. "
            f"grid_thw={grid_thw}"
        )

        D = self.d_model
        half_D = D // 2  # half for height, half for width

        pe = torch.zeros(H, W, D, device=visual_memory.device, dtype=visual_memory.dtype)

        # Row (height) encoding in first half_D dims
        dim_h = torch.arange(half_D // 2, device=visual_memory.device, dtype=torch.float32)
        freq_h = 1.0 / (self.temperature ** (2 * dim_h / half_D))
        rows = torch.arange(H, device=visual_memory.device, dtype=torch.float32)
        angle_h = torch.outer(rows, freq_h)  # [H, half_D//2]
        pe[:, :, 0:half_D:2] = angle_h.sin().unsqueeze(1)
        pe[:, :, 1:half_D:2] = angle_h.cos().unsqueeze(1)

        # Column (width) encoding in second half_D dims
        dim_w = torch.arange(D - half_D, device=visual_memory.device, dtype=torch.float32)
        # handle odd D
        half_rest = (D - half_D) // 2
        freq_w = 1.0 / (self.temperature ** (2 * dim_w[:half_rest] / (D - half_D)))
        cols = torch.arange(W, device=visual_memory.device, dtype=torch.float32)
        angle_w = torch.outer(cols, freq_w)  # [W, half_rest]
        pe[:, :, half_D::2] = angle_w.sin().unsqueeze(0)
        pe[:, :, half_D+1::2] = angle_w.cos().unsqueeze(0)

        pe = pe.reshape(N, D)
        return pe.to(dtype=visual_memory.dtype)
