"""
Perception Module: the core of active perception.

Architecture:
  h_perception  [B, K, D_llm]          (hidden state at <PERCEPTION> position(s))
       │
  QueryAdapter  (D_llm → D_q → D_llm)  (bottleneck MLP + LayerNorm)
       │
       q  [B, K, D_llm]
       │
  CrossAttention(q, visual_memory_pe)   (multi-head, K queries over N patches)
       │
       z_raw  [B, K, D_llm]
       │
  OutputProjection + LayerNorm
       │
  z_perception  [B, K, D_llm]          (pure visual latent — injected at <PERC_OUT>)

Design:
- Pure visual latent space. z_perception is the output of cross-attention +
  projection, with NO residual addition of h_perception. The injected vector
  lives entirely in the visual evidence manifold; the LLM learns to read it
  without prior text contamination.
- attn_weights are returned for diagnostics and optional grounding loss.
- OutputProjection is initialised with gain=0.1 so early training starts with
  small perturbations while the QueryAdapter warms up.
"""
from __future__ import annotations
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QueryAdapter(nn.Module):
    """
    Maps h_perception [D_llm] → q [D_llm] through a bottleneck D_q.

    The bottleneck forces the adapter to compress the reasoning state into
    a structured query rather than copying it verbatim.
    """

    def __init__(self, d_model: int, d_query: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_query = d_query
        self.net = nn.Sequential(
            nn.Linear(d_model, d_query),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_query, d_model),
            nn.LayerNorm(d_model),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PerceptionModule(nn.Module):
    """
    Full perception module: QueryAdapter → CrossAttention → OutputProjection.

    Returns a pure visual latent z_perception with no text-state blending.

    Args:
        d_model:   LLM hidden dimension (e.g., 3584 for Qwen2.5-7B)
        d_query:   Bottleneck dimension in QueryAdapter (default: 256)
        num_heads: Number of attention heads for cross-attention
        dropout:   Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_query: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        self.query_adapter = QueryAdapter(d_model, d_query, dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_output_proj()

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[PerceptionModule] d_model={d_model}, d_query={d_query}, "
            f"num_heads={num_heads}, params={num_params/1e6:.2f}M"
        )

    def _init_output_proj(self):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h_perception: torch.Tensor,     # [B, K, D] or [K, D] for unbatched
        visual_memory: torch.Tensor,    # [B, N, D] or [N, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N] True=ignore
        return_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_perception:      Hidden states at <PERCEPTION> positions.
                               Shape: [B, K, D] where K = num perception steps.
            visual_memory:     Projected visual tokens (external memory).
                               Shape: [B, N, D] where N = num visual patches.
            key_padding_mask:  Optional mask for visual memory (True = ignore).
            return_attn_weights: Whether to return attention weights for diagnostics.

        Returns:
            z_perception:  [B, K, D] — pure visual latent (no h_perception blending)
            attn_weights:  [B, K, N] — averaged over heads (or None)
        """
        unbatched = h_perception.dim() == 2
        if unbatched:
            h_perception = h_perception.unsqueeze(0)
            visual_memory = visual_memory.unsqueeze(0)

        B, K, D = h_perception.shape

        # 1. Query adapter: compress h_perception into a structured query
        q = self.query_adapter(h_perception)  # [B, K, D]

        # 2. Cross-attention over visual memory
        if visual_memory.shape[0] == 1 and B > 1:
            visual_memory = visual_memory.expand(B, -1, -1)

        z_raw, attn_weights = self.cross_attn(
            query=q,
            key=visual_memory,
            value=visual_memory,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
            average_attn_weights=True,
        )

        # 3. Output projection + LayerNorm → pure visual latent
        z = self.layer_norm(self.out_proj(z_raw))  # [B, K, D]

        if unbatched:
            z = z.squeeze(0)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)

        return z, attn_weights

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
