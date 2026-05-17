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
       z_visual  [B, K, D_llm]          (pure visual retrieval output)
       │
  Learned Scalar Gate                   (gate = sigmoid(W_g @ h_perception))
       │
  z_perception = h_perception + gate * z_visual   (gated residual)
       │
  z_perception  [B, K, D_llm]          (injected at <PERC_OUT> positions)

Design choices:
- NO hard spatial sparsity. The bottleneck is at the OUTPUT level (single z token),
  not at the attention level. Soft attention allows distributed, multi-focal, and
  global evidence compression.
- attn_weights are returned for diagnostics and optional grounding loss.
- Gated residual (Issue 5 fix): replaces the hard residual `z = z + h` with a
  learned scalar gate that controls how much visual evidence supplements the text
  prior. At init (gate_proj weights=0, bias=0), gate ≈ 0.5. During training the
  model can specialize toward pure visual retrieval (gate→1) or pure text bypass
  (gate→0), keeping the injected embedding on a consistent manifold.
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
    Full perception module: QueryAdapter + CrossAttention + OutputProjection + GatedResidual.

    Args:
        d_model:    LLM hidden dimension (e.g., 3584 for Qwen2.5-7B)
        d_query:    Bottleneck dimension in QueryAdapter (default: 256)
        num_heads:  Number of attention heads for cross-attention
        dropout:    Dropout probability
        residual:   If True, use a learned scalar gate residual from h_perception.
                    If False, output is pure visual retrieval (z_visual only).
    """

    def __init__(
        self,
        d_model: int,
        d_query: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.residual = residual

        self.query_adapter = QueryAdapter(d_model, d_query, dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

        if residual:
            # Scalar gate: g = sigmoid(W_g h + b_g), init to 0.5 (b_g=0, W_g=0).
            # Controls how much z_visual supplements h_perception.
            # Logging gate.mean() over training reveals whether the module is
            # converging toward visual retrieval (→1) or text bypass (→0).
            self.gate_proj = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)  # sigmoid(0) = 0.5 at init

        self._init_output_proj()

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[PerceptionModule] d_model={d_model}, d_query={d_query}, "
            f"num_heads={num_heads}, residual={residual} (gated), "
            f"params={num_params/1e6:.2f}M"
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
            z_perception:  [B, K, D] — latent visual evidence tokens
            attn_weights:  [B, K, N] — averaged over heads (or None)
        """
        unbatched = h_perception.dim() == 2
        if unbatched:
            h_perception = h_perception.unsqueeze(0)
            visual_memory = visual_memory.unsqueeze(0)

        B, K, D = h_perception.shape

        # 1. Query adapter: compress h_perception into structured query
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
        # z_raw: [B, K, D]

        # 3. Output projection + LayerNorm → pure visual retrieval signal
        z_visual = self.out_proj(z_raw)
        z_visual = self.layer_norm(z_visual)  # [B, K, D]

        # 4. Gated residual from h_perception (Issue 5 fix).
        #    z = h + sigmoid(W_g @ h) * z_visual
        #    At init: gate ≈ 0.5 → z ≈ h + 0.5 * z_visual (stable, in-manifold).
        #    The gate decouples the visual retrieval space from the text space:
        #    z_visual is the visual evidence; h is the text prior; gate is learned.
        if self.residual:
            gate = torch.sigmoid(self.gate_proj(h_perception))  # [B, K, 1]
            z = h_perception + gate * z_visual
        else:
            z = z_visual

        if unbatched:
            z = z.squeeze(0)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)

        return z, attn_weights

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
