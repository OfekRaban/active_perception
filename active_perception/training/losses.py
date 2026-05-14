"""
Modular loss functions for active perception training.

Design philosophy:
- CE-only is the PRIMARY baseline and should work well out of the box.
- All auxiliary losses (alignment, grounding) are OPTIONAL and CONFIGURABLE.
- No hard spatial sparsity; no hard top-k.
- The bottleneck is the single z_perception output token, NOT attention sparsity.

Loss registry:
  L_SFT     — standard cross-entropy on supervised tokens (always active)
  L_sem     — cosine alignment: z_perception vs observation text embedding (optional)
  L_crop    — cosine alignment: z_perception vs visual crop embedding (optional, ablation only)
  L_ground  — soft attention grounding toward bbox patches (optional, warmup only)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    # ── CE loss ──────────────────────────────────────────────────────────────
    lambda_sft: float = 1.0

    # ── Semantic alignment: z_perception ↔ observation text embedding ────────
    use_semantic_alignment: bool = False
    lambda_sem: float = 0.1
    # Projection dim for semantic alignment heads
    sem_proj_dim: int = 256

    # ── Crop/visual alignment: z_perception ↔ pooled crop visual embedding ───
    use_crop_alignment: bool = False        # ablation only; off by default
    lambda_crop: float = 0.05
    crop_proj_dim: int = 256

    # ── Attention grounding: attn_weights softly peaked on bbox patches ───────
    use_grounding: bool = False             # warmup only; off by default
    lambda_ground: float = 0.1
    grounding_temperature: float = 0.5     # <1 = sharper target; avoids hard KL
    grounding_blur_sigma: float = 1.0      # Gaussian blur on patch mask

    # ── Scheduling ───────────────────────────────────────────────────────────
    # lambda values can be externally decayed; store current values here
    # (trainer updates these during warmup → main phase transition)


@dataclass
class LossOutput:
    total: torch.Tensor
    ce: torch.Tensor
    sem: Optional[torch.Tensor] = None
    crop: Optional[torch.Tensor] = None
    ground: Optional[torch.Tensor] = None

    def as_log_dict(self) -> Dict[str, float]:
        d = {
            "loss/total": self.total.item(),
            "loss/ce": self.ce.item(),
        }
        if self.sem is not None:
            d["loss/sem"] = self.sem.item()
        if self.crop is not None:
            d["loss/crop"] = self.crop.item()
        if self.ground is not None:
            d["loss/ground"] = self.ground.item()
        return d


class AlignmentProjector(nn.Module):
    """Small 2-layer MLP projector for cosine alignment losses."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PerceptionLosses(nn.Module):
    """
    All active perception auxiliary losses in one module.

    Used by the trainer. The CE loss itself is computed by the base model
    (via model(labels=...) returning model.loss). This module handles the
    auxiliary objectives.
    """

    def __init__(self, d_model: int, config: LossConfig):
        super().__init__()
        self.config = config
        self.d_model = d_model

        if config.use_semantic_alignment:
            self.z_proj_sem = AlignmentProjector(d_model, config.sem_proj_dim)
            self.text_proj_sem = AlignmentProjector(d_model, config.sem_proj_dim)
            logger.info(
                f"[Losses] Semantic alignment projectors: "
                f"d_model={d_model} → d_proj={config.sem_proj_dim}"
            )

        if config.use_crop_alignment:
            self.z_proj_crop = AlignmentProjector(d_model, config.crop_proj_dim)
            self.visual_proj_crop = AlignmentProjector(d_model, config.crop_proj_dim)
            logger.info(
                f"[Losses] Crop alignment projectors: "
                f"d_model={d_model} → d_proj={config.crop_proj_dim}"
            )

    def compute(
        self,
        loss_ce: torch.Tensor,
        z_perceptions: List[Optional[torch.Tensor]],          # [K, D] per batch item
        attn_weights_list: List[Optional[torch.Tensor]],       # [K, N] per batch item
        obs_text_embeddings: Optional[List[Optional[torch.Tensor]]] = None,  # [K, D]
        crop_embeddings: Optional[List[Optional[torch.Tensor]]] = None,      # [K, D]
        patch_masks: Optional[List[Optional[torch.Tensor]]] = None,          # [K, N]
    ) -> LossOutput:
        """
        Compute total loss.

        Args:
            loss_ce:               CE loss from pass-2 forward (scalar)
            z_perceptions:         list of [K, D] tensors (one per batch item; None if no PERCEPTION)
            attn_weights_list:     list of [K, N] attention weight tensors
            obs_text_embeddings:   list of [K, D] text embeddings of observation text (optional)
            crop_embeddings:       list of [K, D] visual crop embeddings (optional, ablation)
            patch_masks:           list of [K, N] soft patch masks from bboxes (optional)
        """
        cfg = self.config
        total = cfg.lambda_sft * loss_ce
        sem_loss = crop_loss = ground_loss = None

        # ── Semantic alignment ──────────────────────────────────────────────
        if cfg.use_semantic_alignment and obs_text_embeddings is not None:
            sem_loss = self._semantic_alignment_loss(z_perceptions, obs_text_embeddings)
            if sem_loss is not None:
                total = total + cfg.lambda_sem * sem_loss

        # ── Crop alignment (ablation) ────────────────────────────────────────
        if cfg.use_crop_alignment and crop_embeddings is not None:
            crop_loss = self._crop_alignment_loss(z_perceptions, crop_embeddings)
            if crop_loss is not None:
                total = total + cfg.lambda_crop * crop_loss

        # ── Attention grounding ───────────────────────────────────────────────
        if cfg.use_grounding and patch_masks is not None:
            ground_loss = self._grounding_loss(attn_weights_list, patch_masks)
            if ground_loss is not None:
                total = total + cfg.lambda_ground * ground_loss

        return LossOutput(
            total=total,
            ce=loss_ce,
            sem=sem_loss,
            crop=crop_loss,
            ground=ground_loss,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _semantic_alignment_loss(
        self,
        z_perceptions: List[Optional[torch.Tensor]],
        obs_text_embeddings: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        L_sem = mean over all (b, k) pairs of:
            1 - cosine(Proj_z(z_perc[b,k]), Proj_text(obs_embed[b,k]))
        """
        losses = []
        for z_b, t_b in zip(z_perceptions, obs_text_embeddings):
            if z_b is None or t_b is None:
                continue
            # z_b: [K, D], t_b: [K, D]
            K = min(z_b.shape[0], t_b.shape[0])
            z_proj = self.z_proj_sem(z_b[:K])        # [K, d_proj] normalized
            t_proj = self.text_proj_sem(t_b[:K])     # [K, d_proj] normalized
            cos_sim = (z_proj * t_proj).sum(dim=-1)  # [K]
            losses.append((1.0 - cos_sim).mean())

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _crop_alignment_loss(
        self,
        z_perceptions: List[Optional[torch.Tensor]],
        crop_embeddings: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        L_crop = mean over (b, k) of:
            1 - cosine(Proj_z(z[b,k]), Proj_visual(crop_embed[b,k]))
        Ablation only — may encourage region imitation rather than semantic compression.
        """
        losses = []
        for z_b, c_b in zip(z_perceptions, crop_embeddings):
            if z_b is None or c_b is None:
                continue
            K = min(z_b.shape[0], c_b.shape[0])
            z_proj = self.z_proj_crop(z_b[:K])
            c_proj = self.visual_proj_crop(c_b[:K])
            cos_sim = (z_proj * c_proj).sum(dim=-1)
            losses.append((1.0 - cos_sim).mean())

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _grounding_loss(
        self,
        attn_weights_list: List[Optional[torch.Tensor]],
        patch_masks: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        Soft attention grounding loss.

        Instead of hard KL divergence against a binary mask, we use a
        temperature-scaled soft target derived from the bbox patch mask.

        L_ground = KL(log_softmax(attn/T) || softmax(mask/T))

        This is stable even when the mask is sparse because temperature
        scaling smooths out the hard boundaries.
        """
        T = self.config.grounding_temperature
        losses = []
        for attn_b, mask_b in zip(attn_weights_list, patch_masks):
            if attn_b is None or mask_b is None:
                continue
            # attn_b: [K, N], mask_b: [K, N]
            K = min(attn_b.shape[0], mask_b.shape[0])
            a = attn_b[:K]   # [K, N] — already softmax'd
            m = mask_b[:K]   # [K, N] — soft or binary

            # Target distribution: temperature-scaled softmax of mask
            # (for binary masks with temperature, this becomes smoother)
            log_target = F.log_softmax(
                torch.log(m + 1e-8) / T, dim=-1
            )  # [K, N]

            # Predicted log distribution
            log_pred = torch.log(a + 1e-8)  # [K, N]

            # KL(target || pred) — supervised direction
            kl = F.kl_div(log_pred, log_target.exp(), reduction="none")  # [K, N]
            losses.append(kl.sum(dim=-1).mean())  # mean over K, sum over N

        if not losses:
            return None
        return torch.stack(losses).mean()
