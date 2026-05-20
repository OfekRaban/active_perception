"""
Qwen2.5-VL Active Perception Wrapper.

Core responsibility:
  1. Load Qwen2.5-VL from a local path (no HuggingFace download assumed).
  2. Encode the image with the ViT+projector → store as external visual memory.
  3. Replace the N visual patch tokens in the LLM sequence with a single <IMAGE> token.
  4. Fix position_ids / M-RoPE after the reduction.
  5. Two-pass training forward:
       Pass 1 (always no_grad): run LLM → extract h_perception
       Pass 2 (with grad):      inject z_perception at <PERC_OUT>, run LLM → losses
  6. Inference with autoregressive generation and perception interception.

M-RoPE notes:
  Qwen2.5-VL uses 3D position_ids [3, T] for M-RoPE (temporal, height, width).
  When we replace N visual patch tokens with 1 <IMAGE> token, we recompute
  position_ids for the entire modified sequence as sequential 1D text positions.
  This breaks native visual-token RoPE but is intentional: spatial structure now
  lives only in the external visual_memory, accessed via the perception cross-attn.

Special token embeddings (Issue 4 fix):
  The 4 new tokens (<IMAGE>, <PERCEPTION>, <PERC_OUT>, <INIT_PERC_OUT>) are
  registered as a separate nn.Parameter `special_token_embeddings` [4, d_model]
  instead of unfreezing the full 152k-row embedding table. Adam only tracks
  optimizer states for these 4 rows (~14k params) rather than all 545M embedding
  params. The embedding table stays fully frozen. The _embed() helper injects
  the learnable rows at forward time by overwriting positions where input_ids
  equals one of the 4 special token IDs.

Initial perception modes (initial_perception_mode):
  Before CoT begins, the model can receive a pre-injected visual context at
  <INIT_PERC_OUT>. Three ablation configurations:
    "latent"  — PerceptionModule is queried using the <IMAGE> hidden state
                from Pass 1 → z_init is injected at <INIT_PERC_OUT> in Pass 2.
    "spatial" — visual_memory is adaptively avg-pooled to spatial_pool_size²
                tokens; <INIT_PERC_OUT> is expanded to that many positions in
                the sequence, each receiving one pooled visual patch.
    "none"    — <INIT_PERC_OUT> is removed from the sequence entirely; blind
                CoT baseline with no pre-injected visual context.

Pass 1 no_grad (Issue 3 fix):
  Pass 1 (h_perception extraction) always runs under torch.no_grad(),
  unconditionally. This treats h_perception as a stop-gradient input to the
  PerceptionModule — analogous to how RAG/RETRO treat the retrieval step.
  In Stage 3 (LoRA), LoRA gradients still flow through Pass 2 (which uses
  the same LLM weights), so the gradient truncation has negligible empirical
  impact while eliminating the double-graph OOM risk (~20GB saved at 7B).
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .special_tokens import SpecialTokens, add_special_tokens_to_model
from .spatial_encoding import SpatialEncoding2D, SpatialEncodingMode
from .perception_module import PerceptionModule

logger = logging.getLogger(__name__)


@dataclass
class ActivePerceptionConfig:
    # ── Model paths ──────────────────────────────────────────────────────────
    model_path: str = "/path/to/Qwen2.5-VL-7B-Instruct"
    attn_implementation: str = "sdpa"      # "eager" | "sdpa" | "flash_attention_2"
    # ── Architecture ─────────────────────────────────────────────────────────
    d_query: int = 256
    num_perception_heads: int = 8
    spatial_encoding_mode: str = "none"
    # ── Special token initialization ─────────────────────────────────────────
    token_init_strategy: str = "mean_visual_text"
    # ── LoRA ─────────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    # ── Freezing ─────────────────────────────────────────────────────────────
    freeze_vit: bool = True
    freeze_projector: bool = True
    freeze_llm: bool = True
    # ── Initial perception ablation ───────────────────────────────────────────
    initial_perception_mode: str = "latent"  # "latent" | "spatial" | "none"
    spatial_pool_size: int = 4               # NxN for spatial mode (4 → 16 tokens)
    # ── Training ─────────────────────────────────────────────────────────────
    perception_dropout: float = 0.0
    # ── Dtype ────────────────────────────────────────────────────────────────
    torch_dtype: str = "bfloat16"


class ActivePerceptionModel(nn.Module):
    """
    Wraps Qwen2_5_VLForConditionalGeneration with the active perception mechanism.
    """

    def __init__(self, config: ActivePerceptionConfig):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.torch_dtype)

        # ── Load base model and processor ────────────────────────────────────
        logger.info(f"[ActivePerception] Loading base model from {config.model_path}")
        self.base_model, self.processor = self._load_base_model(config.model_path)
        self.tokenizer = self.processor.tokenizer

        # ── Add special tokens (initializes new embedding rows in the table) ─
        logger.info("[ActivePerception] Adding special tokens")
        self.special_tokens = add_special_tokens_to_model(
            self.base_model, self.tokenizer, config.token_init_strategy
        )
        logger.info(f"[ActivePerception] {self.special_tokens}")

        # ── Learnable special token embeddings (Issue 4 fix) ─────────────────
        # Extract initialized values from the embedding table, then freeze the
        # table. Only these 4 rows are trainable, via an external Parameter.
        # Adam stores optimizer states for ~14k params instead of ~545M.
        embed = self.base_model.get_input_embeddings()
        new_ids = [
            self.special_tokens.IMAGE,
            self.special_tokens.PERCEPTION,
            self.special_tokens.PERC_OUT,
            self.special_tokens.INIT_PERC_OUT,
        ]
        with torch.no_grad():
            init_embeds = embed.weight.data[new_ids].clone().float()  # float32 for optimizer stability
        self.special_token_embeddings = nn.Parameter(init_embeds)  # [4, d_model]

        # ── Cache important token IDs ─────────────────────────────────────────
        self.image_pad_id = self._find_image_pad_id()
        self.vision_start_id = self._find_vision_start_id()
        self.vision_end_id = self._find_vision_end_id()

        # ── Perception module ─────────────────────────────────────────────────
        d_llm = self.base_model.config.hidden_size
        self.perception_module = PerceptionModule(
            d_model=d_llm,
            d_query=config.d_query,
            num_heads=config.num_perception_heads,
            dropout=config.perception_dropout,
        ).to(self.dtype)  # match LLM dtype so hidden states flow in without casting

        # ── Spatial encoding (Issue 2 fix: pass merge_size from model config) ─
        merge_size = getattr(
            self.base_model.config.vision_config, "spatial_merge_size", 2
        )
        self.spatial_encoding = SpatialEncoding2D(
            d_model=d_llm,
            mode=SpatialEncodingMode(config.spatial_encoding_mode),
            merge_size=merge_size,
        )
        logger.info(f"[ActivePerception] Vision merger spatial_merge_size={merge_size}")

        # ── Apply freezing ────────────────────────────────────────────────────
        self._apply_freezing()

        # ── Optional LoRA ─────────────────────────────────────────────────────
        if config.use_lora:
            self._apply_lora()

        self._log_trainable_params()

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_base_model(self, model_path: str):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
            from transformers import AutoProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=None,
            attn_implementation=self.config.attn_implementation,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return model, processor

    # =========================================================================
    # Freezing and LoRA
    # =========================================================================

    def _apply_freezing(self):
        cfg = self.config

        # Always freeze the full embedding table (Issue 4 fix).
        # Special tokens are trained via self.special_token_embeddings (nn.Parameter).
        self.base_model.get_input_embeddings().weight.requires_grad_(False)
        logger.info("[ActivePerception] Embedding table frozen; special tokens via external Parameter")

        if cfg.freeze_vit:
            for p in self.base_model.visual.parameters():
                p.requires_grad_(False)
            logger.info("[ActivePerception] ViT frozen")

        if cfg.freeze_projector:
            if hasattr(self.base_model.visual, "merger"):
                for p in self.base_model.visual.merger.parameters():
                    p.requires_grad_(False)
                logger.info("[ActivePerception] Visual merger/projector frozen")

        if cfg.freeze_llm:
            for p in self.base_model.model.parameters():
                p.requires_grad_(False)
            for p in self.base_model.lm_head.parameters():
                p.requires_grad_(False)
            logger.info("[ActivePerception] LLM frozen")

    def _apply_lora(self):
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError("peft required for LoRA: pip install peft")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        logger.info(
            f"[ActivePerception] LoRA applied: r={self.config.lora_rank}, "
            f"alpha={self.config.lora_alpha}, targets={self.config.lora_target_modules}"
        )

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[ActivePerception] Parameters: total={total/1e6:.1f}M, "
            f"trainable={trainable/1e6:.2f}M ({100*trainable/total:.3f}%)"
        )

    # =========================================================================
    # Embedding helper (Issue 4 fix)
    # =========================================================================

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input_ids using the frozen embedding table, then inject the 3
        learnable special-token embeddings at positions where input_ids equals
        IMAGE, PERCEPTION, or PERC_OUT.

        The frozen table handles all 152k vocab tokens; only the 3 special rows
        are overwritten with values from self.special_token_embeddings (float32
        Parameter, cast to model dtype at injection).
        """
        embeds = self.base_model.get_input_embeddings()(input_ids)  # [..., T, D]

        tids = [
            self.special_tokens.IMAGE,
            self.special_tokens.PERCEPTION,
            self.special_tokens.PERC_OUT,
            self.special_tokens.INIT_PERC_OUT,
        ]
        # Fast path: skip clone if no special tokens present
        if not any((input_ids == tid).any() for tid in tids):
            return embeds

        embeds = embeds.clone()
        for i, tid in enumerate(tids):
            mask = (input_ids == tid)
            if mask.any():
                embeds[mask] = self.special_token_embeddings[i].to(
                    dtype=embeds.dtype, device=embeds.device
                )
        return embeds

    # =========================================================================
    # Visual memory encoding
    # =========================================================================

    @torch.no_grad()
    def encode_image_to_memory(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run ViT + projector and return post-merger visual tokens as external memory.

        Returns:
            visual_memory: [N_actual, D_llm]
            where N_actual = (H // merge_size) * (W // merge_size)
        """
        pixel_values = pixel_values.to(self.dtype)
        visual_out = self.base_model.visual(pixel_values, grid_thw=grid_thw)
        return visual_out.detach()

    # =========================================================================
    # Sequence surgery: remove visual patch tokens → single <IMAGE> token
    # =========================================================================

    def build_modified_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace the image-pad token block with a single <IMAGE> token per sequence.
        Recomputes position_ids for the M-RoPE-consistent modified sequence.

        Returns:
            mod_input_ids:      [B, T_new]
            mod_attention_mask: [B, T_new]
            mod_labels:         [B, T_new]
            position_ids:       [B, 3, T_new]
        """
        B, T_orig = input_ids.shape

        mod_ids_list, mod_mask_list, mod_labels_list, pos_ids_list = [], [], [], []

        for b in range(B):
            ids = input_ids[b]
            mask = attention_mask[b]
            lab = labels[b]

            if self.image_pad_id is not None:
                pad_mask = (ids == self.image_pad_id)
                n_pad = pad_mask.sum().item()
            else:
                pad_mask = torch.zeros_like(ids, dtype=torch.bool)
                n_pad = 0

            if n_pad == 0:
                new_ids = ids
                new_mask = mask
                new_lab = lab
            else:
                first_pad = pad_mask.nonzero(as_tuple=True)[0][0].item()
                new_ids = torch.cat([
                    ids[:first_pad],
                    torch.tensor([self.special_tokens.IMAGE], dtype=ids.dtype, device=ids.device),
                    ids[first_pad + n_pad:],
                ])
                new_mask = torch.cat([
                    mask[:first_pad],
                    torch.ones(1, dtype=mask.dtype, device=mask.device),
                    mask[first_pad + n_pad:],
                ])
                new_lab = torch.cat([
                    lab[:first_pad],
                    torch.full((1,), -100, dtype=lab.dtype, device=lab.device),
                    lab[first_pad + n_pad:],
                ])

            T_new = new_ids.shape[0]

            if debug:
                logger.debug(
                    f"[Surgery] b={b}: T_orig={T_orig}, n_pad={n_pad}, T_new={T_new}, "
                    f"PERCEPTION_pos={((new_ids == self.special_tokens.PERCEPTION).nonzero()).squeeze(-1).tolist()}, "
                    f"PERC_OUT_pos={((new_ids == self.special_tokens.PERC_OUT).nonzero()).squeeze(-1).tolist()}"
                )

            pos_ids = self._compute_position_ids(new_ids, T_new, device=ids.device)

            mod_ids_list.append(new_ids)
            mod_mask_list.append(new_mask)
            mod_labels_list.append(new_lab)
            pos_ids_list.append(pos_ids)

        T_max = max(x.shape[0] for x in mod_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0

        def pad_1d(t, length, val):
            p = length - t.shape[0]
            return torch.cat([t, torch.full((p,), val, dtype=t.dtype, device=t.device)])

        mod_input_ids = torch.stack([pad_1d(x, T_max, pad_id) for x in mod_ids_list])
        mod_attention_mask = torch.stack([pad_1d(x, T_max, 0) for x in mod_mask_list])
        mod_labels = torch.stack([pad_1d(x, T_max, -100) for x in mod_labels_list])
        position_ids = torch.stack([
            F.pad(p, (0, T_max - p.shape[1]), value=0) for p in pos_ids_list
        ])

        return mod_input_ids, mod_attention_mask, mod_labels, position_ids

    def _compute_position_ids(
        self, ids: torch.Tensor, T: int, device: torch.device
    ) -> torch.Tensor:
        """
        Compute 3D M-RoPE position IDs for the modified (text-only) sequence.
        All 3 dims get the same sequential value: [t, t, t] for t in 0..T-1.
        Returns: [3, T]
        """
        positions = torch.arange(T, device=device, dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).contiguous()

    # =========================================================================
    # Two-pass training forward
    # =========================================================================

    def training_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        perc_positions: List[List[int]],
        perc_out_positions: List[List[int]],
        debug: bool = False,
        **_,  # absorb extra batch metadata (bboxes, sample_ids, sources, etc.)
    ) -> Dict[str, torch.Tensor]:
        """
        Full two-pass training step with initial_perception_mode dispatch.

        initial_perception_mode:
          "latent"  — <INIT_PERC_OUT> embedding replaced by z_init, computed via
                      PerceptionModule queried from the <IMAGE> hidden state (Pass 1).
          "spatial" — <INIT_PERC_OUT> expanded to spatial_pool_size² positions;
                      each receives one adaptively pooled visual patch. Pass 1 sees
                      the spatial context so h_perception benefits from it.
          "none"    — <INIT_PERC_OUT> removed entirely; blind CoT baseline.

        Pass 1 (always no_grad): run LLM → extract h at <IMAGE> and <PERCEPTION>.
        Pass 2 (with grad): inject all visual tokens; compute CE loss.
        """
        mode = self.config.initial_perception_mode
        init_perc_out_id = self.special_tokens.INIT_PERC_OUT

        # ── Step 1: Encode image → visual memory ─────────────────────────────
        visual_memory = self.encode_image_to_memory(pixel_values, image_grid_thw)

        # ── Step 2: Sequence surgery (image_pad block → <IMAGE>) ─────────────
        mod_ids, mod_mask, mod_labels, position_ids = self.build_modified_sequence(
            input_ids, attention_mask, labels, debug=debug
        )

        # ── Step 3: Mode-specific structural changes to the sequence ──────────
        spatial_tokens: Optional[torch.Tensor] = None
        if mode == "none":
            mod_ids, mod_mask, mod_labels, position_ids = self._remove_token_from_sequences(
                init_perc_out_id, mod_ids, mod_mask, mod_labels, position_ids
            )
        elif mode == "spatial":
            spatial_tokens = self._pool_visual_memory(visual_memory, image_grid_thw)
            n_spatial = self.config.spatial_pool_size ** 2
            mod_ids, mod_mask, mod_labels, position_ids = self._splice_spatial_tokens(
                init_perc_out_id, n_spatial, mod_ids, mod_mask, mod_labels, position_ids
            )

        B, T_new = mod_ids.shape
        device = mod_ids.device

        # ── Step 4: Build Pass 1 embeddings ──────────────────────────────────
        base_embeds = self._embed(mod_ids).clone().to(self.dtype)

        # Zero <PERC_OUT> in Pass 1 (causal: comes after <PERCEPTION>, but safer)
        base_embeds[mod_ids == self.special_tokens.PERC_OUT] = 0.0

        if mode == "latent":
            # z_init not yet computed — zero the placeholder
            base_embeds[mod_ids == init_perc_out_id] = 0.0
        elif mode == "spatial" and spatial_tokens is not None:
            # Inject spatial tokens into Pass 1 so h_perception sees visual context
            pooled_p1 = spatial_tokens.to(dtype=base_embeds.dtype, device=device)
            for b in range(B):
                init_pos = (mod_ids[b] == init_perc_out_id).nonzero(as_tuple=True)[0]
                if len(init_pos) > 0:
                    base_embeds[b, init_pos, :] = pooled_p1

        # ── Pass 1: Extract hidden states (always no_grad) ────────────────────
        with torch.no_grad():
            pass1_out = self._get_inner_transformer()(
                inputs_embeds=base_embeds,
                attention_mask=mod_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        hs_pass1 = pass1_out.last_hidden_state  # [B, T_new, D]

        # ── Step 5: Visual memory for cross-attention ─────────────────────────
        vm = visual_memory.unsqueeze(0)  # [1, N_actual, D]
        if self.config.spatial_encoding_mode != "none":
            vm = self.spatial_encoding(vm, image_grid_thw)

        # ── Step 6: Compute z_init for "latent" mode ─────────────────────────
        z_init_by_batch: Dict[int, torch.Tensor] = {}
        if mode == "latent":
            for b in range(B):
                img_pos = (mod_ids[b] == self.special_tokens.IMAGE).nonzero(as_tuple=True)[0]
                if len(img_pos) == 0:
                    continue
                h_img = hs_pass1[b, img_pos[0].item():img_pos[0].item() + 1, :].unsqueeze(0)
                z_init, _ = self.perception_module(h_img, vm, return_attn_weights=False)
                z_init_by_batch[b] = z_init.squeeze(0).squeeze(0)  # [D]

        # ── Step 7: PerceptionModule for mid-CoT <PERCEPTION> tokens ─────────
        z_perceptions: List[Optional[torch.Tensor]] = []
        attn_weights_list: List[Optional[torch.Tensor]] = []

        for b in range(B):
            b_perc_pos = (mod_ids[b] == self.special_tokens.PERCEPTION).nonzero(
                as_tuple=True
            )[0].tolist()

            if not b_perc_pos:
                z_perceptions.append(None)
                attn_weights_list.append(None)
                continue

            h_p = hs_pass1[b, b_perc_pos, :].unsqueeze(0)  # [1, K, D]
            z, attn_w = self.perception_module(h_p, vm, return_attn_weights=True)
            z_perceptions.append(z.squeeze(0))
            attn_weights_list.append(attn_w.squeeze(0) if attn_w is not None else None)

        # ── Pass 2: Inject all visual tokens; full forward for CE ────────────
        embeds_pass2 = self._embed(mod_ids).to(self.dtype)

        if mode == "latent":
            for b, z_init in z_init_by_batch.items():
                init_pos = (mod_ids[b] == init_perc_out_id).nonzero(as_tuple=True)[0]
                if len(init_pos) > 0:
                    embeds_pass2[b, init_pos[0].item(), :] = z_init.to(self.dtype)

        elif mode == "spatial" and spatial_tokens is not None:
            pooled_p2 = spatial_tokens.to(self.dtype)
            for b in range(B):
                init_pos = (mod_ids[b] == init_perc_out_id).nonzero(as_tuple=True)[0].tolist()
                for k, pos in enumerate(init_pos):
                    embeds_pass2[b, pos, :] = pooled_p2[k]

        for b in range(B):
            if z_perceptions[b] is None:
                continue
            b_perc_out_pos = (mod_ids[b] == self.special_tokens.PERC_OUT).nonzero(
                as_tuple=True
            )[0].tolist()
            z = z_perceptions[b]
            K = z.shape[0]
            assert len(b_perc_out_pos) == K, (
                f"[TwoPass] b={b}: {len(b_perc_out_pos)} PERC_OUT positions "
                f"but {K} z_perception vectors"
            )
            for k, pos in enumerate(b_perc_out_pos):
                embeds_pass2[b, pos, :] = z[k].to(self.dtype)

        pass2_out = self.base_model(
            inputs_embeds=embeds_pass2,
            attention_mask=mod_mask,
            labels=mod_labels,
            output_hidden_states=False,
            use_cache=False,
        )

        return {
            "loss_ce": pass2_out.loss,
            "logits": pass2_out.logits,
            "z_perceptions": z_perceptions,
            "attn_weights_list": attn_weights_list,
            "modified_input_ids": mod_ids,
            "modified_labels": mod_labels,
        }

    # =========================================================================
    # Inference
    # =========================================================================

    @torch.no_grad()
    def generate_with_perception(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation with active perception interception.

        When the model emits <PERCEPTION>:
          1. Extract h_perception from the current hidden state.
          2. Run perception module → z_perception.
          3. Feed z_perception as the next token embedding.
          4. Continue generation.
        """
        device = input_ids.device
        mode = self.config.initial_perception_mode
        init_perc_out_id = self.special_tokens.INIT_PERC_OUT

        visual_memory = self.encode_image_to_memory(pixel_values, image_grid_thw)
        vm = visual_memory.unsqueeze(0)
        if self.config.spatial_encoding_mode != "none":
            vm = self.spatial_encoding(vm, image_grid_thw)

        labels_dummy = input_ids.clone()
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        mod_ids, mod_mask, dummy_lab, position_ids = self.build_modified_sequence(
            input_ids, mask, labels_dummy
        )

        # Handle initial_perception_mode structural changes
        if mode == "none":
            mod_ids, mod_mask, dummy_lab, position_ids = self._remove_token_from_sequences(
                init_perc_out_id, mod_ids, mod_mask, dummy_lab, position_ids
            )
        elif mode == "spatial":
            spatial_tokens = self._pool_visual_memory(visual_memory, image_grid_thw)
            n_spatial = self.config.spatial_pool_size ** 2
            mod_ids, mod_mask, dummy_lab, position_ids = self._splice_spatial_tokens(
                init_perc_out_id, n_spatial, mod_ids, mod_mask, dummy_lab, position_ids
            )

        current_embeds = self._embed(mod_ids).to(self.dtype)

        # Inject initial visual tokens into the starting embeddings
        if mode == "spatial":
            pooled = spatial_tokens.to(self.dtype)
            for b in range(input_ids.shape[0]):
                init_pos = (mod_ids[b] == init_perc_out_id).nonzero(as_tuple=True)[0].tolist()
                for k, pos in enumerate(init_pos):
                    current_embeds[b, pos, :] = pooled[k]
        elif mode == "latent":
            # Warm-up pass: zero INIT_PERC_OUT → get h at IMAGE → compute z_init
            warm_embeds = current_embeds.clone()
            warm_embeds[mod_ids == init_perc_out_id] = 0.0
            warm_out = self._get_inner_transformer()(
                inputs_embeds=warm_embeds,
                attention_mask=mod_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            hs_warm = warm_out.last_hidden_state
            for b in range(input_ids.shape[0]):
                img_pos = (mod_ids[b] == self.special_tokens.IMAGE).nonzero(as_tuple=True)[0]
                if len(img_pos) == 0:
                    continue
                h_img = hs_warm[b, img_pos[0].item():img_pos[0].item() + 1, :].unsqueeze(0)
                z_init, _ = self.perception_module(h_img, vm, return_attn_weights=False)
                init_pos = (mod_ids[b] == init_perc_out_id).nonzero(as_tuple=True)[0]
                if len(init_pos) > 0:
                    current_embeds[b, init_pos[0].item(), :] = (
                        z_init.squeeze(0).squeeze(0).to(self.dtype)
                    )

        generated_ids = []
        past_key_values = None

        for step in range(max_new_tokens):
            out = self._get_inner_transformer()(
                inputs_embeds=current_embeds,
                attention_mask=mod_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = out.last_hidden_state
            past_key_values = out.past_key_values

            logits = self.base_model.lm_head(hidden[:, -1:, :])

            if do_sample and temperature != 1.0:
                logits = logits / temperature
            next_token_ids = logits[:, 0, :].argmax(dim=-1)

            perc_emitted = (next_token_ids == self.special_tokens.PERCEPTION)

            if perc_emitted.any():
                h_perc = hidden[:, -1, :]

                next_embeds_list = []
                for b in range(input_ids.shape[0]):
                    if perc_emitted[b]:
                        z, _ = self.perception_module(
                            h_perception=h_perc[b:b+1].unsqueeze(1),
                            visual_memory=vm,
                        )
                        next_embed = z.squeeze(1).to(self.dtype)
                        if len(generated_ids) <= step:
                            generated_ids.append(
                                torch.full((input_ids.shape[0],), -1, device=device)
                            )
                        generated_ids[step][b] = self.special_tokens.PERCEPTION
                        next_embeds_list.append(next_embed)
                    else:
                        tok_embed = self._embed(next_token_ids[b:b+1]).to(self.dtype)
                        next_embeds_list.append(tok_embed)

                current_embeds = torch.stack(
                    [e.squeeze(0) for e in next_embeds_list], dim=0
                ).unsqueeze(1)
            else:
                if len(generated_ids) <= step:
                    generated_ids.append(next_token_ids)
                else:
                    generated_ids[step] = next_token_ids

                current_embeds = self._embed(next_token_ids.unsqueeze(1)).to(self.dtype)

            new_mask_col = torch.ones(
                input_ids.shape[0], 1, dtype=mod_mask.dtype, device=device
            )
            mod_mask = torch.cat([mod_mask, new_mask_col], dim=1)

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (next_token_ids == eos_id).all():
                break

        if not generated_ids:
            return torch.zeros(input_ids.shape[0], 0, dtype=torch.long, device=device)
        return torch.stack(generated_ids, dim=1)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _find_image_pad_id(self) -> Optional[int]:
        for cand in ["<|image_pad|>", "<image_pad>"]:
            tid = self.tokenizer.convert_tokens_to_ids(cand)
            if tid != self.tokenizer.unk_token_id:
                return tid
        return None

    def _find_vision_start_id(self) -> Optional[int]:
        for cand in ["<|vision_start|>", "<vision_start>"]:
            tid = self.tokenizer.convert_tokens_to_ids(cand)
            if tid != self.tokenizer.unk_token_id:
                return tid
        return None

    def _find_vision_end_id(self) -> Optional[int]:
        for cand in ["<|vision_end|>", "<vision_end>"]:
            tid = self.tokenizer.convert_tokens_to_ids(cand)
            if tid != self.tokenizer.unk_token_id:
                return tid
        return None

    # =========================================================================
    # Initial perception helpers
    # =========================================================================

    def _pool_visual_memory(
        self,
        visual_memory: torch.Tensor,  # [N, D]
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive average pool visual_memory to spatial_pool_size × spatial_pool_size.

        Returns: [spatial_pool_size², D]
        """
        merge_size = getattr(self.base_model.config.vision_config, "spatial_merge_size", 2)
        if grid_thw.dim() == 2:
            _, H_pre, W_pre = grid_thw[0].int().tolist()
        else:
            _, H_pre, W_pre = grid_thw.int().tolist()
        H = int(H_pre) // merge_size
        W = int(W_pre) // merge_size

        _, D = visual_memory.shape
        P = self.config.spatial_pool_size

        vm_grid = visual_memory.reshape(H, W, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        pooled = F.adaptive_avg_pool2d(vm_grid.float(), (P, P))                  # [1, D, P, P]
        return pooled.squeeze(0).permute(1, 2, 0).reshape(P * P, D).to(visual_memory.dtype)

    def _repad_sequences(
        self,
        ids_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        lab_list: List[torch.Tensor],
        pos_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-pad a list of variable-length sequence tensors to max length."""
        T_max = max(x.shape[0] for x in ids_list)
        pad_id = self.tokenizer.pad_token_id or 0
        device = ids_list[0].device

        def pad1d(t, val):
            p = T_max - t.shape[0]
            if p == 0:
                return t
            return torch.cat([t, torch.full((p,), val, dtype=t.dtype, device=device)])

        def pad_pos(p):
            pad = T_max - p.shape[1]
            if pad == 0:
                return p
            return F.pad(p, (0, pad), value=0)

        return (
            torch.stack([pad1d(x, pad_id) for x in ids_list]),
            torch.stack([pad1d(x, 0) for x in mask_list]),
            torch.stack([pad1d(x, -100) for x in lab_list]),
            torch.stack([pad_pos(p) for p in pos_list]),
        )

    def _remove_token_from_sequences(
        self,
        token_id: int,
        mod_ids: torch.Tensor,
        mod_mask: torch.Tensor,
        mod_labels: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove all occurrences of token_id from each sequence and re-pad."""
        B = mod_ids.shape[0]
        ids_list, mask_list, lab_list, pos_list = [], [], [], []
        for b in range(B):
            keep = mod_ids[b] != token_id
            ids_list.append(mod_ids[b][keep])
            mask_list.append(mod_mask[b][keep])
            lab_list.append(mod_labels[b][keep])
            pos_list.append(position_ids[b][:, keep])
        return self._repad_sequences(ids_list, mask_list, lab_list, pos_list)

    def _splice_spatial_tokens(
        self,
        token_id: int,
        n_tokens: int,
        mod_ids: torch.Tensor,
        mod_mask: torch.Tensor,
        mod_labels: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expand the first occurrence of token_id in each sequence to n_tokens copies.

        The placeholder IDs remain as token_id repeated n_tokens times so that
        training_forward can locate the injection positions via (mod_ids == token_id).
        Labels for all n_tokens positions are set to -100.
        Position IDs are recomputed sequentially for the expanded sequence.
        """
        B = mod_ids.shape[0]
        ids_list, mask_list, lab_list, pos_list = [], [], [], []
        for b in range(B):
            ids = mod_ids[b]
            mask = mod_mask[b]
            lab = mod_labels[b]

            init_pos = (ids == token_id).nonzero(as_tuple=True)[0]
            if len(init_pos) == 0:
                ids_list.append(ids)
                mask_list.append(mask)
                lab_list.append(lab)
                pos_list.append(position_ids[b])
                continue

            p = init_pos[0].item()
            rep = torch.full((n_tokens,), token_id, dtype=ids.dtype, device=ids.device)
            new_ids = torch.cat([ids[:p], rep, ids[p + 1:]])
            new_mask = torch.cat([
                mask[:p],
                torch.ones(n_tokens, dtype=mask.dtype, device=mask.device),
                mask[p + 1:],
            ])
            new_lab = torch.cat([
                lab[:p],
                torch.full((n_tokens,), -100, dtype=lab.dtype, device=lab.device),
                lab[p + 1:],
            ])
            T_new = new_ids.shape[0]
            new_pos = torch.arange(T_new, device=ids.device, dtype=torch.long)
            new_pos = new_pos.unsqueeze(0).expand(3, -1).contiguous()

            ids_list.append(new_ids)
            mask_list.append(new_mask)
            lab_list.append(new_lab)
            pos_list.append(new_pos)

        return self._repad_sequences(ids_list, mask_list, lab_list, pos_list)

    def _get_inner_transformer(self):
        """
        Return the inner Qwen2VLModel (backbone without lm_head), unwrapping any
        peft wrapper.

        Without LoRA: self.base_model = Qwen2_5_VLForConditionalGeneration
                      .model = Qwen2_5_VLModel  ✓
        With LoRA:    self.base_model = PeftModelForCausalLM
                      .base_model    = LoraModel
                      .model         = Qwen2_5_VLForConditionalGeneration
                      .model         = Qwen2_5_VLModel  ✓
        """
        m = self.base_model
        try:
            from peft import PeftModel
            if isinstance(m, PeftModel):
                m = m.base_model.model  # LoraModel.model = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            pass
        return m.model

    def get_special_token_ids(self) -> Dict[str, int]:
        return self.special_tokens.as_dict()

    def save_perception_module(self, path: str):
        """Save perception module, spatial encoding, and special token embeddings."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.perception_module.state_dict(), f"{path}/perception_module.pt")
        torch.save(self.spatial_encoding.state_dict(), f"{path}/spatial_encoding.pt")
        torch.save(self.special_token_embeddings.data, f"{path}/special_token_embeddings.pt")
        logger.info(f"[ActivePerception] Perception module saved to {path}")

    def load_perception_module(self, path: str):
        """Load perception module, spatial encoding, and special token embeddings."""
        self.perception_module.load_state_dict(
            torch.load(f"{path}/perception_module.pt", map_location="cpu")
        )
        spa_path = f"{path}/spatial_encoding.pt"
        if os.path.exists(spa_path) and hasattr(self.spatial_encoding, "proj"):
            self.spatial_encoding.load_state_dict(
                torch.load(spa_path, map_location="cpu")
            )
        sp_path = f"{path}/special_token_embeddings.pt"
        if os.path.exists(sp_path):
            self.special_token_embeddings.data.copy_(
                torch.load(sp_path, map_location="cpu")
            )
        logger.info(f"[ActivePerception] Perception module loaded from {path}")
