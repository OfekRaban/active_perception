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
  The 3 new tokens (<IMAGE>, <PERCEPTION>, <PERC_OUT>) are registered as a
  separate nn.Parameter `special_token_embeddings` [3, d_model] instead of
  unfreezing the full 152k-row embedding table. Adam only tracks optimizer
  states for these 3 rows (~10k params) rather than all 545M embedding params.
  The embedding table stays fully frozen. The _embed() helper injects the
  learnable rows at forward time by overwriting positions where input_ids
  equals one of the 3 special token IDs.

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
    # ── Architecture ─────────────────────────────────────────────────────────
    d_query: int = 256
    num_perception_heads: int = 8
    perception_residual: bool = True
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
        # table. Only these 3 rows are trainable, via an external Parameter.
        # Adam stores optimizer states for ~10k params instead of ~545M.
        embed = self.base_model.get_input_embeddings()
        new_ids = [
            self.special_tokens.IMAGE,
            self.special_tokens.PERCEPTION,
            self.special_tokens.PERC_OUT,
        ]
        with torch.no_grad():
            init_embeds = embed.weight.data[new_ids].clone().float()  # float32 for optimizer stability
        self.special_token_embeddings = nn.Parameter(init_embeds)  # [3, d_model]

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
            residual=config.perception_residual,
        )

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
    ) -> Dict[str, torch.Tensor]:
        """
        Full two-pass training step.

        Pass 1 (Issue 3 fix — always no_grad):
          Run LLM to extract h_perception at <PERCEPTION> positions.
          Treating this as stop-gradient avoids holding two full 7B computation
          graphs simultaneously. LoRA gradients still reach the LLM weights
          via Pass 2, so the truncation has negligible empirical impact.

        Pass 2 (with grad):
          Inject z_perception at <PERC_OUT> positions; run full LLM forward
          for CE loss. Gradients flow: CE → LLM (Pass 2) → z_perception →
          PerceptionModule → QueryAdapter → (stop at h_perception).

        Returns dict with:
          loss_ce, logits, z_perceptions, attn_weights_list,
          modified_input_ids, modified_labels
        """
        # ── Step 1: Encode image → visual memory ─────────────────────────────
        visual_memory = self.encode_image_to_memory(pixel_values, image_grid_thw)
        # visual_memory: [N_actual, D]

        # ── Step 2: Sequence surgery ──────────────────────────────────────────
        mod_ids, mod_mask, mod_labels, position_ids = self.build_modified_sequence(
            input_ids, attention_mask, labels, debug=debug
        )

        B, T_new = mod_ids.shape
        device = mod_ids.device

        # ── Step 3: Build Pass 1 embeddings ──────────────────────────────────
        base_embeds = self._embed(mod_ids)           # [B, T_new, D]
        # Zero <PERC_OUT> embeddings so they don't influence h_perception
        # (causal attn: <PERC_OUT> comes after <PERCEPTION>, but zero is safer)
        perc_out_id = self.special_tokens.PERC_OUT
        perc_out_mask_2d = (mod_ids == perc_out_id)
        base_embeds = base_embeds.clone()
        base_embeds[perc_out_mask_2d] = 0.0

        # ── Pass 1: Extract h_perception (Issue 3 fix: always no_grad) ───────
        with torch.no_grad():
            pass1_out = self.base_model.model(
                inputs_embeds=base_embeds,
                attention_mask=mod_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                use_cache=False,
            )
        hs_pass1 = pass1_out.last_hidden_state  # [B, T_new, D]

        # ── Run PerceptionModule per batch item ───────────────────────────────
        z_perceptions = []
        attn_weights_list = []

        vm = visual_memory.unsqueeze(0)  # [1, N_actual, D]
        if self.config.spatial_encoding_mode != "none":
            vm = self.spatial_encoding(vm, image_grid_thw)

        for b in range(B):
            # Re-detect PERCEPTION positions in the modified (surgery-adjusted) sequence
            b_perc_pos = (mod_ids[b] == self.special_tokens.PERCEPTION).nonzero(
                as_tuple=True
            )[0].tolist()

            if not b_perc_pos:
                z_perceptions.append(None)
                attn_weights_list.append(None)
                continue

            K = len(b_perc_pos)
            h_p = hs_pass1[b, b_perc_pos, :]   # [K, D] — stop-gradient input
            h_p_batched = h_p.unsqueeze(0)       # [1, K, D]

            z, attn_w = self.perception_module(
                h_perception=h_p_batched,
                visual_memory=vm,
                return_attn_weights=True,
            )
            z = z.squeeze(0)        # [K, D]
            attn_w = attn_w.squeeze(0) if attn_w is not None else None

            z_perceptions.append(z)
            attn_weights_list.append(attn_w)

        # ── Pass 2: Inject z_perception at <PERC_OUT>, full forward ──────────
        embeds_pass2 = self._embed(mod_ids).to(self.dtype)

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
            position_ids=position_ids,
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

        visual_memory = self.encode_image_to_memory(pixel_values, image_grid_thw)
        vm = visual_memory.unsqueeze(0)
        if self.config.spatial_encoding_mode != "none":
            vm = self.spatial_encoding(vm, image_grid_thw)

        labels_dummy = input_ids.clone()
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        mod_ids, mod_mask, _, position_ids = self.build_modified_sequence(
            input_ids, mask, labels_dummy
        )

        current_embeds = self._embed(mod_ids).to(self.dtype)

        generated_ids = []
        past_key_values = None
        current_pos_ids = position_ids  # [B, 3, T]

        for step in range(max_new_tokens):
            out = self.base_model.model(
                inputs_embeds=current_embeds,
                attention_mask=mod_mask,
                position_ids=current_pos_ids,
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

            T_current = mod_mask.shape[1] + step + 1
            new_mask_col = torch.ones(
                input_ids.shape[0], 1, dtype=mod_mask.dtype, device=device
            )
            mod_mask = torch.cat([mod_mask, new_mask_col], dim=1)

            new_pos = current_pos_ids[:, :, -1:] + 1
            current_pos_ids = new_pos

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
