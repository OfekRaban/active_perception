"""
Special token management for Active Perception.

Adds <IMAGE>, <PERCEPTION>, <PERC_OUT> to the tokenizer and resizes model embeddings.

Token semantics:
  <IMAGE>      — replaces the full visual-token block in the LLM sequence.
                 The model sees only this single token where N visual patches used to be.
                 Visual information is ONLY accessible via <PERCEPTION>.
  <PERCEPTION> — emitted by the model when it decides to query visual memory.
                 Its hidden state h_perception drives the cross-attention retrieval.
  <PERC_OUT>   — placeholder token whose embedding is REPLACED by z_perception
                 before the second-pass LLM forward. Labels at this position are -100.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<IMAGE>", "<PERCEPTION>", "<PERC_OUT>"]


class SpecialTokens:
    """Holds token IDs for our custom special tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.IMAGE: int = tokenizer.convert_tokens_to_ids("<IMAGE>")
        self.PERCEPTION: int = tokenizer.convert_tokens_to_ids("<PERCEPTION>")
        self.PERC_OUT: int = tokenizer.convert_tokens_to_ids("<PERC_OUT>")
        self._validate()

    def _validate(self):
        for name in ("IMAGE", "PERCEPTION", "PERC_OUT"):
            tid = getattr(self, name)
            if tid == self.tokenizer.unk_token_id:
                raise RuntimeError(
                    f"Special token <{name}> not found in tokenizer. "
                    "Call add_special_tokens_to_model() first."
                )

    def as_dict(self) -> Dict[str, int]:
        return {
            "IMAGE": self.IMAGE,
            "PERCEPTION": self.PERCEPTION,
            "PERC_OUT": self.PERC_OUT,
        }

    def __repr__(self) -> str:
        return (
            f"SpecialTokens(IMAGE={self.IMAGE}, "
            f"PERCEPTION={self.PERCEPTION}, "
            f"PERC_OUT={self.PERC_OUT})"
        )


def add_special_tokens_to_model(
    model: nn.Module,
    tokenizer,
    init_strategy: str = "mean_visual_text",
) -> SpecialTokens:
    """
    Add <IMAGE>, <PERCEPTION>, <PERC_OUT> to the tokenizer and resize model embeddings.

    init_strategy controls how new token embeddings are initialized:
      "random"           — default PyTorch random init (often too far from existing space)
      "mean_vocab"       — mean of all existing embeddings (safe, in-distribution)
      "mean_visual_text" — average of image_pad + think tokens (best warm start for PERCEPTION)
      "image_pad"        — copy of existing <|image_pad|> embedding

    Returns a SpecialTokens object with the assigned IDs.
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    logger.info(f"[SpecialTokens] Added {num_added} new special tokens: {SPECIAL_TOKENS}")

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(
        f"[SpecialTokens] Embedding table resized: {old_vocab_size} → {new_vocab_size}"
    )

    st = SpecialTokens(tokenizer)

    _initialize_new_embeddings(model, tokenizer, st, init_strategy)

    return st


def _initialize_new_embeddings(
    model: nn.Module,
    tokenizer,
    st: SpecialTokens,
    strategy: str,
):
    """Initialize embeddings for the new special tokens."""
    embed_layer = model.get_input_embeddings()

    with torch.no_grad():
        if strategy == "mean_vocab":
            mean_embed = embed_layer.weight.mean(dim=0)
            for tid in [st.IMAGE, st.PERCEPTION, st.PERC_OUT]:
                embed_layer.weight[tid] = mean_embed.clone()
            logger.info("[SpecialTokens] Initialized new tokens with mean vocab embedding")

        elif strategy == "mean_visual_text":
            # <PERCEPTION>: blend of <think> (LLM reasoning token) + <|image_pad|>
            think_id = tokenizer.convert_tokens_to_ids("<think>")
            img_pad_id = _find_image_pad_id(tokenizer)

            if think_id != tokenizer.unk_token_id and img_pad_id is not None:
                perception_init = (
                    0.5 * embed_layer.weight[think_id].clone() +
                    0.5 * embed_layer.weight[img_pad_id].clone()
                )
            elif think_id != tokenizer.unk_token_id:
                perception_init = embed_layer.weight[think_id].clone()
            else:
                perception_init = embed_layer.weight.mean(dim=0)

            embed_layer.weight[st.PERCEPTION] = perception_init

            # <IMAGE>: use image_pad embedding if available
            if img_pad_id is not None:
                embed_layer.weight[st.IMAGE] = embed_layer.weight[img_pad_id].clone()
            else:
                embed_layer.weight[st.IMAGE] = embed_layer.weight.mean(dim=0)

            # <PERC_OUT>: initialize to zeros — will always be replaced before LLM sees it
            embed_layer.weight[st.PERC_OUT] = torch.zeros_like(embed_layer.weight[0])

            logger.info("[SpecialTokens] Initialized with mean_visual_text strategy")

        elif strategy == "image_pad":
            img_pad_id = _find_image_pad_id(tokenizer)
            if img_pad_id is None:
                logger.warning("[SpecialTokens] image_pad token not found; falling back to mean_vocab")
                _initialize_new_embeddings(model, tokenizer, st, "mean_vocab")
                return
            for tid in [st.IMAGE, st.PERCEPTION, st.PERC_OUT]:
                embed_layer.weight[tid] = embed_layer.weight[img_pad_id].clone()
            logger.info("[SpecialTokens] Initialized new tokens with image_pad embedding")

        elif strategy == "random":
            logger.info("[SpecialTokens] Using default random initialization for new tokens")
            pass  # already random after resize

        else:
            raise ValueError(f"Unknown init_strategy: {strategy}")

    # Also initialize the output (LM head) projection for new tokens
    # Important: if the LM head is tied to embeddings, this is handled automatically.
    # If not tied, initialize the output rows as well.
    lm_head = _get_lm_head(model)
    if lm_head is not None and lm_head.weight.data_ptr() != embed_layer.weight.data_ptr():
        with torch.no_grad():
            mean_lm = lm_head.weight.mean(dim=0)
            for tid in [st.IMAGE, st.PERCEPTION, st.PERC_OUT]:
                lm_head.weight[tid] = mean_lm.clone()
        logger.info("[SpecialTokens] Initialized LM head rows for new tokens")


def _find_image_pad_id(tokenizer) -> Optional[int]:
    """Find the <|image_pad|> token ID in Qwen2.5-VL tokenizer."""
    candidates = ["<|image_pad|>", "<image_pad>"]
    for cand in candidates:
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid != tokenizer.unk_token_id:
            return tid
    return None


def _get_lm_head(model: nn.Module) -> Optional[nn.Linear]:
    """Get the LM head of the model if it exists and is not tied."""
    for attr in ("lm_head", "embed_out"):
        head = getattr(model, attr, None)
        if head is not None and isinstance(head, nn.Linear):
            return head
    return None
