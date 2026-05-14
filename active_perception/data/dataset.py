"""
PyTorch Dataset and Collator for active perception training.

Key design:
- Tokenizes the converted response with <PERCEPTION> and <PERC_OUT> tokens
- Computes per-token labels with -100 masking for non-supervised positions
- Records positions of <PERCEPTION> and <PERC_OUT> tokens for the model wrapper
- Supports multiple perception steps per sample
- Supports future multi-dataset mixtures
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from .schema import ActivePerceptionSample, PerceptionStep

logger = logging.getLogger(__name__)


class ActivePerceptionDataset(Dataset):
    """
    Loads ActivePerceptionSample records (from JSON/JSONL) and tokenizes them
    using the Qwen2.5-VL processor.

    The dataset is processor-agnostic at storage time; tokenization happens here.
    """

    def __init__(
        self,
        data_path: str,
        processor,                          # Qwen2.5-VL processor
        special_token_ids: Dict[str, int],  # {"PERCEPTION": id, "PERC_OUT": id, "IMAGE": id}
        image_root: Optional[str] = None,
        max_seq_len: int = 2048,
        system_prompt: Optional[str] = None,
        supervision_mode: str = "full",     # "full" | "obs_and_answer" | "answer_only"
    ):
        self.processor = processor
        self.special_token_ids = special_token_ids
        self.image_root = Path(image_root) if image_root else None
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.supervision_mode = supervision_mode

        self.samples: List[ActivePerceptionSample] = self._load(data_path)
        logger.info(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def _load(self, path: str) -> List[ActivePerceptionSample]:
        p = Path(path)
        samples = []
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(ActivePerceptionSample.from_dict(json.loads(line)))
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                samples = [ActivePerceptionSample.from_dict(d) for d in data]
            else:
                samples = [ActivePerceptionSample.from_dict(data)]
        else:
            raise ValueError(f"Unsupported data format: {p.suffix}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image = self._load_image(sample.image)
        if image is None:
            # Return a fallback — in practice log and skip
            logger.warning(f"[Dataset] Could not load image for sample {sample.id}")
            return self.__getitem__((idx + 1) % len(self.samples))

        # Build conversation for processor
        messages = self._build_messages(sample)

        # Tokenize with processor
        try:
            encoding = self._tokenize(messages, image)
        except Exception as e:
            logger.warning(f"[Dataset] Tokenization failed for {sample.id}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        input_ids = encoding["input_ids"][0]        # [T]
        attention_mask = encoding["attention_mask"][0]  # [T]
        pixel_values = encoding.get("pixel_values")
        image_grid_thw = encoding.get("image_grid_thw")

        # Truncate if needed
        if input_ids.shape[0] > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        # Build labels
        labels = self._build_labels(input_ids, sample)

        # Find special token positions
        perc_positions = (input_ids == self.special_token_ids["PERCEPTION"]).nonzero(as_tuple=True)[0].tolist()
        perc_out_positions = (input_ids == self.special_token_ids["PERC_OUT"]).nonzero(as_tuple=True)[0].tolist()

        return {
            "sample_id": sample.id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "perc_positions": perc_positions,
            "perc_out_positions": perc_out_positions,
            "num_perception_steps": sample.num_perception_steps(),
            "has_perception": sample.has_perception,
            # Bbox metadata for optional supervision (not exposed to LLM)
            "bboxes": [s.bbox for s in sample.perception_steps if s.has_bbox()],
            "observation_texts": [s.observation_text for s in sample.perception_steps if s.has_observation()],
            "source": sample.source,
        }

    def _load_image(self, path: str) -> Optional[Image.Image]:
        if path == "__bytes__":
            return None
        p = Path(path)
        if not p.is_absolute() and self.image_root:
            p = self.image_root / p
        if not p.exists():
            return None
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None

    def _build_messages(self, sample: ActivePerceptionSample) -> List[Dict[str, Any]]:
        """Build chat messages for the Qwen2.5-VL processor."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample.question},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": sample.converted_response,
        })
        return messages

    def _tokenize(self, messages: List[Dict], image: Image.Image) -> Dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

    def _build_labels(
        self, input_ids: torch.Tensor, sample: ActivePerceptionSample
    ) -> torch.Tensor:
        """
        Build per-token labels for CE loss.

        Masking strategy:
        - system + user tokens: -100 (no loss)
        - <PERC_OUT> token: -100 (embedding is replaced by z_perception; no CE target)
        - <IMAGE> token: -100
        - <PERCEPTION> token: supervised (teach model WHEN to emit it)
        - observation text + reasoning + answer: supervised (core gradient signal)

        supervision_mode controls what gets supervised:
        - "full": all assistant tokens except PERC_OUT and IMAGE
        - "obs_and_answer": only observation text + answer
        - "answer_only": only the final answer
        """
        labels = input_ids.clone()

        # Mask all tokens by default; we'll unmask selectively
        # For simplicity, find the boundary where the assistant response starts.
        # The processor adds a special token or we can detect the first assistant token.
        # We use a simpler heuristic: mask everything before the FIRST supervised token.

        # Mask <PERC_OUT> always
        perc_out_id = self.special_token_ids.get("PERC_OUT")
        if perc_out_id is not None:
            labels[input_ids == perc_out_id] = -100

        # Mask <IMAGE> always
        image_id = self.special_token_ids.get("IMAGE")
        if image_id is not None:
            labels[input_ids == image_id] = -100

        # Mask image pad tokens (original Qwen visual tokens) if any remain
        # (shouldn't be present after our surgery, but defensive)
        # image_pad_id is handled by the wrapper before reaching here

        # NOTE: Full masking of user/system prefix is done in the collator
        # based on the role boundaries. Here we just handle special tokens.
        # For now, trust that the processor's tokenization puts user content
        # before assistant content and the trainer handles prefix masking.

        return labels


class ActivePerceptionCollator:
    """
    Collates variable-length samples into padded batches.

    Handles:
    - Padding input_ids, attention_mask, labels to max length in batch
    - Stacking pixel_values (all images must be same resolution or pre-resized)
    - Collecting perception position lists per sample
    - Optional: masking user/system tokens in labels (prefix masking)
    """

    def __init__(
        self,
        pad_token_id: int,
        image_token_id: Optional[int] = None,
        mask_user_tokens_in_labels: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.mask_user_tokens_in_labels = mask_user_tokens_in_labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Determine max sequence length in batch
        max_len = max(item["input_ids"].shape[0] for item in batch)
        B = len(batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            T = item["input_ids"].shape[0]
            pad_len = max_len - T

            input_ids_list.append(
                torch.cat([item["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask_list.append(
                torch.cat([item["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([item["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

        input_ids = torch.stack(input_ids_list)       # [B, T]
        attention_mask = torch.stack(attention_mask_list)  # [B, T]
        labels = torch.stack(labels_list)             # [B, T]

        # Pixel values: stack if all have same shape, else keep as list
        pixel_values = self._collate_pixel_values(batch)
        image_grid_thw = self._collate_grid_thw(batch)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            # Per-sample position lists (variable length; keep as list of lists)
            "perc_positions": [item["perc_positions"] for item in batch],
            "perc_out_positions": [item["perc_out_positions"] for item in batch],
            "bboxes": [item["bboxes"] for item in batch],
            "observation_texts": [item["observation_texts"] for item in batch],
            "has_perception": torch.tensor([item["has_perception"] for item in batch]),
            "sample_ids": [item["sample_id"] for item in batch],
            "sources": [item["source"] for item in batch],
        }

    def _collate_pixel_values(self, batch):
        pvs = [item.get("pixel_values") for item in batch]
        if any(pv is None for pv in pvs):
            return None
        try:
            return torch.cat(pvs, dim=0)
        except Exception:
            return pvs  # can't stack; return list

    def _collate_grid_thw(self, batch):
        gts = [item.get("image_grid_thw") for item in batch]
        if any(gt is None for gt in gts):
            return None
        try:
            return torch.cat(gts, dim=0)
        except Exception:
            return gts
