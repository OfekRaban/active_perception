"""
VGR → ActivePerception format converter.

Actual VGR dataset format (assistant side):
    <think>
    ...reasoning...<SOT>[x1, y1, x2, y2]<EOT><image>...more reasoning...
    </think>
    final answer

    Coordinates are normalized to [0, 1].
    The <image> token always immediately follows <EOT>.
    There is NO observation/replay text after the bbox token.

Converted format:
    <think>
    ...reasoning...<PERCEPTION> <PERC_OUT>...more reasoning...
    </think>
    final answer

Key design decisions:
- BBox coordinates stored as supervision metadata ONLY (never in model text).
- VGR has no observation text — observation_text=None, target_type="none".
- bbox_normalized=True for all VGR steps (coords in [0, 1]).
- The trailing <image> token is consumed as part of the bbox pattern.
- Multi-bbox samples handled transparently via finditer.
"""
from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple

from .schema import ActivePerceptionSample, PerceptionStep

logger = logging.getLogger(__name__)

# VGR actual bbox pattern: <SOT>[x1, y1, x2, y2]<EOT><image>
# Coords are normalized floats in [0, 1]. The <image> token is always present.
_BBOX_RE = re.compile(
    r'<SOT>\[([^\]]+)\]<EOT><image>',
    re.IGNORECASE,
)


def _parse_coords(coord_str: str) -> List[float]:
    """'0.49, 0.57, 0.67, 1.0' → [0.49, 0.57, 0.67, 1.0]"""
    parts = [x.strip() for x in coord_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 coords, got {len(parts)}: {coord_str!r}")
    return [float(x) for x in parts]


class VGRConverter:
    """
    Converts VGR dataset samples to the unified ActivePerceptionSample format.

    Usage:
        converter = VGRConverter(image_root="/path/to/images")
        samples = list(converter.convert_dataset_file("vgr_train.parquet"))
    """

    PERCEPTION_TOKEN = "<PERCEPTION>"
    PERC_OUT_TOKEN = "<PERC_OUT>"

    def __init__(
        self,
        image_root: Optional[str] = None,
        verbose: bool = False,
    ):
        self.image_root = Path(image_root) if image_root else None
        self.verbose = verbose
        self._stats: Dict[str, int] = {
            "total": 0, "converted": 0, "skipped": 0,
            "single_step": 0, "multi_step": 0, "no_bbox": 0,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert_sample(self, raw: Dict[str, Any], sample_id: Optional[str] = None) -> Optional[ActivePerceptionSample]:
        """Convert one raw VGR sample dict to ActivePerceptionSample."""
        self._stats["total"] += 1
        try:
            sample = self._convert(raw, sample_id or str(self._stats["total"]))
            if sample is not None:
                self._stats["converted"] += 1
                n = sample.num_perception_steps()
                if n == 0:
                    self._stats["no_bbox"] += 1
                elif n == 1:
                    self._stats["single_step"] += 1
                else:
                    self._stats["multi_step"] += 1
            else:
                self._stats["skipped"] += 1
            return sample
        except Exception as e:
            logger.warning(f"[VGRConverter] Failed sample {sample_id}: {e}")
            self._stats["skipped"] += 1
            return None

    def convert_dataset_file(self, path: str) -> Iterator[ActivePerceptionSample]:
        """Load and convert a VGR parquet or jsonl file."""
        p = Path(path)
        if p.suffix == ".parquet":
            yield from self._convert_parquet(p)
        elif p.suffix in (".jsonl", ".json"):
            yield from self._convert_jsonl(p)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0

    # ------------------------------------------------------------------
    # Core conversion logic
    # ------------------------------------------------------------------

    def _convert(self, raw: Dict[str, Any], sample_id: str) -> Optional[ActivePerceptionSample]:
        conversations = raw.get("conversations", [])
        if len(conversations) < 2:
            return None

        human_turn = conversations[0]
        gpt_turn = conversations[1]

        if human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt":
            return None

        question = human_turn.get("value", "").replace("<image>", "").strip()
        raw_response = gpt_turn.get("value", "")

        if not raw_response:
            return None

        converted_response, steps = self._parse_and_convert_response(raw_response)
        answer = self._extract_answer(raw_response)
        image_path = self._resolve_image(raw.get("image", ""))

        sample = ActivePerceptionSample(
            id=sample_id,
            image=image_path,
            source="vgr",
            question=question,
            converted_response=converted_response,
            converted_answer=answer,
            perception_steps=steps,
            has_perception=len(steps) > 0,
            metadata={"original_response_len": len(raw_response)},
        )

        if self.verbose:
            logger.debug(
                f"[VGRConverter] id={sample_id} steps={len(steps)} "
                f"resp_len={len(converted_response)}"
            )

        return sample

    def _parse_and_convert_response(
        self, response: str
    ) -> Tuple[str, List[PerceptionStep]]:
        """
        Replace all <SOT>[x1, y1, x2, y2]<EOT><image> occurrences with
        <PERCEPTION> <PERC_OUT> and collect bbox metadata as PerceptionSteps.
        """
        steps: List[PerceptionStep] = []
        parts: List[str] = []
        cursor = 0

        for i, m in enumerate(_BBOX_RE.finditer(response)):
            try:
                bbox = _parse_coords(m.group(1))
            except ValueError as e:
                logger.warning(f"[VGRConverter] Skipping malformed bbox: {e}")
                continue

            parts.append(response[cursor:m.start()])
            parts.append(f"{self.PERCEPTION_TOKEN} {self.PERC_OUT_TOKEN}")
            cursor = m.end()

            steps.append(PerceptionStep(
                index=i,
                bbox=bbox,
                bbox_normalized=True,
                observation_text=None,
                target_type="none",
            ))

        parts.append(response[cursor:])
        return "".join(parts), steps

    def _extract_answer(self, response: str) -> str:
        """Text after </think>, or the full response if no think block."""
        think_end = response.rfind("</think>")
        if think_end == -1:
            return response.strip()
        return response[think_end + len("</think>"):].strip()

    def _resolve_image(self, image_val: Any) -> str:
        if isinstance(image_val, bytes):
            return "__bytes__"
        path = str(image_val)
        if self.image_root and not Path(path).is_absolute():
            return str(self.image_root / path)
        return path

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    def _convert_parquet(self, path: Path) -> Iterator[ActivePerceptionSample]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for parquet reading: pip install pandas pyarrow")
        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            sample = self.convert_sample(row.to_dict(), sample_id=f"vgr_{i}")
            if sample is not None:
                yield sample

    def _convert_jsonl(self, path: Path) -> Iterator[ActivePerceptionSample]:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                sample = self.convert_sample(raw, sample_id=f"vgr_{i}")
                if sample is not None:
                    yield sample
