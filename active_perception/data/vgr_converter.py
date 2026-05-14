"""
VGR → ActivePerception format converter.

VGR dataset format (assistant side):
    <think>
    ...reasoning...<sot>[x1,y1,x2,y2]<eot>
    observation / visual replay text
    ...continued reasoning...
    </think>
    final answer

Converted format:
    <think>
    ...reasoning...<PERCEPTION> <PERC_OUT>
    observation / visual replay text
    ...continued reasoning...
    </think>
    final answer

Key design decisions:
- BBox coordinates are stored as supervision metadata ONLY (never in model text).
- Observation text is PRESERVED after <PERC_OUT>. The CE loss on this text
  is the primary gradient signal into z_perception / cross-attention / Query Adapter.
- Multi-step support: multiple <sot>...<eot> → multiple <PERCEPTION> <PERC_OUT> pairs.
"""
from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple

from .schema import ActivePerceptionSample, PerceptionStep

logger = logging.getLogger(__name__)

# VGR bbox token pattern: <sot>[x1,y1,x2,y2]<eot> or <sot>x1,y1,x2,y2<eot>
_BBOX_RE = re.compile(
    r'<sot>\[?'
    r'([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)'
    r'\]?<eot>'
)

# Observation text: text immediately after <eot> up to the next paragraph break
# or next <sot> or </think>. We capture it greedily but stop at natural boundaries.
# This pattern captures the "visual replay" paragraph that VGR inserts after each bbox.
_OBS_TEXT_RE = re.compile(
    r'<sot>\[?[\d.,\s]+\]?<eot>'   # the bbox token (already consumed)
    r'\s*'
    r'(.*?)'                         # observation text (lazy)
    r'(?=\n\s*\n|\n<sot>|</think>|$)',  # stop at blank line, next bbox, or end of think
    re.DOTALL
)


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
        keep_original_obs_text: bool = True,
        min_obs_text_len: int = 5,
        verbose: bool = False,
    ):
        self.image_root = Path(image_root) if image_root else None
        self.keep_original_obs_text = keep_original_obs_text
        self.min_obs_text_len = min_obs_text_len
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

        # Extract question (strip <image> tag)
        question = human_turn.get("value", "").replace("<image>", "").strip()
        raw_response = gpt_turn.get("value", "")

        if not raw_response:
            return None

        # Parse the response: convert bboxes → <PERCEPTION> <PERC_OUT>
        converted_response, steps = self._parse_and_convert_response(raw_response)

        # Extract answer (text after </think>)
        answer = self._extract_answer(raw_response)

        # Resolve image path
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
        Find all bbox tokens in the response, extract observation text,
        replace with <PERCEPTION> <PERC_OUT>, and return converted response + steps.
        """
        steps: List[PerceptionStep] = []
        result = response

        # We process matches from left to right, building the converted string.
        # We need to track offset shifts as we replace substrings.

        # First pass: find all bboxes and their positions + observation texts
        events = self._find_bbox_events(response)

        if not events:
            return response, []

        # Second pass: rebuild the string with replacements
        converted_parts = []
        cursor = 0

        for i, (match_start, match_end, bbox, obs_text) in enumerate(events):
            # Text before this bbox
            converted_parts.append(response[cursor:match_start])

            # Replacement: <PERCEPTION> <PERC_OUT>
            replacement = f"{self.PERCEPTION_TOKEN} {self.PERC_OUT_TOKEN}"
            if obs_text and self.keep_original_obs_text:
                # Preserve observation text AFTER <PERC_OUT>
                replacement = replacement + "\n" + obs_text

            converted_parts.append(replacement)
            cursor = match_end + (len(obs_text) if obs_text else 0)

            step = PerceptionStep(
                index=i,
                bbox=bbox,
                bbox_normalized=False,  # VGR uses pixel coords by default
                observation_text=obs_text if obs_text else None,
                target_type="observation_text" if obs_text else "none",
            )
            steps.append(step)

        # Remaining text
        converted_parts.append(response[cursor:])
        converted = "".join(converted_parts)

        return converted, steps

    def _find_bbox_events(
        self, text: str
    ) -> List[Tuple[int, int, List[float], Optional[str]]]:
        """
        Find all <sot>bbox<eot> occurrences and extract trailing observation text.

        Returns list of (match_start, match_end, bbox_list, observation_text).
        match_end points to end of <eot> token (NOT including obs_text).
        """
        events = []
        for m in _BBOX_RE.finditer(text):
            bbox = [float(m.group(i)) for i in range(1, 5)]
            bbox_end = m.end()

            # Extract observation text: text after <eot> up to paragraph break / next bbox / </think>
            obs_text = self._extract_obs_text(text, bbox_end)

            events.append((m.start(), bbox_end, bbox, obs_text))

        return events

    def _extract_obs_text(self, text: str, start: int) -> Optional[str]:
        """Extract observation/replay text starting at `start` in `text`."""
        tail = text[start:]

        # Skip leading whitespace / newlines
        stripped_start = len(tail) - len(tail.lstrip())
        tail = tail.lstrip()

        if not tail:
            return None

        # Find where observation text ends:
        # - blank line (paragraph break)
        # - next <sot>
        # - </think>
        # We take text up to the first of these
        end_patterns = [
            re.search(r'\n\s*\n', tail),        # blank line
            re.search(r'<sot>', tail),           # next bbox
            re.search(r'</think>', tail),        # end of think block
        ]
        end_positions = [m.start() for m in end_patterns if m is not None]

        if end_positions:
            obs_end = min(end_positions)
        else:
            obs_end = len(tail)

        obs_text = tail[:obs_end].strip()

        if len(obs_text) < self.min_obs_text_len:
            return None

        return obs_text

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer text (after </think>)."""
        think_end = response.rfind("</think>")
        if think_end == -1:
            # No think block; treat entire response as answer
            return response.strip()
        return response[think_end + len("</think>"):].strip()

    def _resolve_image(self, image_val: Any) -> str:
        """Resolve image path. image_val may be a string path or bytes."""
        if isinstance(image_val, bytes):
            # Raw bytes - caller must handle storage; return placeholder
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
