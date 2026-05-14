"""
Unified intermediate format for active perception datasets.

Supports:
- Multiple perception steps per sample
- Optional bbox metadata (supervision only, never shown to LLM)
- Optional observation/replay text (key gradient signal)
- Future dataset sources (DeepEyesV2, etc.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PerceptionStep:
    """One active perception event within a reasoning trace."""
    index: int                                        # zero-based step index
    bbox: Optional[List[float]] = None               # [x1, y1, x2, y2] in pixel or normalized coords
    bbox_normalized: bool = False                    # True if bbox coords are in [0,1]
    observation_text: Optional[str] = None           # visual replay / observation text from source dataset
    # What kind of alignment target is available for this step
    # "observation_text" | "crop_visual" | "none"
    target_type: str = "observation_text"

    def has_bbox(self) -> bool:
        return self.bbox is not None and len(self.bbox) == 4

    def has_observation(self) -> bool:
        return self.observation_text is not None and len(self.observation_text.strip()) > 0


@dataclass
class ActivePerceptionSample:
    """
    One training sample in the unified active perception format.

    The converted_response field contains the full model response using
    <PERCEPTION> and <PERC_OUT> tokens in place of original bbox/action tokens.

    The original bbox coordinates are stored in perception_steps for
    supervision use only — they are NEVER part of the model's text output.
    """
    id: str
    image: str                                        # absolute or relative path to image file
    source: str                                       # "vgr" | "deepeyes" | "synthetic" | "none"
    question: str                                     # the human question
    converted_response: str                           # full model response with <PERCEPTION>/<PERC_OUT>
    converted_answer: str                             # final answer string (extracted)
    perception_steps: List[PerceptionStep] = field(default_factory=list)
    has_perception: bool = False
    # Arbitrary metadata for debugging / filtering
    metadata: Dict[str, Any] = field(default_factory=dict)

    def num_perception_steps(self) -> int:
        return len(self.perception_steps)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActivePerceptionSample":
        steps = [PerceptionStep(**s) for s in d.pop("perception_steps", [])]
        return cls(**d, perception_steps=steps)
