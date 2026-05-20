"""Unit tests for VGR converter (actual VGR format: <SOT>[...]<EOT><image>)."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.data.vgr_converter import VGRConverter
from active_perception.data.schema import ActivePerceptionSample


# Actual VGR format: UPPERCASE <SOT>/<EOT>, normalized [0,1] coords, <image> suffix
FAKE_SAMPLE = {
    "image": "images/test.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is in the top-left corner?"},
        {"from": "gpt", "value": (
            "<think>\nLet me look at the image carefully.\n"
            "<SOT>[0.0, 0.02, 0.5, 0.36]<EOT><image>\n"
            "Based on this, the answer is red.\n"
            "</think>\nred"
        )},
    ]
}

FAKE_MULTI_STEP = {
    "image": "images/test2.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nHow many objects are visible?"},
        {"from": "gpt", "value": (
            "<think>\nFirst I check the left side.\n"
            "<SOT>[0.0, 0.0, 0.5, 1.0]<EOT><image>\n"
            "Now the right side.\n"
            "<SOT>[0.5, 0.0, 1.0, 1.0]<EOT><image>\n"
            "Total: 3 objects.\n"
            "</think>\n3"
        )},
    ]
}

FAKE_NO_BBOX = {
    "image": "images/test3.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat color is the sky?"},
        {"from": "gpt", "value": "<think>\nThe sky looks blue.\n</think>\nblue"},
    ]
}

FAKE_BBOX_OUTSIDE_THINK = {
    "image": "images/test4.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe the rider."},
        {"from": "gpt", "value": (
            "<think>\nI need to identify the rider.\n</think>\n"
            "- <SOT>[0.49, 0.57, 0.67, 1.0]<EOT><image>\n\n"
            "The rider is wearing a helmet."
        )},
    ]
}

FAKE_BBOX_INSIDE_AND_OUTSIDE = {
    "image": "images/test5.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat are the two regions?"},
        {"from": "gpt", "value": (
            "<think>\nRegion A is <SOT>[0.0, 0.2, 1.0, 0.57]<EOT><image> here.\n</think>\n"
            "Also region B: <SOT>[0.0, 0.31, 1.0, 0.62]<EOT><image>"
        )},
    ]
}


class TestVGRConverter:
    def setup_method(self):
        self.converter = VGRConverter(verbose=True)

    def test_single_step_conversion(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_0")
        assert sample is not None
        assert isinstance(sample, ActivePerceptionSample)
        assert sample.has_perception
        assert sample.num_perception_steps() == 1
        step = sample.perception_steps[0]
        assert step.bbox == [0.0, 0.02, 0.5, 0.36]
        assert step.bbox_normalized is True
        assert step.observation_text is None
        assert step.target_type == "none"
        assert "<PERCEPTION>" in sample.converted_response
        assert "<PERC_OUT>" in sample.converted_response

    def test_original_bbox_tokens_removed(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_tokens")
        assert sample is not None
        assert "<SOT>" not in sample.converted_response
        assert "<EOT>" not in sample.converted_response
        assert "<sot>" not in sample.converted_response
        assert "<eot>" not in sample.converted_response

    def test_bbox_coords_not_in_converted_response(self):
        """BBox coordinates must NEVER appear as text in the converted response."""
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_nobbox_in_text")
        assert sample is not None
        assert "0.0, 0.02, 0.5, 0.36" not in sample.converted_response
        assert "[0.0, 0.02, 0.5, 0.36]" not in sample.converted_response

    def test_multi_step_conversion(self):
        sample = self.converter.convert_sample(FAKE_MULTI_STEP, "test_multi")
        assert sample is not None
        assert sample.num_perception_steps() == 2
        assert sample.perception_steps[0].index == 0
        assert sample.perception_steps[1].index == 1
        assert sample.perception_steps[0].bbox == [0.0, 0.0, 0.5, 1.0]
        assert sample.perception_steps[1].bbox == [0.5, 0.0, 1.0, 1.0]
        assert sample.converted_response.count("<PERCEPTION>") == 2
        assert sample.converted_response.count("<PERC_OUT>") == 2

    def test_no_bbox_sample(self):
        sample = self.converter.convert_sample(FAKE_NO_BBOX, "test_nobbox")
        assert sample is not None
        assert not sample.has_perception
        assert sample.num_perception_steps() == 0
        assert "<PERCEPTION>" not in sample.converted_response

    def test_bbox_outside_think(self):
        sample = self.converter.convert_sample(FAKE_BBOX_OUTSIDE_THINK, "test_outside")
        assert sample is not None
        assert sample.has_perception
        assert sample.num_perception_steps() == 1
        assert sample.perception_steps[0].bbox == [0.49, 0.57, 0.67, 1.0]
        assert "<PERCEPTION>" in sample.converted_response

    def test_bbox_inside_and_outside_think(self):
        sample = self.converter.convert_sample(FAKE_BBOX_INSIDE_AND_OUTSIDE, "test_both")
        assert sample is not None
        assert sample.num_perception_steps() == 2
        assert sample.converted_response.count("<PERCEPTION>") == 2

    def test_normalized_coords_all_steps(self):
        sample = self.converter.convert_sample(FAKE_MULTI_STEP, "test_norm")
        assert sample is not None
        for step in sample.perception_steps:
            assert step.bbox_normalized is True
            assert all(0.0 <= c <= 1.0 for c in step.bbox), \
                f"Coord out of [0,1]: {step.bbox}"

    def test_answer_extraction(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_ans")
        assert sample is not None
        assert sample.converted_answer == "red"

    def test_invalid_sample(self):
        bad = {"conversations": []}
        result = self.converter.convert_sample(bad, "bad")
        assert result is None

    def test_wrong_role_order(self):
        bad = {
            "image": "img.jpg",
            "conversations": [
                {"from": "gpt", "value": "answer"},
                {"from": "human", "value": "<image>\nQ?"},
            ]
        }
        result = self.converter.convert_sample(bad, "bad_roles")
        assert result is None

    def test_image_token_stripped_from_question(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_img_strip")
        assert sample is not None
        assert "<image>" not in sample.question

    def test_stats(self):
        self.converter.reset_stats()
        self.converter.convert_sample(FAKE_SAMPLE, "s1")
        self.converter.convert_sample(FAKE_MULTI_STEP, "s2")
        self.converter.convert_sample(FAKE_NO_BBOX, "s3")
        stats = self.converter.get_stats()
        assert stats["total"] == 3
        assert stats["converted"] == 3
        assert stats["single_step"] == 1
        assert stats["multi_step"] == 1
        assert stats["no_bbox"] == 1

    def test_surrounding_text_preserved(self):
        """Text around the bbox token must survive the conversion intact."""
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_preserve")
        assert sample is not None
        assert "Let me look at the image carefully" in sample.converted_response
        assert "Based on this, the answer is red" in sample.converted_response

    def test_case_insensitive_pattern(self):
        """Converter must handle lowercase <sot>/<eot> variants too."""
        raw = {
            "image": "img.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nQ?"},
                {"from": "gpt", "value": "<think>\n<sot>[0.1, 0.2, 0.3, 0.4]<eot><image>\n</think>\nA"},
            ]
        }
        sample = self.converter.convert_sample(raw, "case_insensitive")
        assert sample is not None
        assert sample.num_perception_steps() == 1
        assert "<PERCEPTION>" in sample.converted_response
