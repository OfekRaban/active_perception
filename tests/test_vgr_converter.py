"""Unit tests for VGR converter."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.data.vgr_converter import VGRConverter
from active_perception.data.schema import ActivePerceptionSample


FAKE_SAMPLE = {
    "image": "images/test.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is in the top-left corner?"},
        {"from": "gpt", "value": (
            "<think>\nLet me look at the image carefully.\n"
            "<sot>[10,20,100,80]<eot>\n"
            "The top-left corner shows a red traffic light.\n"
            "Based on this observation, the answer is red.\n"
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
            "<sot>[0,0,200,400]<eot>\n"
            "On the left there are 2 cars.\n"
            "Now the right side.\n"
            "<sot>[200,0,400,400]<eot>\n"
            "On the right there is 1 bus.\n"
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
        assert step.bbox == [10.0, 20.0, 100.0, 80.0]
        assert "<PERCEPTION>" in sample.converted_response
        assert "<PERC_OUT>" in sample.converted_response
        assert "<sot>" not in sample.converted_response
        assert "<eot>" not in sample.converted_response

    def test_observation_text_preserved(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_obs")
        assert sample is not None
        step = sample.perception_steps[0]
        assert step.observation_text is not None
        assert "traffic light" in step.observation_text.lower() or len(step.observation_text) > 5
        # Observation text should appear in converted response after <PERC_OUT>
        perc_out_pos = sample.converted_response.find("<PERC_OUT>")
        assert perc_out_pos != -1
        text_after = sample.converted_response[perc_out_pos:]
        assert "red traffic light" in text_after or "top-left" in text_after or len(text_after) > 10

    def test_multi_step_conversion(self):
        sample = self.converter.convert_sample(FAKE_MULTI_STEP, "test_multi")
        assert sample is not None
        assert sample.num_perception_steps() == 2
        assert sample.perception_steps[0].index == 0
        assert sample.perception_steps[1].index == 1
        assert sample.converted_response.count("<PERCEPTION>") == 2
        assert sample.converted_response.count("<PERC_OUT>") == 2

    def test_no_bbox_sample(self):
        sample = self.converter.convert_sample(FAKE_NO_BBOX, "test_nobbox")
        assert sample is not None
        assert not sample.has_perception
        assert sample.num_perception_steps() == 0
        assert "<PERCEPTION>" not in sample.converted_response

    def test_answer_extraction(self):
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_ans")
        assert sample is not None
        assert sample.converted_answer == "red"

    def test_invalid_sample(self):
        bad = {"conversations": []}
        result = self.converter.convert_sample(bad, "bad")
        assert result is None

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

    def test_bbox_not_in_converted_response(self):
        """BBox coordinates must NEVER appear as text in the converted response."""
        sample = self.converter.convert_sample(FAKE_SAMPLE, "test_nobbox_in_text")
        assert sample is not None
        assert "[10,20,100,80]" not in sample.converted_response
        assert "10,20,100,80" not in sample.converted_response
