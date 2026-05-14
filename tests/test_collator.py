"""Unit tests for the dataset collator (no model loading required)."""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.data.dataset import ActivePerceptionCollator


def make_fake_item(seq_len, n_perc=1, n_perc_out=1, pad_token_id=0):
    """Create a fake dataset item for collator testing."""
    input_ids = torch.randint(1, 100, (seq_len,))
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    labels = torch.randint(1, 100, (seq_len,))

    PERC_ID = 200
    PERC_OUT_ID = 201
    perc_positions = []
    perc_out_positions = []

    # Place PERCEPTION and PERC_OUT tokens at known positions
    for i in range(n_perc):
        pos_p = 5 + i * 10
        pos_o = pos_p + 1
        if pos_o < seq_len:
            input_ids[pos_p] = PERC_ID
            input_ids[pos_o] = PERC_OUT_ID
            labels[pos_o] = -100
            perc_positions.append(pos_p)
            perc_out_positions.append(pos_o)

    return {
        "sample_id": f"fake_{seq_len}",
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.randn(1, 3, 224, 224),
        "image_grid_thw": torch.tensor([[1, 14, 14]]),
        "perc_positions": perc_positions,
        "perc_out_positions": perc_out_positions,
        "bboxes": [[10, 20, 100, 80]] * n_perc,
        "observation_texts": ["fake observation"] * n_perc,
        "has_perception": n_perc > 0,
        "num_perception_steps": n_perc,
        "source": "vgr",
    }


class TestActivePerceptionCollator:
    def setup_method(self):
        self.collator = ActivePerceptionCollator(pad_token_id=0)

    def test_single_item_batch(self):
        item = make_fake_item(20)
        batch = self.collator([item])
        assert batch["input_ids"].shape == (1, 20)
        assert batch["attention_mask"].shape == (1, 20)
        assert batch["labels"].shape == (1, 20)

    def test_variable_length_padding(self):
        items = [make_fake_item(15), make_fake_item(25), make_fake_item(20)]
        batch = self.collator(items)
        assert batch["input_ids"].shape == (3, 25)
        assert batch["attention_mask"].shape == (3, 25)
        # Shorter sequences should have zeros in attention_mask at padded positions
        assert batch["attention_mask"][0, 15:].sum() == 0
        assert batch["attention_mask"][2, 20:].sum() == 0

    def test_labels_padded_with_minus_100(self):
        items = [make_fake_item(10), make_fake_item(20)]
        batch = self.collator(items)
        # Labels beyond seq len should be -100
        assert (batch["labels"][0, 10:] == -100).all()

    def test_perc_positions_preserved(self):
        items = [make_fake_item(30, n_perc=2), make_fake_item(30, n_perc=1)]
        batch = self.collator(items)
        assert len(batch["perc_positions"][0]) == 2
        assert len(batch["perc_positions"][1]) == 1

    def test_perc_out_labels_are_minus_100(self):
        item = make_fake_item(30, n_perc=1)
        batch = self.collator([item])
        for pos in batch["perc_out_positions"][0]:
            assert batch["labels"][0, pos].item() == -100

    def test_pixel_values_stacked(self):
        items = [make_fake_item(20), make_fake_item(20)]
        batch = self.collator(items)
        pv = batch["pixel_values"]
        assert pv is not None
