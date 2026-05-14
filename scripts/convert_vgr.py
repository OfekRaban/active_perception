#!/usr/bin/env python3
"""
Convert a VGR dataset file to the unified ActivePerception JSONL format.

Usage:
    python scripts/convert_vgr.py \
        --input data/raw/vgr_train.parquet \
        --output data/vgr_converted_train.jsonl \
        --image_root /path/to/images \
        --split train \
        --max_samples 5000  # optional, for quick experiments
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.data.vgr_converter import VGRConverter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to VGR dataset (parquet or jsonl)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--image_root", default=None, help="Root directory for image paths")
    p.add_argument("--split", default="train", choices=["train", "eval", "test"])
    p.add_argument("--max_samples", type=int, default=-1, help="Max samples (-1 = all)")
    p.add_argument("--eval_fraction", type=float, default=0.02,
                   help="Fraction of data to hold out as eval (only if split=train)")
    p.add_argument("--min_obs_text_len", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converter = VGRConverter(
        image_root=args.image_root,
        min_obs_text_len=args.min_obs_text_len,
        verbose=args.verbose,
    )

    logger.info(f"Converting {args.input} → {args.output}")
    samples = []
    for sample in converter.convert_dataset_file(args.input):
        samples.append(sample)
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break
        if len(samples) % 1000 == 0:
            logger.info(f"  Converted {len(samples)} samples so far...")

    logger.info(f"Conversion stats: {converter.get_stats()}")

    # Split train/eval if requested
    if args.split == "train" and args.eval_fraction > 0:
        n_eval = max(1, int(len(samples) * args.eval_fraction))
        eval_samples = samples[-n_eval:]
        train_samples = samples[:-n_eval]

        eval_path = out_path.parent / out_path.name.replace("train", "eval")
        _write_jsonl(train_samples, out_path)
        _write_jsonl(eval_samples, eval_path)
        logger.info(f"Wrote {len(train_samples)} train samples to {out_path}")
        logger.info(f"Wrote {len(eval_samples)} eval samples to {eval_path}")
    else:
        _write_jsonl(samples, out_path)
        logger.info(f"Wrote {len(samples)} samples to {out_path}")

    # Print distribution stats
    has_perception = sum(1 for s in samples if s.has_perception)
    multi_step = sum(1 for s in samples if s.num_perception_steps() > 1)
    has_obs_text = sum(
        1 for s in samples if any(p.has_observation() for p in s.perception_steps)
    )
    logger.info(f"Samples with perception:   {has_perception}/{len(samples)} ({100*has_perception/max(len(samples),1):.1f}%)")
    logger.info(f"Samples with >1 step:      {multi_step}/{len(samples)}")
    logger.info(f"Samples with obs text:     {has_obs_text}/{len(samples)}")


def _write_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
