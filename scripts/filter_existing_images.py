#!/usr/bin/env python3
"""
Filter the converted VGR dataset to keep only samples whose images exist on disk.

Resolves paths using IMAGE_ROOTS (prefix → local directory mapping).
Outputs:
    data/vgr_existing_train.jsonl
    data/vgr_existing_eval.jsonl

Usage:
    /cortex/users/rabanof/conda_envs/qwen49/bin/python scripts/filter_existing_images.py \
        [--train data/vgr_converted_train.jsonl] \
        [--eval  data/vgr_converted_eval.jsonl]
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.data.utils import resolve_image_path, IMAGE_ROOTS


def filter_file(input_path: str, output_path: str) -> dict:
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = kept = missing = 0
    prefix_kept = Counter()
    prefix_missing = Counter()

    with open(inp, "r", encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            sample = json.loads(line)
            image_path = sample.get("image", "")
            prefix = Path(image_path).parts[0] if image_path and "/" in image_path else "unknown"

            resolved = resolve_image_path(image_path)
            if resolved is not None:
                fout.write(line + "\n")
                kept += 1
                prefix_kept[prefix] += 1
            else:
                missing += 1
                prefix_missing[prefix] += 1

            if total % 10000 == 0:
                logger.info(f"  Scanned {total:,}  kept {kept:,}  missing {missing:,}")

    return {
        "total": total,
        "kept": kept,
        "missing": missing,
        "prefix_kept": dict(prefix_kept.most_common()),
        "prefix_missing": dict(prefix_missing.most_common()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/vgr_converted_train.jsonl")
    parser.add_argument("--eval",  default="data/vgr_converted_eval.jsonl")
    parser.add_argument("--out_train", default="data/vgr_existing_train.jsonl")
    parser.add_argument("--out_eval",  default="data/vgr_existing_eval.jsonl")
    args = parser.parse_args()

    logger.info(f"Active image roots: {IMAGE_ROOTS}")

    for label, inp, out in [
        ("TRAIN", args.train, args.out_train),
        ("EVAL",  args.eval,  args.out_eval),
    ]:
        logger.info(f"\n=== {label}: {inp} → {out} ===")
        stats = filter_file(inp, out)

        kept_pct = 100 * stats["kept"] / max(stats["total"], 1)
        logger.info(f"  Total   : {stats['total']:,}")
        logger.info(f"  Kept    : {stats['kept']:,}  ({kept_pct:.1f}%)")
        logger.info(f"  Missing : {stats['missing']:,}")
        logger.info(f"  Kept by prefix:")
        for prefix, count in sorted(stats["prefix_kept"].items(), key=lambda x: -x[1]):
            logger.info(f"    {prefix:<20} {count:,}")
        if stats["prefix_missing"]:
            logger.info(f"  Missing by prefix (top 5):")
            for prefix, count in list(stats["prefix_missing"].items())[:5]:
                logger.info(f"    {prefix:<20} {count:,}")

    logger.info("\nDone. Update configs/base.yaml data_path to point to vgr_existing_*.jsonl")


if __name__ == "__main__":
    main()
