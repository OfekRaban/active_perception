#!/usr/bin/env python3
"""
Run causal diagnostics on a trained active perception checkpoint.

Usage:
    python scripts/eval_diagnostics.py \
        --config configs/exp1_ce_only.yaml \
        --checkpoint runs/exp1_ce_only/checkpoint-best \
        --data data/vgr_converted_eval.jsonl \
        --max_samples 100
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator
from active_perception.eval.causal_diagnostics import CausalDiagnostics, DiagnosticMode
from torch.utils.data import DataLoader
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--data", required=True, help="Eval JSONL path")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="JSON output path for results")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ActivePerceptionConfig(**cfg.get("model", {}))
    model = ActivePerceptionModel(model_cfg)
    model.load_perception_module(args.checkpoint)
    model = model.to(device)
    model.eval()

    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )
    eval_dataset = ActivePerceptionDataset(
        data_path=args.data,
        processor=model.processor,
        special_token_ids=model.get_special_token_ids(),
        image_root=cfg.get("data", {}).get("image_root"),
    )
    eval_dl = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    diagnostics = CausalDiagnostics(model, device)
    results = diagnostics.run_all(eval_dl, max_samples=args.max_samples)

    out = {k: {"ce_loss": v.ce_loss, "n_samples": v.n_samples} for k, v in results.items()}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
