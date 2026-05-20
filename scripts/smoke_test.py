#!/usr/bin/env python3
"""
End-to-end smoke test. Loads the real Qwen2.5-VL model and runs:
  1. Model load + special token registration
  2. Dataset load (first 4 samples from converted JSONL)
  3. training_forward on 2 batches → loss is finite, shapes correct
  4. generate_with_perception on 1 sample → tokens produced

Usage:
    /cortex/users/rabanof/conda_envs/qwen49/bin/python scripts/smoke_test.py \
        --config configs/base.yaml
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator
from torch.utils.data import DataLoader


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check(cond: bool, msg: str):
    status = "PASS" if cond else "FAIL"
    logger.info(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_batches", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Model load ─────────────────────────────────────────────────────────
    logger.info("=== [1] Loading model ===")
    t0 = time.time()
    model_cfg = ActivePerceptionConfig(**{
        k: v for k, v in cfg["model"].items()
        if k in ActivePerceptionConfig.__dataclass_fields__
    })
    model = ActivePerceptionModel(model_cfg).to(device)
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    special_ids = model.get_special_token_ids()
    check(all(v > 0 for v in special_ids.values()), f"Special token IDs valid: {special_ids}")
    check(model.special_token_embeddings.shape[0] == 4, "special_token_embeddings shape [4, d]")

    # ── 2. Dataset load ───────────────────────────────────────────────────────
    logger.info("=== [2] Loading dataset (first 4 samples) ===")
    data_cfg = cfg["data"]
    dataset = ActivePerceptionDataset(
        data_path=data_cfg["data_path"],
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        supervision_mode=data_cfg.get("supervision_mode", "full"),
    )
    check(len(dataset) > 0, f"Dataset non-empty: {len(dataset)} samples")

    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    # ── 3. training_forward on N batches ─────────────────────────────────────
    logger.info(f"=== [3] training_forward x{args.n_batches} batches ===")
    model.train()
    losses = []
    for i, batch in enumerate(dl):
        if i >= args.n_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        t0 = time.time()
        out = model.training_forward(**batch)
        elapsed = time.time() - t0

        loss = out["loss_ce"]
        check(torch.isfinite(loss), f"Batch {i}: loss={loss.item():.4f} is finite")
        check(loss.item() > 0, f"Batch {i}: loss > 0")
        logger.info(f"  Batch {i}: loss={loss.item():.4f}  time={elapsed:.1f}s")
        losses.append(loss.item())

        z_list = out.get("z_perceptions", [])
        if z_list and z_list[0] is not None:
            z = z_list[0]
            check(z.dim() >= 2, f"Batch {i}: z.dim()={z.dim()} >= 2")
            check(z.shape[-1] == model.base_model.config.hidden_size,
                  f"Batch {i}: z last dim = d_llm")

    # ── 4. generate_with_perception on 1 sample ───────────────────────────────
    logger.info("=== [4] generate_with_perception (1 sample, max_new_tokens=32) ===")
    model.eval()
    sample = dataset[0]
    batch = collator([sample])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        t0 = time.time()
        generated_ids = model.generate_with_perception(
            **batch,
            max_new_tokens=32,
        )
        elapsed = time.time() - t0

    check(generated_ids is not None, "generate_with_perception returned non-None")
    check(generated_ids.shape[-1] > 0, f"Generated {generated_ids.shape[-1]} tokens")
    decoded = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"  Generated in {elapsed:.1f}s: {decoded[:120]!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=== ALL CHECKS PASSED ===")
    logger.info(f"  Losses: {[f'{l:.4f}' for l in losses]}")


if __name__ == "__main__":
    main()
