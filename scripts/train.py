#!/usr/bin/env python3
"""
Main training script for active perception experiments.

Usage:
    # Experiment 1: CE-only
    python scripts/train.py --config configs/exp1_ce_only.yaml

    # Experiment 2: warmup + CE
    python scripts/train.py --config configs/exp2_warmup_ce.yaml

    # Override individual values
    python scripts/train.py --config configs/exp1_ce_only.yaml \
        --model.model_path /data/models/Qwen2.5-VL-7B-Instruct \
        --training.max_steps 1000
"""
import argparse
import logging
import sys
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
from active_perception.training.losses import LossConfig
from active_perception.training.trainer import ActivePerceptionTrainer, TrainerConfig

from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Handle defaults inheritance (simple flat merge)
    if "defaults" in cfg:
        base_names = cfg.pop("defaults")
        for base_name in base_names:
            base_path = Path(config_path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as fb:
                    base_cfg = yaml.safe_load(fb)
                base_cfg.update(cfg)
                cfg = base_cfg
    return cfg


def apply_cli_overrides(cfg: dict, overrides: list) -> dict:
    """Apply --key.subkey value overrides from CLI."""
    for override in overrides:
        if "=" not in override:
            continue
        key_path, value = override.split("=", 1)
        keys = key_path.lstrip("-").split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse value as int/float/bool
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        d[keys[-1]] = value
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args()

    # Load YAML config
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, unknown)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    model_cfg = ActivePerceptionConfig(**cfg.get("model", {}))
    logger.info(f"Model config: {model_cfg}")
    model = ActivePerceptionModel(model_cfg)
    model = model.to(device)

    if cfg.get("training", {}).get("gradient_checkpointing", True):
        model.base_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ── Build datasets ────────────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    special_ids = model.get_special_token_ids()

    train_dataset = ActivePerceptionDataset(
        data_path=data_cfg["data_path"],
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
        system_prompt=data_cfg.get("system_prompt"),
        supervision_mode=data_cfg.get("supervision_mode", "full"),
    )

    eval_dataset = None
    eval_dl = None
    if data_cfg.get("eval_data_path"):
        eval_dataset = ActivePerceptionDataset(
            data_path=data_cfg["eval_data_path"],
            processor=model.processor,
            special_token_ids=special_ids,
            image_root=data_cfg.get("image_root"),
            max_seq_len=data_cfg.get("max_seq_len", 2048),
        )

    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )

    train_cfg = cfg.get("training", {})
    train_dl = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("per_device_train_batch_size", 1),
        shuffle=True,
        collate_fn=collator,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )
    if eval_dataset:
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer_config = TrainerConfig(**{
        k: v for k, v in train_cfg.items()
        if k in TrainerConfig.__dataclass_fields__
    })
    loss_config = LossConfig(**{
        k: v for k, v in cfg.get("losses", {}).items()
        if k in LossConfig.__dataclass_fields__
    })

    trainer = ActivePerceptionTrainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        trainer_config=trainer_config,
        loss_config=loss_config,
        device=device,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
