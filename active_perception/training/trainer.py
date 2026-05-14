"""
Active Perception Trainer.

Training stages:
  Stage 0 (warmup, optional):
    - Freeze LLM, ViT, projector
    - Train only: PerceptionModule + optional alignment projectors
    - Loss: L_ground (anti-collapse warmup) + optionally L_sem
    - Very few steps (e.g., 100–500)
    - Purpose: prevent uniform-attention collapse at init

  Stage 1 (CE-only):
    - Freeze LLM, ViT, projector
    - Train: PerceptionModule + special token embeddings
    - Loss: L_SFT only (CE on obs text + reasoning + answer)
    - Core hypothesis test: can CE alone teach latent visual retrieval?

  Stage 2 (CE + optional alignment):
    - Same freezing as Stage 1
    - Loss: L_SFT + optional L_sem
    - Tests whether semantic alignment improves convergence

  Stage 3 (LoRA, optional):
    - ViT + projector frozen
    - LLM unfrozen via LoRA
    - Full joint training

Each stage is driven by a TrainerConfig; stages share the same trainer loop.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import LossConfig, PerceptionLosses, LossOutput

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "runs/exp1_ce_only"
    experiment_name: str = "exp1_ce_only"

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs: int = 3
    max_steps: int = -1                 # -1 = unlimited (use num_epochs)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0

    # ── Optimizer ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    lr_scheduler_type: str = "cosine"  # "cosine" | "linear" | "constant"

    # ── Warmup stage (optional) ───────────────────────────────────────────────
    do_warmup_stage: bool = False
    warmup_stage_steps: int = 200
    warmup_lr: float = 5e-4
    # After warmup, lambda_ground → 0.0

    # ── Logging / checkpointing ───────────────────────────────────────────────
    log_every_n_steps: int = 10
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 500
    save_total_limit: int = 3

    # ── Precision ────────────────────────────────────────────────────────────
    bf16: bool = True
    gradient_checkpointing: bool = True

    # ── Debugging ────────────────────────────────────────────────────────────
    debug_first_n_batches: int = 2      # run surgery debug on first N batches
    log_attn_weights_every: int = 200   # log attention weight entropy every N steps


class ActivePerceptionTrainer:
    """
    Training loop for active perception experiments.

    Designed for clean ablations: the same trainer handles all stages
    via different TrainerConfig + LossConfig combinations.
    """

    def __init__(
        self,
        model,                              # ActivePerceptionModel
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        trainer_config: TrainerConfig,
        loss_config: LossConfig,
        device: torch.device,
    ):
        self.model = model
        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader
        self.tcfg = trainer_config
        self.lcfg = loss_config
        self.device = device

        # Loss module (auxiliary losses only; CE comes from model forward)
        d_model = model.base_model.config.hidden_size
        self.loss_fn = PerceptionLosses(d_model, loss_config).to(device)

        # Output directory
        os.makedirs(trainer_config.output_dir, exist_ok=True)
        self._save_configs()

        # Setup logging
        self._setup_logging()

        self.global_step = 0
        self.best_eval_loss = float("inf")
        self._checkpoint_paths: List[str] = []

    # =========================================================================
    # Main training entry points
    # =========================================================================

    def train(self):
        """Run the full training procedure (warmup stage if enabled, then main)."""
        if self.tcfg.do_warmup_stage:
            logger.info("[Trainer] Starting warmup stage")
            self._run_warmup_stage()
            logger.info("[Trainer] Warmup stage complete. Switching to main training.")

        logger.info("[Trainer] Starting main training stage")
        self._setup_optimizer_and_scheduler(
            lr=self.tcfg.learning_rate,
            max_steps=self._compute_max_steps(),
        )
        self._run_training_loop(max_steps=self._compute_max_steps())
        logger.info("[Trainer] Training complete.")

    # =========================================================================
    # Warmup stage
    # =========================================================================

    def _run_warmup_stage(self):
        """
        Short warmup to prevent uniform-attention collapse.
        Only trains PerceptionModule + optional alignment projectors.
        LLM, ViT, projector all frozen.
        """
        warmup_params = list(self.model.perception_module.parameters())
        warmup_params += list(self.loss_fn.parameters())
        warmup_params = [p for p in warmup_params if p.requires_grad]

        optimizer = torch.optim.AdamW(warmup_params, lr=self.tcfg.warmup_lr)
        scheduler = self._build_scheduler(optimizer, self.tcfg.warmup_stage_steps)

        # Enable grounding loss during warmup
        original_lambda_ground = self.lcfg.lambda_ground
        original_use_grounding = self.lcfg.use_grounding
        self.lcfg.use_grounding = True

        warmup_step = 0
        for batch in self._cycle_dataloader(self.train_dl):
            if warmup_step >= self.tcfg.warmup_stage_steps:
                break
            batch = self._to_device(batch)
            debug = warmup_step < self.tcfg.debug_first_n_batches

            loss_out = self._forward_step(batch, debug=debug)
            loss_out.total.backward()

            nn.utils.clip_grad_norm_(warmup_params, self.tcfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            warmup_step += 1
            if warmup_step % self.tcfg.log_every_n_steps == 0:
                log_d = loss_out.as_log_dict()
                log_d["warmup_step"] = warmup_step
                log_d["lr"] = scheduler.get_last_lr()[0]
                logger.info(f"[Warmup] {log_d}")
                self._write_log(log_d, prefix="warmup")

        # Disable grounding after warmup
        self.lcfg.use_grounding = False
        self.lcfg.lambda_ground = 0.0
        logger.info("[Warmup] Grounding loss disabled after warmup stage.")

    # =========================================================================
    # Main training loop
    # =========================================================================

    def _run_training_loop(self, max_steps: int):
        self.model.train()
        self.loss_fn.train()

        optimizer = self.optimizer
        scheduler = self.scheduler

        accum_loss = 0.0
        accum_steps = 0
        t0 = time.time()

        for batch in self._cycle_dataloader(self.train_dl):
            if 0 < max_steps <= self.global_step:
                break

            batch = self._to_device(batch)
            debug = self.global_step < self.tcfg.debug_first_n_batches

            # Forward
            loss_out = self._forward_step(batch, debug=debug)
            loss = loss_out.total / self.tcfg.gradient_accumulation_steps
            loss.backward()

            accum_loss += loss_out.total.item()
            accum_steps += 1

            if accum_steps % self.tcfg.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad] +
                    list(self.loss_fn.parameters()),
                    self.tcfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.tcfg.log_every_n_steps == 0:
                    step_time = time.time() - t0
                    log_d = loss_out.as_log_dict()
                    log_d.update({
                        "step": self.global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "steps_per_sec": self.tcfg.log_every_n_steps / step_time,
                    })
                    # Log attention weight diagnostics
                    if self.global_step % self.tcfg.log_attn_weights_every == 0:
                        log_d.update(self._compute_attn_diagnostics(batch))

                    logger.info(
                        f"[Step {self.global_step}] " +
                        " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in log_d.items())
                    )
                    self._write_log(log_d)
                    t0 = time.time()
                    accum_loss = 0.0

                # Evaluation
                if (self.eval_dl is not None and
                        self.global_step % self.tcfg.eval_every_n_steps == 0):
                    eval_loss = self._evaluate()
                    logger.info(f"[Eval] step={self.global_step} eval_loss={eval_loss:.4f}")
                    self._write_log({"step": self.global_step, "eval/loss": eval_loss})
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self._save_checkpoint("best")
                    self.model.train()

                # Checkpointing
                if self.global_step % self.tcfg.save_every_n_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

    # =========================================================================
    # Forward step
    # =========================================================================

    def _forward_step(self, batch: Dict, debug: bool = False) -> LossOutput:
        """Single training forward pass through model + loss computation."""
        with torch.cuda.amp.autocast(enabled=self.tcfg.bf16, dtype=torch.bfloat16):
            out = self.model.training_forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                perc_positions=batch["perc_positions"],
                perc_out_positions=batch["perc_out_positions"],
                debug=debug,
            )

        # Compute auxiliary losses
        # For semantic alignment: would need obs text embeddings (pre-computed)
        # For now: pass None → auxiliary losses gracefully skip
        loss_out = self.loss_fn.compute(
            loss_ce=out["loss_ce"],
            z_perceptions=out["z_perceptions"],
            attn_weights_list=out["attn_weights_list"],
            obs_text_embeddings=batch.get("obs_text_embeddings"),
            crop_embeddings=batch.get("crop_embeddings"),
            patch_masks=batch.get("patch_masks"),
        )

        return loss_out

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.eval_dl:
            batch = self._to_device(batch)
            with torch.cuda.amp.autocast(enabled=self.tcfg.bf16, dtype=torch.bfloat16):
                out = self.model.training_forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    perc_positions=batch["perc_positions"],
                    perc_out_positions=batch["perc_out_positions"],
                )
            total_loss += out["loss_ce"].item()
            n_batches += 1
            if n_batches >= 50:  # limit eval batches for speed
                break
        return total_loss / max(n_batches, 1)

    # =========================================================================
    # Attention diagnostics
    # =========================================================================

    def _compute_attn_diagnostics(self, batch: Dict) -> Dict[str, float]:
        """Log attention weight entropy to check for collapse."""
        try:
            with torch.no_grad():
                out = self.model.training_forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    perc_positions=batch["perc_positions"],
                    perc_out_positions=batch["perc_out_positions"],
                )
            entropies = []
            for attn_w in out["attn_weights_list"]:
                if attn_w is None:
                    continue
                # attn_w: [K, N], compute entropy per query
                ent = -(attn_w * (attn_w + 1e-9).log()).sum(dim=-1).mean()
                entropies.append(ent.item())
            if entropies:
                import statistics
                return {
                    "diag/attn_entropy_mean": statistics.mean(entropies),
                    "diag/attn_entropy_max": max(entropies),
                }
        except Exception as e:
            logger.warning(f"[Trainer] Attention diagnostics failed: {e}")
        return {}

    # =========================================================================
    # Optimizer / scheduler / misc helpers
    # =========================================================================

    def _setup_optimizer_and_scheduler(self, lr: float, max_steps: int):
        params = (
            [p for p in self.model.parameters() if p.requires_grad] +
            list(self.loss_fn.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=self.tcfg.weight_decay
        )
        self.scheduler = self._build_scheduler(self.optimizer, max_steps)

    def _build_scheduler(self, optimizer, max_steps: int):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR

        warmup = self.tcfg.warmup_steps
        if warmup > 0 and max_steps > warmup:
            warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            if self.tcfg.lr_scheduler_type == "cosine":
                main_sched = CosineAnnealingLR(optimizer, T_max=max_steps - warmup)
            else:
                main_sched = ConstantLR(optimizer, factor=1.0, total_iters=max_steps)
            return SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup])
        elif self.tcfg.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(max_steps, 1))
        else:
            return ConstantLR(optimizer, factor=1.0, total_iters=max_steps)

    def _compute_max_steps(self) -> int:
        if self.tcfg.max_steps > 0:
            return self.tcfg.max_steps
        steps_per_epoch = len(self.train_dl) // self.tcfg.gradient_accumulation_steps
        return steps_per_epoch * self.tcfg.num_epochs

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _cycle_dataloader(self, dl: DataLoader):
        """Yield batches indefinitely."""
        while True:
            for batch in dl:
                yield batch

    def _save_checkpoint(self, tag: str):
        ckpt_dir = os.path.join(self.tcfg.output_dir, f"checkpoint-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save perception module (always)
        self.model.save_perception_module(ckpt_dir)
        # Save loss projectors if any
        if any(p.requires_grad for p in self.loss_fn.parameters()):
            torch.save(self.loss_fn.state_dict(), f"{ckpt_dir}/loss_fn.pt")
        # Save LoRA adapter if active
        if self.model.config.use_lora:
            self.model.base_model.save_pretrained(ckpt_dir)
        # Save config
        with open(f"{ckpt_dir}/trainer_config.json", "w") as f:
            import dataclasses
            json.dump(dataclasses.asdict(self.tcfg), f, indent=2)

        self._checkpoint_paths.append(ckpt_dir)
        # Rotate checkpoints
        if (self.tcfg.save_total_limit > 0 and
                len(self._checkpoint_paths) > self.tcfg.save_total_limit + 1):
            old = self._checkpoint_paths.pop(0)
            if "best" not in old:
                import shutil
                shutil.rmtree(old, ignore_errors=True)

        logger.info(f"[Trainer] Checkpoint saved: {ckpt_dir}")

    def _save_configs(self):
        import dataclasses
        with open(os.path.join(self.tcfg.output_dir, "trainer_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.tcfg), f, indent=2)
        with open(os.path.join(self.tcfg.output_dir, "loss_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.lcfg), f, indent=2)

    def _setup_logging(self):
        log_path = os.path.join(self.tcfg.output_dir, "train.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)
        self._log_path = log_path
        self._log_rows: List[Dict] = []

    def _write_log(self, d: Dict, prefix: str = "train"):
        self._log_rows.append(d)
        # Write JSONL log
        jsonl_path = os.path.join(self.tcfg.output_dir, f"{prefix}_metrics.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(d) + "\n")
