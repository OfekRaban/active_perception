"""
Causal diagnostics for active perception.

Goal: determine whether z_perception is actually causally used by the model,
or whether the LLM has learned to ignore the injected evidence token.

Diagnostic modes:
  zero_z       — replace z_perception with zeros; measure CE degradation
  noise_z      — replace z_perception with random Gaussian noise
  mean_z       — replace z_perception with mean-pooled visual memory
  swap_z       — replace z_perception from sample A with z from sample B
  identity_z   — use actual z_perception (baseline, should be best)

Expected result if z is causally used:
  CE(zero_z) > CE(identity_z)
  CE(noise_z) > CE(identity_z)
  CE(mean_z) ~≥ CE(identity_z) (mean is weakly informative)
  CE(swap_z) > CE(identity_z)  (wrong evidence hurts)

If CE is similar across all modes, the model is NOT using z_perception.
This is the first thing to check.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DiagnosticMode(str, Enum):
    IDENTITY = "identity_z"
    ZERO = "zero_z"
    NOISE = "noise_z"
    MEAN = "mean_z"
    SWAP = "swap_z"


@dataclass
class DiagnosticResult:
    mode: str
    ce_loss: float
    n_samples: int
    ce_on_obs_text: Optional[float] = None   # CE specifically on observation text tokens
    ce_on_answer: Optional[float] = None     # CE specifically on answer tokens


class CausalDiagnostics:
    """
    Run diagnostic forward passes with different z_perception substitutions.
    Measures CE degradation to test causal utility of z_perception.
    """

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run_all(
        self,
        dataloader: DataLoader,
        max_samples: int = 100,
        modes: Optional[List[DiagnosticMode]] = None,
    ) -> Dict[str, DiagnosticResult]:
        """
        Run all diagnostic modes and return results.

        Args:
            dataloader: DataLoader of eval samples
            max_samples: how many samples to evaluate
            modes: which diagnostic modes to run (default: all)

        Returns:
            dict mapping mode name → DiagnosticResult
        """
        if modes is None:
            modes = list(DiagnosticMode)

        # Collect batches
        batches = []
        n = 0
        for batch in dataloader:
            batches.append(self._to_device(batch))
            n += batch["input_ids"].shape[0]
            if n >= max_samples:
                break

        results = {}
        for mode in modes:
            logger.info(f"[Diagnostics] Running mode={mode}")
            result = self._run_mode(batches, mode, max_samples)
            results[mode.value] = result
            logger.info(
                f"[Diagnostics] {mode.value}: ce={result.ce_loss:.4f} "
                f"(n={result.n_samples})"
            )

        self._print_summary(results)
        return results

    def _run_mode(
        self,
        batches: List[Dict],
        mode: DiagnosticMode,
        max_samples: int,
    ) -> DiagnosticResult:
        """Run a single diagnostic mode across all batches."""
        total_ce = 0.0
        n_samples = 0
        n_tokens = 0

        # For swap mode: collect z_perceptions from all batches first
        if mode == DiagnosticMode.SWAP:
            swap_pool = self._collect_z_perceptions(batches)
        else:
            swap_pool = None

        for batch_idx, batch in enumerate(batches):
            if n_samples >= max_samples:
                break

            ce, n_tok = self._forward_with_mode(batch, mode, swap_pool, batch_idx)
            total_ce += ce * n_tok
            n_tokens += n_tok
            n_samples += batch["input_ids"].shape[0]

        avg_ce = total_ce / max(n_tokens, 1)
        return DiagnosticResult(mode=mode.value, ce_loss=avg_ce, n_samples=n_samples)

    def _forward_with_mode(
        self,
        batch: Dict,
        mode: DiagnosticMode,
        swap_pool: Optional[List],
        batch_idx: int,
    ) -> Tuple[float, int]:
        """
        Run forward pass with the specified z_perception substitution.
        Returns (total_ce, num_supervised_tokens).
        """
        # Step 1: get visual memory and build modified sequence
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        visual_memory = self.model.encode_image_to_memory(pixel_values, image_grid_thw)
        mod_ids, mod_mask, mod_labels, position_ids = self.model.build_modified_sequence(
            input_ids, attention_mask, labels
        )

        B = mod_ids.shape[0]
        device = mod_ids.device

        # Step 2: Run pass 1 to get h_perception (identity needed for all modes)
        base_embeds = self.model.base_model.get_input_embeddings()(mod_ids).clone()
        perc_out_mask = (mod_ids == self.model.special_tokens.PERC_OUT)
        base_embeds[perc_out_mask] = 0.0

        pass1_out = self.model.base_model.model(
            inputs_embeds=base_embeds,
            attention_mask=mod_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False,
        )
        hs = pass1_out.last_hidden_state  # [B, T, D]

        # Step 3: Get z_perceptions for identity mode
        vm = visual_memory.unsqueeze(0)
        z_identity = []
        for b in range(B):
            perc_pos = (mod_ids[b] == self.model.special_tokens.PERCEPTION).nonzero(
                as_tuple=True
            )[0].tolist()
            if not perc_pos:
                z_identity.append(None)
                continue
            h_p = hs[b, perc_pos, :].unsqueeze(0)
            z, _ = self.model.perception_module(h_p, vm)
            z_identity.append(z.squeeze(0))

        # Step 4: Build z for this mode
        z_for_mode = self._apply_mode(z_identity, mode, visual_memory, swap_pool, batch_idx)

        # Step 5: Pass 2 with substituted z
        embeds_pass2 = self.model.base_model.get_input_embeddings()(mod_ids).clone()
        embeds_pass2 = embeds_pass2.to(self.model.dtype)

        for b in range(B):
            z_b = z_for_mode[b]
            if z_b is None:
                continue
            perc_out_pos = (mod_ids[b] == self.model.special_tokens.PERC_OUT).nonzero(
                as_tuple=True
            )[0].tolist()
            K = min(z_b.shape[0], len(perc_out_pos))
            for k in range(K):
                embeds_pass2[b, perc_out_pos[k], :] = z_b[k].to(self.model.dtype)

        pass2_out = self.model.base_model(
            inputs_embeds=embeds_pass2,
            attention_mask=mod_mask,
            position_ids=position_ids,
            labels=mod_labels,
            output_hidden_states=False,
            use_cache=False,
        )

        ce = pass2_out.loss.item() if pass2_out.loss is not None else 0.0
        n_sup = (mod_labels != -100).sum().item()
        return ce, max(n_sup, 1)

    def _apply_mode(
        self,
        z_identity: List[Optional[torch.Tensor]],
        mode: DiagnosticMode,
        visual_memory: torch.Tensor,
        swap_pool: Optional[List],
        batch_idx: int,
    ) -> List[Optional[torch.Tensor]]:
        """Substitute z_perception according to the diagnostic mode."""
        if mode == DiagnosticMode.IDENTITY:
            return z_identity

        result = []
        for b, z in enumerate(z_identity):
            if z is None:
                result.append(None)
                continue

            K, D = z.shape

            if mode == DiagnosticMode.ZERO:
                result.append(torch.zeros_like(z))

            elif mode == DiagnosticMode.NOISE:
                noise = torch.randn_like(z) * z.std()
                result.append(noise)

            elif mode == DiagnosticMode.MEAN:
                # Mean of visual memory, broadcast to [K, D]
                mean_z = visual_memory.mean(dim=0, keepdim=True).expand(K, -1)
                result.append(mean_z)

            elif mode == DiagnosticMode.SWAP:
                # Use z from a different sample
                swap_z = self._get_swap_z(swap_pool, batch_idx, b, K, D, z.device)
                result.append(swap_z)

            else:
                result.append(z)

        return result

    def _collect_z_perceptions(self, batches: List[Dict]) -> List[torch.Tensor]:
        """Collect all z_perceptions from the dataset for swap mode."""
        pool = []
        for batch in batches:
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            mod_ids, mod_mask, mod_labels, position_ids = self.model.build_modified_sequence(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
            visual_memory = self.model.encode_image_to_memory(pixel_values, image_grid_thw)
            vm = visual_memory.unsqueeze(0)

            base_embeds = self.model.base_model.get_input_embeddings()(mod_ids).clone()
            base_embeds[(mod_ids == self.model.special_tokens.PERC_OUT)] = 0.0
            out = self.model.base_model.model(
                inputs_embeds=base_embeds,
                attention_mask=mod_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            hs = out.last_hidden_state
            B = mod_ids.shape[0]
            for b in range(B):
                perc_pos = (mod_ids[b] == self.model.special_tokens.PERCEPTION).nonzero(
                    as_tuple=True
                )[0].tolist()
                if perc_pos:
                    h_p = hs[b, perc_pos, :].unsqueeze(0)
                    z, _ = self.model.perception_module(h_p, vm)
                    pool.append(z.squeeze(0).cpu())
        return pool

    def _get_swap_z(self, pool, batch_idx, b, K, D, device):
        if not pool:
            return torch.zeros(K, D, device=device)
        idx = (batch_idx * 7 + b * 13 + 1) % len(pool)
        swap = pool[idx].to(device)
        # Match K dimension
        if swap.shape[0] >= K:
            return swap[:K]
        return swap.mean(dim=0, keepdim=True).expand(K, -1)

    def _to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _print_summary(self, results: Dict[str, DiagnosticResult]):
        logger.info("\n" + "="*60)
        logger.info("CAUSAL DIAGNOSTIC SUMMARY")
        logger.info("="*60)
        baseline = results.get(DiagnosticMode.IDENTITY.value)
        baseline_ce = baseline.ce_loss if baseline else float("nan")
        logger.info(f"{'Mode':<20} {'CE Loss':>10} {'ΔCE vs identity':>18}")
        logger.info("-"*50)
        for mode_name, result in sorted(results.items()):
            delta = result.ce_loss - baseline_ce
            logger.info(f"{mode_name:<20} {result.ce_loss:>10.4f} {delta:>+18.4f}")
        logger.info("="*60)
        if baseline_ce > 0:
            zero_result = results.get(DiagnosticMode.ZERO.value)
            if zero_result:
                rel = (zero_result.ce_loss - baseline_ce) / baseline_ce * 100
                if rel > 5:
                    logger.info(f"✓ z_perception appears CAUSALLY USEFUL (zero_z is {rel:.1f}% worse)")
                else:
                    logger.info(f"✗ WARNING: z_perception may NOT be causally used (zero_z delta={rel:.1f}%)")
        logger.info("="*60 + "\n")
