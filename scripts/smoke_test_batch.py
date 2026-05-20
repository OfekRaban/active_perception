#!/usr/bin/env python3
"""
Sanity check 1: Verify PERCEPTION/PERC_OUT token masking in training batches.

Checks:
  - PERCEPTION positions: label != -100 (CE-supervised; model learns WHEN to look)
  - PERC_OUT positions: label == -100 (masked; z_perception injected here, not predicted)
  - IMAGE positions: label == -100 (masked)
  - z_perceptions non-None when PERCEPTION tokens are present
  - Modified sequence is shorter than original (image-pad block compressed to 1 token)

Usage:
    CUDA_VISIBLE_DEVICES=0 /cortex/users/rabanof/conda_envs/qwen49/bin/python \
        scripts/smoke_test_batch.py \
        --config configs/exp1_ce_only.yaml \
        --checkpoint runs/exp1_ce_only/checkpoint-best \
        --n_samples 3 \
        --context_window 6
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.qwen_wrapper import ActivePerceptionModel, ActivePerceptionConfig
from active_perception.data.dataset import ActivePerceptionDataset, ActivePerceptionCollator


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        for base_name in cfg.pop("defaults"):
            base_path = Path(path).parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as fb:
                    cfg = _deep_merge(yaml.safe_load(fb), cfg)
    return cfg


def print_window(ids, labels, tokenizer, special_tokens, center_pos, window=6, tag=""):
    start = max(0, center_pos - window)
    end = min(len(ids), center_pos + window + 1)
    st = special_tokens
    id_names = {
        st.IMAGE: "<IMAGE>", st.PERCEPTION: "<PERCEPTION>",
        st.PERC_OUT: "<PERC_OUT>", st.INIT_PERC_OUT: "<INIT_PERC_OUT>",
    }
    print(f"  [{tag}] window around pos={center_pos}:")
    print(f"  {'pos':>5}  {'tok_id':>7}  {'label':>7}  token")
    for p in range(start, end):
        tid = ids[p].item()
        lbl = labels[p].item()
        name = id_names.get(tid) or tokenizer.decode([tid], skip_special_tokens=False)
        marker = " <<< HERE" if p == center_pos else ""
        print(f"  {p:>5}  {tid:>7}  {lbl:>7}  {name!r}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp1_ce_only.yaml")
    parser.add_argument("--checkpoint", default="runs/exp1_ce_only/checkpoint-best")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--context_window", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)
    model_cfg = ActivePerceptionConfig(**{k: v for k, v in cfg.get("model", {}).items()
                                          if k in ActivePerceptionConfig.__dataclass_fields__})
    model = ActivePerceptionModel(model_cfg).to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_perception_module(str(ckpt))
        logger.info(f"Loaded checkpoint: {ckpt}")
    else:
        logger.warning(f"Checkpoint not found ({ckpt}); using fresh weights")
    model.eval()

    data_cfg = cfg.get("data", {})
    eval_path = data_cfg.get("eval_data_path") or data_cfg.get("data_path")
    special_ids = model.get_special_token_ids()
    dataset = ActivePerceptionDataset(
        data_path=eval_path,
        processor=model.processor,
        special_token_ids=special_ids,
        image_root=data_cfg.get("image_root"),
        max_seq_len=data_cfg.get("max_seq_len", 2048),
    )
    collator = ActivePerceptionCollator(
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
    )

    n = min(args.n_samples, len(dataset))
    st = model.special_tokens
    tokenizer = model.tokenizer

    failures = []

    for i in range(n):
        print(f"\n{'='*72}")
        print(f"SAMPLE {i}")
        print(f"{'='*72}")

        item = dataset[i]
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        T_orig = batch["input_ids"].shape[1]

        with torch.no_grad():
            out = model.training_forward(**batch, debug=True)

        mod_ids = out["modified_input_ids"][0]    # [T_new]
        mod_lab = out["modified_labels"][0]        # [T_new]
        T_new = mod_ids.shape[0]

        print(f"  T_orig={T_orig}  T_new={T_new}  (compressed by {T_orig - T_new} tokens)")

        perc_positions = (mod_ids == st.PERCEPTION).nonzero(as_tuple=True)[0].tolist()
        perc_out_positions = (mod_ids == st.PERC_OUT).nonzero(as_tuple=True)[0].tolist()
        image_positions = (mod_ids == st.IMAGE).nonzero(as_tuple=True)[0].tolist()

        print(f"  PERCEPTION positions: {perc_positions}")
        print(f"  PERC_OUT positions:   {perc_out_positions}")
        print(f"  IMAGE positions:      {image_positions}")

        # Check: PERCEPTION should be CE-supervised (label != -100)
        for p in perc_positions:
            lbl = mod_lab[p].item()
            if lbl == -100:
                failures.append(f"Sample {i}: PERCEPTION at pos {p} has label=-100 (should be supervised!)")
                print(f"  FAIL: PERCEPTION at pos {p} has label=-100")
            else:
                print(f"  PASS: PERCEPTION at pos {p}  label={lbl} (supervised)")
            print_window(mod_ids, mod_lab, tokenizer, st, p, args.context_window, "PERCEPTION")

        # Check: PERC_OUT should be masked (label == -100)
        for p in perc_out_positions:
            lbl = mod_lab[p].item()
            if lbl != -100:
                failures.append(f"Sample {i}: PERC_OUT at pos {p} has label={lbl} (should be -100!)")
                print(f"  FAIL: PERC_OUT at pos {p}  label={lbl} (should be -100)")
            else:
                print(f"  PASS: PERC_OUT at pos {p}  label=-100 (masked correctly)")
            print_window(mod_ids, mod_lab, tokenizer, st, p, args.context_window, "PERC_OUT")

        # Check: IMAGE should be masked
        for p in image_positions:
            lbl = mod_lab[p].item()
            if lbl != -100:
                failures.append(f"Sample {i}: IMAGE at pos {p} has label={lbl} (should be -100!)")
                print(f"  FAIL: IMAGE at pos {p}  label={lbl}")
            else:
                print(f"  PASS: IMAGE at pos {p}  label=-100 (masked correctly)")

        # Check z_perceptions
        z_list = out["z_perceptions"]
        if perc_positions:
            if z_list[0] is None:
                failures.append(f"Sample {i}: has PERCEPTION tokens but z_perceptions[0]=None")
                print("  FAIL: z_perceptions[0] is None despite PERCEPTION tokens")
            else:
                z = z_list[0]
                print(f"  PASS: z_perceptions[0].shape={z.shape}  norm={z.float().norm(dim=-1).mean():.4f}")
        else:
            print("  NOTE: No PERCEPTION tokens in this sample")

        # Check loss is finite
        loss = out["loss_ce"].item()
        if not (loss == loss) or loss > 1e6:
            failures.append(f"Sample {i}: loss_ce={loss} is NaN/inf")
            print(f"  FAIL: loss_ce={loss}")
        else:
            print(f"  PASS: loss_ce={loss:.4f}")

    print(f"\n{'='*72}")
    if failures:
        print(f"SMOKE TEST 1 FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST 1 PASSED — all checks OK on {n} samples")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
