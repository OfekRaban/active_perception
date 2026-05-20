#!/usr/bin/env bash
# Phase 1: 100-sample overfit ablation across initial_perception_mode.
# Runs none → latent → spatial sequentially on GPU 0.
# After each run, evaluates checkpoint-best and saves summary JSON.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase1.sh 2>&1 | tee outputs/ablation/phase1_run.log

set -uo pipefail

PYTHON=/cortex/users/rabanof/conda_envs/qwen49/bin/python
ROOT=/cortex/users/rabanof/projects/active_perception
cd "$ROOT"

mkdir -p outputs/ablation

echo "========================================================================"
echo "PHASE 1 START: $(date)"
echo "========================================================================"

for MODE in none latent spatial; do
    CFG="configs/phase1_${MODE}.yaml"
    RUN_DIR="runs/phase1_${MODE}"

    echo ""
    echo "--- Phase 1 [${MODE}] TRAIN: $(date) ---"
    mkdir -p "$RUN_DIR"
    $PYTHON scripts/train.py \
        --config "$CFG" \
        2>&1 | tee "${RUN_DIR}/train_stdout.log"
    TRAIN_EXIT=${PIPESTATUS[0]}

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: train.py failed for mode=${MODE} (exit $TRAIN_EXIT)" >&2
        continue
    fi

    echo ""
    echo "--- Phase 1 [${MODE}] EVAL: $(date) ---"
    $PYTHON scripts/eval_checkpoint_summary.py \
        --run_dir "$RUN_DIR" \
        --config  "$CFG" \
        --checkpoint_tag best \
        --num_samples 50 \
        --output_dir outputs/ablation \
        2>&1 | tee "${RUN_DIR}/eval_summary.log"
    EVAL_EXIT=${PIPESTATUS[0]}

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: eval_checkpoint_summary.py failed for mode=${MODE} (exit $EVAL_EXIT)" >&2
    fi

    echo ""
    echo "--- Phase 1 [${MODE}] DONE: $(date) ---"
done

echo ""
echo "========================================================================"
echo "PHASE 1 COMPLETE: $(date)"
echo "========================================================================"
echo ""
echo "Summaries:"
for MODE in none latent spatial; do
    JSON="outputs/ablation/phase1_${MODE}_summary.json"
    if [ -f "$JSON" ]; then
        echo ""
        echo "  [$MODE]"
        python3 -c "
import json, sys
d = json.load(open('$JSON'))
tl = d['training_log']
z  = d['z_ablation']
gr = d['grounding']
be  = tl.get('best_eval_loss')
ft  = tl.get('final_train_loss')
gap = tl.get('overfit_gap')
dz  = z.get('delta_zero')
dr  = z.get('delta_random')
grr = gr.get('grounding_ratio')
ent = gr.get('attn_entropy_mean')
pan = z.get('normal',{}).get('post_perc_acc_mean')
paz = z.get('zero',{}).get('post_perc_acc_mean')
fmt = lambda v, s='.4f': f'{v:{s}}' if v is not None else 'N/A'
print(f'    best_eval_loss : {fmt(be)}')
print(f'    final_train_loss: {fmt(ft)}')
print(f'    overfit_gap    : {fmt(gap, \"+.4f\")}')
print(f'    grounding_ratio: {fmt(grr)}')
print(f'    attn_entropy   : {fmt(ent)}')
print(f'    z delta_zero   : {fmt(dz, \"+.4f\")}')
print(f'    z delta_random : {fmt(dr, \"+.4f\")}')
print(f'    acc_post_perc  : normal={fmt(pan)}  zero={fmt(paz)}')
print(f'    verdict        : {d[\"verdict\"]}')
" 2>/dev/null || echo "    (parse error — see $JSON)"
    else
        echo "  [$MODE] summary not found"
    fi
done
