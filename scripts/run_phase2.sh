#!/usr/bin/env bash
# Phase 2: LR sweep on mode=none, 2000-sample train set.
# Best mode from Phase 1: none (stable, lowest eval loss).
# Runs lr3e5 → lr1e4 → lr3e4 sequentially on GPU 0.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase2.sh 2>&1 | tee outputs/ablation/phase2_run.log

set -uo pipefail

PYTHON=/cortex/users/rabanof/conda_envs/qwen49/bin/python
ROOT=/cortex/users/rabanof/projects/active_perception
cd "$ROOT"

mkdir -p outputs/ablation

echo "========================================================================"
echo "PHASE 2 START: $(date)"
echo "Mode: none | LR sweep: 3e-5, 1e-4, 3e-4 | Train: 2000 samples"
echo "========================================================================"

for LR_TAG in lr3e5 lr1e4 lr3e4; do
    CFG="configs/phase2_${LR_TAG}.yaml"
    RUN_DIR="runs/phase2_${LR_TAG}"

    echo ""
    echo "--- Phase 2 [${LR_TAG}] TRAIN: $(date) ---"
    mkdir -p "$RUN_DIR"
    $PYTHON scripts/train.py \
        --config "$CFG" \
        2>&1 | tee "${RUN_DIR}/train_stdout.log"
    TRAIN_EXIT=${PIPESTATUS[0]}

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: train.py failed for lr=${LR_TAG} (exit $TRAIN_EXIT)" >&2
        continue
    fi

    echo ""
    echo "--- Phase 2 [${LR_TAG}] EVAL: $(date) ---"
    $PYTHON scripts/eval_checkpoint_summary.py \
        --run_dir "$RUN_DIR" \
        --config  "$CFG" \
        --checkpoint_tag best \
        --num_samples 50 \
        --output_dir outputs/ablation \
        2>&1 | tee "${RUN_DIR}/eval_summary.log"
    EVAL_EXIT=${PIPESTATUS[0]}

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: eval_checkpoint_summary.py failed for lr=${LR_TAG} (exit $EVAL_EXIT)" >&2
    fi

    echo ""
    echo "--- Phase 2 [${LR_TAG}] DONE: $(date) ---"
done

echo ""
echo "========================================================================"
echo "PHASE 2 COMPLETE: $(date)"
echo "========================================================================"
echo ""
echo "Summaries:"
for LR_TAG in lr3e5 lr1e4 lr3e4; do
    JSON="outputs/ablation/phase2_${LR_TAG}_summary.json"
    if [ -f "$JSON" ]; then
        echo ""
        echo "  [${LR_TAG}]"
        python3 -c "
import json
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
        echo "  [${LR_TAG}] summary not found"
    fi
done
