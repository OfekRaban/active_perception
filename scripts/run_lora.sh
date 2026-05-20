#!/usr/bin/env bash
# Rerun the 3 failed LoRA experiments now that peft is installed.
#
# GPU 0: exp12  → then exp13  (chained)
# GPU 1: exp12b (parallel)
#
# Usage:
#   bash scripts/run_lora.sh 2>&1 | tee outputs/overnight/lora_run.log

set -uo pipefail

PYTHON=/cortex/users/rabanof/conda_envs/qwen49/bin/python
ROOT=/cortex/users/rabanof/projects/active_perception
cd "$ROOT"

mkdir -p outputs/overnight/exp12_lora_full22k_none_lr2e5
mkdir -p outputs/overnight/exp12b_lora_full22k_none_lr5e5
mkdir -p outputs/overnight/exp13_lora_ground_full22k_none_lambda005_lr2e5
mkdir -p outputs/ablation

echo "========================================================================"
echo "LoRA RUN START: $(date)"
echo "peft version: $($PYTHON -c 'import peft; print(peft.__version__)')"
echo "GPU 0: exp12 → exp13 (chained)"
echo "GPU 1: exp12b (parallel)"
echo "========================================================================"

run_one() {
    local GPU="$1"
    local EXP="$2"
    local CFG="$3"

    local RUN_DIR="runs/${EXP}"
    local LOG_DIR="outputs/overnight/${EXP}"
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    echo "" ; echo "--- ${EXP} TRAIN START [GPU${GPU}]: $(date) ---"
    CUDA_VISIBLE_DEVICES="$GPU" $PYTHON scripts/train.py --config "$CFG" \
        2>&1 | tee "$LOG_DIR/train_stdout.log"
    local TRAIN_EXIT=${PIPESTATUS[0]}

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: train.py failed for $EXP (exit $TRAIN_EXIT)" >&2
        return 1
    fi

    echo "" ; echo "--- ${EXP} EVAL START [GPU${GPU}]: $(date) ---"
    CUDA_VISIBLE_DEVICES="$GPU" $PYTHON scripts/eval_checkpoint_summary.py \
        --run_dir  "$RUN_DIR" \
        --config   "$CFG" \
        --checkpoint_tag best \
        --num_samples  50 \
        --gen_samples  100 \
        --qualitative_n 10 \
        --output_dir  outputs/ablation \
        2>&1 | tee "$LOG_DIR/eval_stdout.log"
    local EVAL_EXIT=${PIPESTATUS[0]}

    echo "--- ${EXP} DONE [GPU${GPU}] (train=$TRAIN_EXIT eval=$EVAL_EXIT): $(date) ---"
}

# GPU 0: exp12 then exp13 sequentially
(run_one 0 exp12_lora_full22k_none_lr2e5 \
     configs/exp12_lora_full22k_none_lr2e5.yaml && \
 run_one 0 exp13_lora_ground_full22k_none_lambda005_lr2e5 \
     configs/exp13_lora_ground_full22k_none_lambda005_lr2e5.yaml) &
PID0=$!

# GPU 1: exp12b
(run_one 1 exp12b_lora_full22k_none_lr5e5 \
     configs/exp12b_lora_full22k_none_lr5e5.yaml) &
PID1=$!

echo "Launched. PIDs: $PID0 $PID1"
wait "$PID0"; E0=$?
wait "$PID1"; E1=$?

echo ""
echo "========================================================================"
echo "LoRA RUN COMPLETE: $(date)"
echo "Exit codes: gpu0_chain=$E0  gpu1=$E1"
echo "========================================================================"

for EXP in exp12_lora_full22k_none_lr2e5 exp12b_lora_full22k_none_lr5e5 \
           exp13_lora_ground_full22k_none_lambda005_lr2e5; do
    JSON="outputs/ablation/${EXP}_summary.json"
    if [ -f "$JSON" ]; then
        echo ""
        echo "  [$EXP]"
        $PYTHON -c "
import json
d = json.load(open('$JSON'))
tl = d.get('training_log', {})
z  = d.get('z_ablation', {})
gr = d.get('grounding', {})
ge = d.get('generation_eval', {})
fmt = lambda v, s='.4f': f'{v:{s}}' if v is not None else 'N/A'
print(f'    best_eval_loss : {fmt(tl.get(\"best_eval_loss\"))}')
print(f'    final_train    : {fmt(tl.get(\"final_train_loss\"))}')
print(f'    overfit_gap    : {fmt(tl.get(\"overfit_gap\"), \"+.4f\")}')
print(f'    grounding_ratio: {fmt(gr.get(\"grounding_ratio\"))}')
print(f'    z delta_zero   : {fmt(z.get(\"delta_zero\"), \"+.4f\")}')
n_acc = z.get(\"normal\", {}).get(\"post_perc_acc_mean\")
z_acc = z.get(\"zero\",   {}).get(\"post_perc_acc_mean\")
print(f'    acc_post_perc  : normal={fmt(n_acc)}  zero={fmt(z_acc)}')
print(f'    gen_accuracy   : {fmt(ge.get(\"accuracy\"))}')
print(f'    verdict        : {d.get(\"verdict\", \"N/A\")}')
" 2>/dev/null || echo "    (summary not found)"
    else
        echo "  [$EXP] summary not found"
    fi
done
