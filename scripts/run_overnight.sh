#!/usr/bin/env bash
# Overnight experiment batch: 8 runs on 8 H100 GPUs (one per GPU).
#
# Sanity checks run first (sequentially on GPU 0), then all 8 training
# runs launch in parallel.  Each run is followed by a full eval sweep
# (z-ablation, grounding, generation accuracy, qualitative examples).
#
# Usage:
#   bash scripts/run_overnight.sh 2>&1 | tee outputs/overnight/overnight_run.log
#
# GPU assignments:
#   GPU0 → exp8_continue_latent_full22k_lr3e5   (load exp1 checkpoint)
#   GPU1 → exp9_none_full22k_lr3e4
#   GPU2 → exp9b_none_full22k_lr1e4
#   GPU3 → exp12_lora_full22k_none_lr2e5
#   GPU4 → exp12b_lora_full22k_none_lr5e5
#   GPU5 → exp11_ground_full22k_none_lambda005
#   GPU6 → exp13_lora_ground_full22k_none_lambda005_lr2e5
#   GPU7 → exp10_ground_warmup_full22k_none_500steps

set -uo pipefail

PYTHON=/cortex/users/rabanof/conda_envs/qwen49/bin/python
ROOT=/cortex/users/rabanof/projects/active_perception
EVAL_CFG=configs/exp1_ce_only.yaml   # for smoke tests (uses exp1 checkpoint)
SMOKE_CKPT=runs/exp1_ce_only/checkpoint-best

cd "$ROOT"
mkdir -p outputs/overnight outputs/ablation

echo "========================================================================"
echo "OVERNIGHT BATCH START: $(date)"
echo "========================================================================"

# ── Sanity checks (sequential, GPU 0) ────────────────────────────────────────
echo ""
echo "--- SMOKE TEST 1: Batch token/label inspection ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/smoke_test_batch.py \
    --config "$EVAL_CFG" \
    --checkpoint "$SMOKE_CKPT" \
    --n_samples 3 \
    2>&1 | tee outputs/overnight/smoke_test_batch.log
SMOKE1_EXIT=${PIPESTATUS[0]}

echo ""
echo "--- SMOKE TEST 2: Grounding loss verification ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/smoke_test_grounding.py \
    --config "$EVAL_CFG" \
    --checkpoint "$SMOKE_CKPT" \
    2>&1 | tee outputs/overnight/smoke_test_grounding.log
SMOKE2_EXIT=${PIPESTATUS[0]}

if [ $SMOKE1_EXIT -ne 0 ] || [ $SMOKE2_EXIT -ne 0 ]; then
    echo ""
    echo "ABORT: Sanity checks failed (smoke1=$SMOKE1_EXIT smoke2=$SMOKE2_EXIT)." >&2
    echo "Fix the issues before launching overnight runs." >&2
    exit 1
fi

echo ""
echo "All sanity checks passed. Launching 8 training runs in parallel."
echo "========================================================================"

# ── Helper: train + eval one experiment ──────────────────────────────────────
run_one() {
    local GPU="$1"
    local EXP="$2"
    local CFG="$3"
    local INIT_CKPT="${4:-}"      # optional --init_checkpoint

    local RUN_DIR="runs/${EXP}"
    local LOG_DIR="outputs/overnight/${EXP}"
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    echo "" >> "$LOG_DIR/run.log"
    echo "=== ${EXP} TRAIN START: $(date) ===" >> "$LOG_DIR/run.log"

    local INIT_ARG=""
    if [ -n "$INIT_CKPT" ]; then
        INIT_ARG="--init_checkpoint $INIT_CKPT"
    fi

    CUDA_VISIBLE_DEVICES="$GPU" $PYTHON scripts/train.py \
        --config "$CFG" $INIT_ARG \
        2>&1 | tee "$LOG_DIR/train_stdout.log"
    local TRAIN_EXIT=${PIPESTATUS[0]}

    echo "=== ${EXP} TRAIN DONE (exit=$TRAIN_EXIT): $(date) ===" >> "$LOG_DIR/run.log"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: train.py failed for $EXP (exit $TRAIN_EXIT)" >&2
        return 1
    fi

    echo "=== ${EXP} EVAL START: $(date) ===" >> "$LOG_DIR/run.log"
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

    echo "=== ${EXP} EVAL DONE (exit=$EVAL_EXIT): $(date) ===" >> "$LOG_DIR/run.log"
    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: eval failed for $EXP (exit $EVAL_EXIT)" >&2
    fi
}

# ── Launch all 8 training runs in parallel ────────────────────────────────────
(run_one 0 exp8_continue_latent_full22k_lr3e5 \
    configs/exp8_continue_latent_full22k_lr3e5.yaml \
    "$SMOKE_CKPT") &
PID0=$!

(run_one 1 exp9_none_full22k_lr3e4 \
    configs/exp9_none_full22k_lr3e4.yaml) &
PID1=$!

(run_one 2 exp9b_none_full22k_lr1e4 \
    configs/exp9b_none_full22k_lr1e4.yaml) &
PID2=$!

(run_one 3 exp12_lora_full22k_none_lr2e5 \
    configs/exp12_lora_full22k_none_lr2e5.yaml) &
PID3=$!

(run_one 4 exp12b_lora_full22k_none_lr5e5 \
    configs/exp12b_lora_full22k_none_lr5e5.yaml) &
PID4=$!

(run_one 5 exp11_ground_full22k_none_lambda005 \
    configs/exp11_ground_full22k_none_lambda005.yaml) &
PID5=$!

(run_one 6 exp13_lora_ground_full22k_none_lambda005_lr2e5 \
    configs/exp13_lora_ground_full22k_none_lambda005_lr2e5.yaml) &
PID6=$!

(run_one 7 exp10_ground_warmup_full22k_none_500steps \
    configs/exp10_ground_warmup_full22k_none_500steps.yaml) &
PID7=$!

echo ""
echo "All 8 runs launched. PIDs: $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7"
echo "Waiting for all to finish..."
echo ""

# Wait for all background jobs
EXIT_CODES=()
for PID in $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7; do
    wait "$PID"
    EXIT_CODES+=($?)
done

echo ""
echo "========================================================================"
echo "OVERNIGHT BATCH COMPLETE: $(date)"
echo "========================================================================"
echo ""
echo "Per-run exit codes: ${EXIT_CODES[*]}"
echo ""

# ── Print summary table from saved JSON files ─────────────────────────────────
echo "RESULTS SUMMARY:"
for EXP in \
    exp8_continue_latent_full22k_lr3e5 \
    exp9_none_full22k_lr3e4 \
    exp9b_none_full22k_lr1e4 \
    exp12_lora_full22k_none_lr2e5 \
    exp12b_lora_full22k_none_lr5e5 \
    exp11_ground_full22k_none_lambda005 \
    exp13_lora_ground_full22k_none_lambda005_lr2e5 \
    exp10_ground_warmup_full22k_none_500steps; do

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
print(f'    attn_entropy   : {fmt(gr.get(\"attn_entropy_mean\"))}')
print(f'    z delta_zero   : {fmt(z.get(\"delta_zero\"), \"+.4f\")}')
print(f'    z delta_random : {fmt(z.get(\"delta_random\"), \"+.4f\")}')
n_acc = z.get(\"normal\", {}).get(\"post_perc_acc_mean\")
z_acc = z.get(\"zero\",   {}).get(\"post_perc_acc_mean\")
print(f'    acc_post_perc  : normal={fmt(n_acc)}  zero={fmt(z_acc)}')
print(f'    gen_accuracy   : {fmt(ge.get(\"accuracy\"))}')
print(f'    verdict        : {d.get(\"verdict\", \"N/A\")}')
" 2>/dev/null || echo "    (parse error — see $JSON)"
    else
        echo ""
        echo "  [$EXP] summary not found"
    fi
done

echo ""
echo "========================================================================"
