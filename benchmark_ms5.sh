#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

TRAIN_CSV="fashion_train.csv"
TEST_CSV="fashion_test.csv"
SEED=42
VAL_SPLIT=0.1

OUT_DIR="benchmarks/ms5"
BIN_DIR="$OUT_DIR/bin"
LOG_DIR="$OUT_DIR/logs"
MODEL_DIR="$OUT_DIR/models"

mkdir -p "$BIN_DIR" "$LOG_DIR" "$MODEL_DIR"

if [[ ! -f "$TRAIN_CSV" || ! -f "$TEST_CSV" ]]; then
    echo "Dataset CSV files are missing. Run: ./setup.sh --data"
    exit 1
fi

SUMMARY_CSV="$OUT_DIR/summary.csv"
PER_CLASS_CSV="$OUT_DIR/per_class.csv"

echo "config,hidden1,hidden2,l2,train_seconds,top1_acc,top2_acc,top3_acc" > "$SUMMARY_CSV"
echo "config,class_id,top1_acc" > "$PER_CLASS_CSV"

run_config() {
    local cfg="$1"
    local h1="$2"
    local h2="$3"
    local l2="$4"

    local bin="$BIN_DIR/fashion_model_${cfg}"
    local model="$MODEL_DIR/${cfg}.bin"
    local train_log="$LOG_DIR/${cfg}_train.log"
    local eval_log="$LOG_DIR/${cfg}_eval.log"

    echo "[ms5] Building $cfg (H1=$h1 H2=$h2 L2=$l2)..."
    gcc -Wall -O2 -DHIDDEN_UNITS="$h1" -DHIDDEN2_UNITS="$h2" -o "$bin" fashion_model.c -lm

    echo "[ms5] Training $cfg..."
    local start=$SECONDS
    "$bin" --seed "$SEED" --l2 "$l2" --val-split "$VAL_SPLIT" --model "$model" > "$train_log"
    local elapsed=$((SECONDS - start))

    echo "[ms5] Evaluating $cfg..."
    "$bin" --model "$model" eval "$TEST_CSV" > "$eval_log"

    local top1 top2 top3
    top1="$(awk -F'= ' '/Top-1 accuracy/{gsub("%", "", $2); print $2}' "$eval_log" | tr -d ' ')"
    top2="$(awk -F'= ' '/Top-2 accuracy/{gsub("%", "", $2); print $2}' "$eval_log" | tr -d ' ')"
    top3="$(awk -F'= ' '/Top-3 accuracy/{gsub("%", "", $2); print $2}' "$eval_log" | tr -d ' ')"

    echo "$cfg,$h1,$h2,$l2,$elapsed,$top1,$top2,$top3" >> "$SUMMARY_CSV"

    awk -v cfg="$cfg" '
        /^  Class/ {
            class_id = $2
            gsub(":", "", class_id)
            pct = $NF
            gsub("%", "", pct)
            printf "%s,%s,%s\n", cfg, class_id, pct
        }
    ' "$eval_log" >> "$PER_CLASS_CSV"

    echo "[ms5] Done: $cfg -> top1=${top1}% train=${elapsed}s"
}

# Hidden-size sweep
run_config "h32_l2_0" 32 0 0.0
run_config "h64_l2_0" 64 0 0.0
run_config "h128_l2_0" 128 0 0.0

# Two-hidden-layer experiment
run_config "h128_h64_l2_0" 128 64 0.0

# L2 sweep on strong single-hidden baseline
run_config "h128_l2_1e4" 128 0 0.0001
run_config "h128_l2_5e4" 128 0 0.0005
run_config "h128_l2_1e3" 128 0 0.001

echo "[ms5] Benchmark complete."
echo "[ms5] Summary: $SUMMARY_CSV"
echo "[ms5] Per-class: $PER_CLASS_CSV"
