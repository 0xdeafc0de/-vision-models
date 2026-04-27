#!/usr/bin/env bash
# setup.sh — Download Fashion-MNIST dataset and build fashion_model
#
# Usage:
#   ./setup.sh           # download data + build
#   ./setup.sh --data    # download data only
#   ./setup.sh --build   # build only (skip download)
#   ./setup.sh --train   # train model and save weights
#   ./setup.sh --test    # run deterministic smoke tests

set -euo pipefail

MIRROR="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)
TRAIN_CSV="fashion_train.csv"
TEST_CSV="fashion_test.csv"
MODEL_FILE="fashion_model.bin"
PASSED=0
FAILED=0
SKIPPED=0

# ── helpers ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

pass() { PASSED=$((PASSED + 1)); log "PASS: $*"; }
fail() { FAILED=$((FAILED + 1)); warn "FAIL: $*"; }
skip() { SKIPPED=$((SKIPPED + 1)); warn "SKIP: $*"; }

summary() {
    echo
    log "Summary: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped"
    if [[ $FAILED -eq 0 ]]; then
        log "Completed successfully."
    else
        warn "Completed with failures."
    fi
}

require() {
    command -v "$1" &>/dev/null || die "'$1' is required but not found. Please install it."
}

# ── download ───────────────────────────────────────────────────────────────────
download_data() {
    require python3

    log "Downloading Fashion-MNIST binary files from $MIRROR..."
    for f in "${FILES[@]}"; do
        if [[ -f "$f" ]]; then
            log "  $f already exists, skipping."
        else
            log "  Downloading $f..."
            python3 -c "
import urllib.request
urllib.request.urlretrieve('$MIRROR/$f', '$f')
print('  Done.')
"
        fi
    done

    log "Converting to CSV..."
    python3 - <<'PYEOF'
import gzip, struct, csv, array, os, sys

def read_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = array.array('B', f.read())
    return n, rows * cols, data

def read_labels(path):
    with gzip.open(path, 'rb') as f:
        struct.unpack('>II', f.read(8))
        data = array.array('B', f.read())
    return data

def convert(img_file, lbl_file, out_csv):
    if os.path.exists(out_csv):
        print(f'  {out_csv} already exists, skipping.')
        return
    print(f'  Converting {img_file} -> {out_csv}...')
    n, pixels, images = read_images(img_file)
    labels = read_labels(lbl_file)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(n):
            w.writerow([labels[i]] + list(images[i*pixels:(i+1)*pixels]))
    print(f'  Done: {n} samples written to {out_csv}.')

convert('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 'fashion_train.csv')
convert('t10k-images-idx3-ubyte.gz',  't10k-labels-idx1-ubyte.gz',  'fashion_test.csv')
PYEOF

    log "Dataset ready: $TRAIN_CSV (60000 samples), $TEST_CSV (10000 samples)"
}

# ── build ──────────────────────────────────────────────────────────────────────
build() {
    require gcc

    log "Building fashion_model..."
    gcc -Wall -O2 -o fashion_model fashion_model.c -lm
    log "  Built: fashion_model"
}

# ── smoke test ─────────────────────────────────────────────────────────────────
run_tests() {
    log "────────────────────────────────────────────────────────────"
    log "Test: fashion_model info"
    log "────────────────────────────────────────────────────────────"
    if ./fashion_model info; then
        pass "fashion_model info"
    else
        fail "fashion_model info"
    fi

    if [[ ! -f "$MODEL_FILE" ]]; then
        skip "$MODEL_FILE missing (run ./setup.sh --train first)"
        return
    fi

    if [[ ! -f "$TEST_CSV" ]]; then
        skip "$TEST_CSV missing (run ./setup.sh --data first)"
        return
    fi

    log "────────────────────────────────────────────────────────────"
    log "Test: fashion_model test (5 samples, deterministic order)"
    log "────────────────────────────────────────────────────────────"
    local test_output
    if test_output=$(./fashion_model test "$TEST_CSV" 5 2>&1); then
        printf "%s\n" "$test_output"
        if [[ "$test_output" == *"nan%"* ]]; then
            fail "fashion_model test $TEST_CSV 5 (NaN probabilities detected)"
        else
            pass "fashion_model test $TEST_CSV 5"
        fi
    else
        printf "%s\n" "$test_output"
        fail "fashion_model test $TEST_CSV 5"
    fi
}

run_train() {
    [[ -f "$TRAIN_CSV" ]] || die "$TRAIN_CSV missing. Run ./setup.sh --data first."

    log "────────────────────────────────────────────────────────────"
    log "Train: fashion_model"
    log "────────────────────────────────────────────────────────────"
    if ./fashion_model; then
        if [[ -f "$MODEL_FILE" ]]; then
            pass "training wrote $MODEL_FILE"
        else
            fail "training completed but $MODEL_FILE not found"
        fi
    else
        fail "training command failed"
    fi
}

# ── main ───────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --data)   download_data; pass "download and conversion" ;;
    --build)  build; pass "build fashion_model" ;;
    --train)  run_train ;;
    --test)   run_tests ;;
    all)      download_data; pass "download and conversion"; build; pass "build fashion_model" ;;
    *)
        echo "Usage: $0 [--data | --build | --train | --test | all]"
        echo ""
        echo "  (no args / all)  Download data and build"
        echo "  --data           Download and convert Fashion-MNIST CSVs only"
        echo "  --build          Build fashion_model only"
        echo "  --train          Train and write $MODEL_FILE"
        echo "  --test           Run deterministic tests (requires $MODEL_FILE)"
        exit 1
        ;;
esac

summary

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
