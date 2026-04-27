#!/usr/bin/env bash
# setup.sh — Download Fashion-MNIST dataset and build fashion_model
#
# Usage:
#   ./setup.sh           # download data + build
#   ./setup.sh --data    # download data only
#   ./setup.sh --build   # build only (skip download)
#   ./setup.sh --test    # run smoke tests

set -euo pipefail

MIRROR="https://ossci-datasets.s3.amazonaws.com/fashion-mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)
TRAIN_CSV="fashion_train.csv"
TEST_CSV="fashion_test.csv"

# ── helpers ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

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
    ./fashion_model info

    if [[ ! -f model.bin ]]; then
        log "────────────────────────────────────────────────────────────"
        log "Training model (this may take 1-2 minutes)..."
        log "────────────────────────────────────────────────────────────"
        timeout 180 ./fashion_model || warn "Training timed out or failed"
    fi

    if [[ -f model.bin && -f "$TEST_CSV" ]]; then
        log "────────────────────────────────────────────────────────────"
        log "Test: fashion_model test (20 samples)"
        log "────────────────────────────────────────────────────────────"
        ./fashion_model test "$TEST_CSV" 20

        log "────────────────────────────────────────────────────────────"
        log "Test: fashion_model eval (full test set)"
        log "────────────────────────────────────────────────────────────"
        ./fashion_model eval "$TEST_CSV"
    fi
}

# ── main ───────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --data)   download_data ;;
    --build)  build ;;
    --test)   run_tests ;;
    all)      download_data; build ;;
    *)
        echo "Usage: $0 [--data | --build | --test | all]"
        echo ""
        echo "  (no args / all)  Download data and build"
        echo "  --data           Download and convert Fashion-MNIST CSVs only"
        echo "  --build          Build fashion_model only"
        echo "  --test           Run tests (requires model.bin)"
        exit 1
        ;;
esac
