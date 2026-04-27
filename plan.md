# Vision Models Plan

Last updated: 2026-04-27
Project: Fashion-MNIST continuation of perceptron learning

## Current Status

Completed:
- Created `fashion_model.c` (2-layer NN: 784 -> 32 -> 10, ReLU + Softmax)
- Added training, eval, test, info commands
- Added confusion matrix and per-class accuracy reporting
- Added `setup.sh` for data download/build/test
- Added `README.md` and `.gitignore`
- Fixed setup download/convert flow for Fashion-MNIST
- Fixed macOS compatibility (removed `timeout` dependency)
- Fixed CLI crash on no-arg run; default now starts training
- Repository initialized and committed

Known-good commands:
- `./setup.sh --data`
- `./setup.sh --build`
- `./fashion_model` (train)
- `./fashion_model eval fashion_test.csv`
- `./fashion_model test fashion_test.csv 50`
- `./fashion_model info`

## Primary Goal

Build a practical Fashion-MNIST baseline that demonstrates transfer from MNIST concepts to a harder vision task, while keeping implementation readable and educational.

## Next Milestones

1. Model Artifact Hygiene
- Use a dedicated weights file name: `fashion_model.bin` (not generic `model.bin`)
- Add CLI option for model path (default to `fashion_model.bin`)
- Ensure README + setup are aligned with this naming

2. Setup Workflow Improvements
- Add `--train` mode to `setup.sh`
- Improve `--test` to run deterministic short checks without retraining unless needed
- Add clear success/failure summary at script end

3. Metrics & Reporting
- Add train loss logging every N iterations
- Add optional validation split and validation accuracy output
- Add top-2 / top-3 accuracy reporting for analysis

4. Reproducibility
- Add optional fixed seed via CLI (`--seed`)
- Print seed and config in model card
- Capture training config in saved model metadata (or sidecar text file)

5. Performance / Accuracy Upgrade Path
- Increase hidden size (32 -> 64/128) and benchmark speed/accuracy
- Try 2 hidden layers: 784 -> 128 -> 64 -> 10
- Add L2 regularization sweep and compare class-wise results

6. Documentation
- Add a benchmark section in README with actual measured results
- Add troubleshooting section (dataset errors, missing model, path mistakes)
- Add a short “resume from checkpoint” workflow

## Stretch Goals

- Add mini CNN implementation in C for stronger Fashion-MNIST accuracy
- Add export of confusion matrix to CSV
- Add tiny Python script for plotting metrics (loss/accuracy curves)

## Resume Checklist (when coming back)

1. Pull latest branch and run:
   - `./setup.sh --build`
2. Verify baseline:
   - `./fashion_model info`
   - `./fashion_model eval fashion_test.csv`
3. Start with Milestone 1 (artifact naming: `fashion_model.bin`)
4. Commit in small steps (one milestone per commit)

## Suggested Commit Sequence

- `feat: use fashion_model.bin and configurable model path`
- `feat: add --train mode and improve setup test flow`
- `feat: add validation split and richer metrics`
- `feat: add reproducibility controls (seed + config output)`
- `perf: benchmark hidden-size/depth variants`
- `docs: add benchmark and troubleshooting sections`
