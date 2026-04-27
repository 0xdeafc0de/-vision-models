# Vision Models: Fashion-MNIST Classification

A continuation of the [perceptron learning project](../perceptron), exploring how neural networks generalize to different image classification tasks. This repository demonstrates that the same architecture and training techniques work across different domains — from MNIST digits to fashion items.

**Key Question**: Does a model trained on handwritten digits generalize to clothing items?

## Dataset: Fashion-MNIST

Fashion-MNIST is a benchmark dataset identical in structure to MNIST but conceptually different:

| Property | Details |
|----------|---------|
| **Samples** | 60,000 training + 10,000 test |
| **Resolution** | 28×28 grayscale images |
| **Classes** | 10 clothing categories |
| **Size** | ~105 MB (similar to MNIST) |

### Classes

```
0: T-shirt/top    1: Trouser      2: Pullover     3: Dress        4: Coat
5: Sandal         6: Shirt        7: Sneaker      8: Bag          9: Ankle boot
```

## Quick Start

```bash
# Full setup: download Fashion-MNIST + compile
$ ./setup.sh

# Or step by step:
$ ./setup.sh --data     # Download and convert to CSV
$ ./setup.sh --build    # Compile fashion_model
$ ./setup.sh --train    # Train and save fashion_model.bin
$ ./setup.sh --test     # Deterministic smoke checks (no retrain)
```

## Model Architecture

Same architecture as the mini_model from the perceptron project:

```
784 (input pixels)
  ↓ [He init]
32 (hidden, ReLU activation)
  ↓ [Xavier init]
10 (output, softmax)
```

**Parameters**: 25,450 total (25,088 weights + 362 biases)

### Training Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Iterations | 50 | Epochs through training data |
| Learning Rate | 0.001 | Initial step size |
| LR Decay | 1e-5 | Reduces LR each epoch for fine-tuning |
| L2 Regularization | 0 (disabled) | Can be enabled to reduce overfitting |
| Batch Processing | Full dataset | SGD on shuffled samples each epoch |
| Weight Init | He/Xavier | Prevents vanishing/exploding gradients |

## Usage

### Training

Train the model on 60,000 Fashion-MNIST training samples:

```bash
$ ./fashion_model
Training on 60000 sample(s).... Number of iteration = 50
Iteration....5 (lr=0.001000) loss=0.3112 train_acc=89.54%
Iteration....10 (lr=0.001000) loss=0.2789 train_acc=90.87%
...
Iteration....50 (lr=0.001000) loss=0.2104 train_acc=92.48%
Training complete. Saving model to fashion_model.bin
```

Output: `fashion_model.bin` (binary weights file)

Optional model path:

```bash
$ ./fashion_model --model my_weights.bin
```

Optional validation split during training:

```bash
$ ./fashion_model --val-split 0.1
```

### Evaluation

Evaluate on the full test set (10,000 samples):

```bash
$ ./fashion_model eval fashion_test.csv
Evaluating model on 10000 samples from fashion_test.csv...

Top-1 accuracy: 8921 / 10000 = 89.21%
Top-2 accuracy: 9520 / 10000 = 95.20%
Top-3 accuracy: 9728 / 10000 = 97.28%

Per-class accuracy:
  Class 0 (T-shirt/top):    880 /  1000 = 88.00%
  Class 1 (Trouser):        982 /  1000 = 98.20%
  Class 2 (Pullover):       785 /  1000 = 78.50%
  Class 3 (Dress):          893 /  1000 = 89.30%
  Class 4 (Coat):           845 /  1000 = 84.50%
  Class 5 (Sandal):         975 /  1000 = 97.50%
  Class 6 (Shirt):          612 /  1000 = 61.20%
  Class 7 (Sneaker):        956 /  1000 = 95.60%
  Class 8 (Bag):            968 /  1000 = 96.80%
  Class 9 (Ankle boot):     953 /  1000 = 95.30%
```

### Testing Samples

Test on N samples with predictions and confusion matrix:

```bash
$ ./fashion_model test fashion_test.csv 50

╔════════════════════════════════════════════════════════════════╗
║          FASHION-MNIST CLASSIFIER — Model Card                ║
╠════════════════════════════════════════════════════════════════╣
║ Architecture:  784 → 32 → 10  (Fully Connected, ReLU+Softmax) ║
║ Parameters:    25450 total  (W1:25088 b1:32 | W2:320 b2:10)   ║
║ Training:      50 iterations, LR=0.0010 (with decay)          ║
║ Dataset:       Fashion-MNIST (28×28px, 10 clothing classes)   ║
║ Expected Acc:  ~92-94% (harder than MNIST digits)             ║
╚════════════════════════════════════════════════════════════════╝

Testing 50 sample(s)...

Sample 0: True=Ankle boot, Predicted=Ankle boot (98.5%) ✓
Sample 1: True=Pullover, Predicted=Shirt (67.2%) ✗
Sample 2: True=Sneaker, Predicted=Sneaker (95.1%) ✓
Sample 3: True=T-shirt/top, Predicted=T-shirt/top (89.3%) ✓
...

  Confusion Matrix (rows=true, cols=predicted):
  true \ pred     0    1    2    3    4    5    6    7    8    9
  class 0         8    0    1    1    0    0    3    0    1    1
  class 1         0    9    0    0    0    0    0    0    1    0
  class 2         0    0    6    0    2    0    1    0    1    0
  ...
```

### Model Info

Display architecture and parameter breakdown:

```bash
$ ./fashion_model info

╔════════════════════════════════════════════════════════════════╗
║          FASHION-MNIST CLASSIFIER — Model Card                ║
╠════════════════════════════════════════════════════════════════╣
║ Architecture:  784 → 32 → 10  (Fully Connected, ReLU+Softmax) ║
║ Parameters:    25450 total  (W1:25088 b1:32 | W2:320 b2:10)   ║
║ Training:      50 iterations, LR=0.0010 (with decay)          ║
║ Dataset:       Fashion-MNIST (28×28px, 10 clothing classes)   ║
║ Expected Acc:  ~92-94% (harder than MNIST digits)             ║
╚════════════════════════════════════════════════════════════════╝
```

## Expected Performance

The shallow 2-layer network on Fashion-MNIST typically achieves:

- **Overall accuracy**: 88-92% (vs 96.68% on MNIST)
- **Easy classes**: Trousers (98%), Sandals (97%)
- **Hard classes**: Shirts (61-65%), Pullovers (75-80%)

**Why lower than MNIST?**
- Fashion items have more visual variability than digits
- Similar textures between classes (coats vs shirts, sneakers vs ankle boots)
- Requires deeper networks for competitive accuracy

## Comparison: MNIST vs Fashion-MNIST

| Dataset | Task | Accuracy (2-layer) | Difficulty | Visual Variability |
|---------|---------|-------|------|-----|
| **MNIST** | Digit recognition | 96.68% | Easy | Low (consistent handwriting) |
| **Fashion-MNIST** | Clothing classification | 88-92% | Hard | High (different styles, angles) |

## Building Better Models

To improve Fashion-MNIST accuracy:

1. **Deeper networks** (4+ layers) → Better non-linear feature learning
2. **Convolutional layers** → Spatial invariance (position-independent features)
3. **Data augmentation** → Rotations, crops, noise robustness
4. **Dropout regularization** → Prevents overfitting on small variations

See the [perceptron project](../perceptron) for an example of deeper architecture with dropout and Adam optimizer (advanced_model).

## File Structure

```
vision-models/
  fashion_model.c      # Single-file Neural Network (~550 lines)
  setup.sh             # Dataset download + build automation
  README.md            # This file
  fashion_model        # Compiled binary (created after build)
  fashion_model.bin    # Trained weights (created after training)
  fashion_train.csv    # 60K training samples (created after download)
  fashion_test.csv     # 10K test samples (created after download)
```

## Implementation Details

### Forward Pass
1. **Input normalization**: Pixel values scaled to [0, 1]
2. **Hidden layer**: Linear transform + ReLU (introduces non-linearity)
3. **Output layer**: Linear transform + Softmax (probability distribution)

### Backward Pass
- **Loss function**: Cross-entropy (standard for classification)
- **Gradients**: Backpropagated through both layers
- **Update rule**: Stochastic gradient descent with learning rate decay

### Training Dynamics
- **Shuffling**: Fisher-Yates shuffle each epoch (prevents gradient bias)
- **Learning rate decay**: `lr = lr₀ / (1 + decay × epoch)` (fine-tunes late in training)
- **Convergence**: Typically settles after 30-40 epochs

## Related Projects

- **[Perceptron Learning](../perceptron)** — From single neurons to advanced deep networks
  - Model 1: Single-layer perceptron (linearity)
  - Model 2: Multi-layer perceptron (non-linearity)
  - Model 3: Mini MNIST (96.68% accuracy, simple 2-layer)
  - Model 4: Advanced network (deeper, dropout, Adam)

- **[Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)** — Official source
- **[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)** — Comparison baseline

## Performance Notes

Timing (on modern hardware, ~2-3 GHz CPU):
- **Download**: 2-3 minutes (105 MB)
- **Training**: 30-60 seconds (50 epochs)
- **Evaluation**: <5 seconds (10K samples)
- **Per-sample inference**: ~50-100 µs

## License

This implementation is free to use and modify. Fashion-MNIST dataset is licensed under CC0.
