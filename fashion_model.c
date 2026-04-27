#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Fashion-MNIST Classifier
 * Classifies clothing/fashion items into 10 categories
 * Architecture: 784 (input) → 32 (hidden, ReLU) → 10 (output, softmax)
 * Dataset: Fashion-MNIST (60K training, 10K test, 28×28 grayscale)
 * 
 * Classes:
 *   0: T-shirt/top       5: Sandal
 *   1: Trouser          6: Shirt
 *   2: Pullover         7: Sneaker
 *   3: Dress            8: Bag
 *   4: Coat             9: Ankle boot
 */

#ifndef INPUT_SIZE
#define INPUT_SIZE 784
#endif

#ifndef HIDDEN_UNITS
#define HIDDEN_UNITS 32
#endif

#ifndef HIDDEN2_UNITS
#define HIDDEN2_UNITS 0
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 10
#endif

#ifndef MAX_SAMPLES
#define MAX_SAMPLES 70000
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 50
#endif

#ifndef LOG_EVERY
#define LOG_EVERY 5
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.001
#endif

#ifndef LR_DECAY_RATE
#define LR_DECAY_RATE 1e-5
#endif

#define DEFAULT_L2_LAMBDA 0.0f
#define DEFAULT_MODEL_PATH "fashion_model.bin"

static float g_l2_lambda = DEFAULT_L2_LAMBDA;

typedef struct {
    float *W1, *b1;  // Input → Hidden
    float *W2, *b2;  // Hidden1 → Hidden2 or Hidden1 → Output
    float *W3, *b3;  // Hidden2 → Output (only used when HIDDEN2_UNITS > 0)
} FashionModel;

typedef struct {
    float *X;
    float *y;
    int n_samples;
} Dataset;

const char *fashion_labels[] = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

int predict(FashionModel *model, float *x);

void print_model_card(const char *model_path, unsigned int seed, int seed_set, double val_split);

// ─────────────────────────────────────────────────────────────
// Activation Functions
// ─────────────────────────────────────────────────────────────

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float *x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    // Guard against non-finite normalization to avoid propagating NaNs.
    if (!isfinite(sum) || sum <= 0.0f) {
        float uniform = 1.0f / size;
        for (int i = 0; i < size; i++) {
            x[i] = uniform;
        }
        return;
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ─────────────────────────────────────────────────────────────
// Weight Initialization
// ─────────────────────────────────────────────────────────────

void he_init(float *w, int fan_in, int fan_out) {
    float limit = sqrt(6.0 / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        w[i] = (2.0 * rand() / RAND_MAX - 1.0) * limit;
    }
}

void xavier_init(float *w, int fan_in, int fan_out) {
    float limit = sqrt(6.0 / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        w[i] = (2.0 * rand() / RAND_MAX - 1.0) * limit;
    }
}

// ─────────────────────────────────────────────────────────────
// Model Initialization
// ─────────────────────────────────────────────────────────────

FashionModel* init_model() {
    FashionModel *model = (FashionModel*)malloc(sizeof(FashionModel));
    
    model->W1 = (float*)malloc(INPUT_SIZE * HIDDEN_UNITS * sizeof(float));
    model->b1 = (float*)calloc(HIDDEN_UNITS, sizeof(float));

    if (HIDDEN2_UNITS > 0) {
        model->W2 = (float*)malloc(HIDDEN_UNITS * HIDDEN2_UNITS * sizeof(float));
        model->b2 = (float*)calloc(HIDDEN2_UNITS, sizeof(float));
        model->W3 = (float*)malloc(HIDDEN2_UNITS * OUTPUT_SIZE * sizeof(float));
        model->b3 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    } else {
        model->W2 = (float*)malloc(HIDDEN_UNITS * OUTPUT_SIZE * sizeof(float));
        model->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
        model->W3 = NULL;
        model->b3 = NULL;
    }
    
    // He initialization for ReLU hidden layer
    he_init(model->W1, INPUT_SIZE, HIDDEN_UNITS);

    if (HIDDEN2_UNITS > 0) {
        he_init(model->W2, HIDDEN_UNITS, HIDDEN2_UNITS);
        xavier_init(model->W3, HIDDEN2_UNITS, OUTPUT_SIZE);
    } else {
        // Xavier for softmax output
        xavier_init(model->W2, HIDDEN_UNITS, OUTPUT_SIZE);
    }
    
    return model;
}

void free_model(FashionModel *model) {
    if (model) {
        free(model->W1);
        free(model->b1);
        free(model->W2);
        free(model->b2);
        free(model->W3);
        free(model->b3);
        free(model);
    }
}

// ─────────────────────────────────────────────────────────────
// Forward Pass
// ─────────────────────────────────────────────────────────────

typedef struct {
    float *z1, *a1;  // Hidden layer 1
    float *z2, *a2;  // Hidden layer 2 (optional)
    float *z3, *a3;  // Output layer
} ForwardCache;

ForwardCache* forward_pass(FashionModel *model, float *x) {
    ForwardCache *cache = (ForwardCache*)malloc(sizeof(ForwardCache));
    
    cache->z1 = (float*)malloc(HIDDEN_UNITS * sizeof(float));
    cache->a1 = (float*)malloc(HIDDEN_UNITS * sizeof(float));
    
    // Hidden layer: ReLU
    for (int j = 0; j < HIDDEN_UNITS; j++) {
        cache->z1[j] = model->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            cache->z1[j] += model->W1[i * HIDDEN_UNITS + j] * x[i];
        }
        cache->a1[j] = relu(cache->z1[j]);
    }
    
    if (HIDDEN2_UNITS > 0) {
        cache->z2 = (float*)malloc(HIDDEN2_UNITS * sizeof(float));
        cache->a2 = (float*)malloc(HIDDEN2_UNITS * sizeof(float));

        for (int j = 0; j < HIDDEN2_UNITS; j++) {
            cache->z2[j] = model->b2[j];
            for (int i = 0; i < HIDDEN_UNITS; i++) {
                cache->z2[j] += model->W2[i * HIDDEN2_UNITS + j] * cache->a1[i];
            }
            cache->a2[j] = relu(cache->z2[j]);
        }

        cache->z3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        cache->a3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            cache->z3[j] = model->b3[j];
            for (int i = 0; i < HIDDEN2_UNITS; i++) {
                cache->z3[j] += model->W3[i * OUTPUT_SIZE + j] * cache->a2[i];
            }
            cache->a3[j] = cache->z3[j];
        }
        softmax(cache->a3, OUTPUT_SIZE);
    } else {
        cache->z2 = NULL;
        cache->a2 = NULL;

        cache->z3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
        cache->a3 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            cache->z3[j] = model->b2[j];
            for (int i = 0; i < HIDDEN_UNITS; i++) {
                cache->z3[j] += model->W2[i * OUTPUT_SIZE + j] * cache->a1[i];
            }
            cache->a3[j] = cache->z3[j];
        }
        softmax(cache->a3, OUTPUT_SIZE);
    }
    
    return cache;
}

void free_cache(ForwardCache *cache) {
    if (cache) {
        free(cache->z1);
        free(cache->a1);
        free(cache->z2);
        free(cache->a2);
        free(cache->z3);
        free(cache->a3);
        free(cache);
    }
}

// ─────────────────────────────────────────────────────────────
// Backward Pass (Simplified)
// ─────────────────────────────────────────────────────────────

void backward_pass(FashionModel *model, ForwardCache *cache, float *x, int true_label, float lr) {
    // Output layer gradients (cross-entropy + softmax)
    float dz2[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dz2[i] = cache->a3[i];
        if (i == true_label) dz2[i] -= 1;
    }

    if (HIDDEN2_UNITS > 0) {
        float dz_hidden2[HIDDEN2_UNITS];
        for (int i = 0; i < HIDDEN2_UNITS; i++) {
            float grad_sum = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                grad_sum += model->W3[i * OUTPUT_SIZE + j] * dz2[j];
            }
            dz_hidden2[i] = grad_sum * relu_derivative(cache->z2[i]);
        }

        float dz1[HIDDEN_UNITS];
        for (int i = 0; i < HIDDEN_UNITS; i++) {
            float grad_sum = 0.0f;
            for (int j = 0; j < HIDDEN2_UNITS; j++) {
                grad_sum += model->W2[i * HIDDEN2_UNITS + j] * dz_hidden2[j];
            }
            dz1[i] = grad_sum * relu_derivative(cache->z1[i]);
        }

        // Update W3 and b3
        for (int i = 0; i < HIDDEN2_UNITS; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float grad = cache->a2[i] * dz2[j] + g_l2_lambda * model->W3[i * OUTPUT_SIZE + j];
                model->W3[i * OUTPUT_SIZE + j] -= lr * grad;
            }
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            model->b3[j] -= lr * dz2[j];
        }

        // Update W2 and b2
        for (int i = 0; i < HIDDEN_UNITS; i++) {
            for (int j = 0; j < HIDDEN2_UNITS; j++) {
                float grad = cache->a1[i] * dz_hidden2[j] + g_l2_lambda * model->W2[i * HIDDEN2_UNITS + j];
                model->W2[i * HIDDEN2_UNITS + j] -= lr * grad;
            }
        }
        for (int j = 0; j < HIDDEN2_UNITS; j++) {
            model->b2[j] -= lr * dz_hidden2[j];
        }

        // Update W1 and b1
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_UNITS; j++) {
                float grad = x[i] * dz1[j] + g_l2_lambda * model->W1[i * HIDDEN_UNITS + j];
                model->W1[i * HIDDEN_UNITS + j] -= lr * grad;
            }
        }
        for (int j = 0; j < HIDDEN_UNITS; j++) {
            model->b1[j] -= lr * dz1[j];
        }
    } else {
        float dz1[HIDDEN_UNITS];
        for (int i = 0; i < HIDDEN_UNITS; i++) {
            float grad_sum = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                grad_sum += model->W2[i * OUTPUT_SIZE + j] * dz2[j];
            }
            dz1[i] = grad_sum * relu_derivative(cache->z1[i]);
        }

        // Update W2 and b2
        for (int i = 0; i < HIDDEN_UNITS; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float grad = cache->a1[i] * dz2[j] + g_l2_lambda * model->W2[i * OUTPUT_SIZE + j];
                model->W2[i * OUTPUT_SIZE + j] -= lr * grad;
            }
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            model->b2[j] -= lr * dz2[j];
        }

        // Update W1 and b1
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_UNITS; j++) {
                float grad = x[i] * dz1[j] + g_l2_lambda * model->W1[i * HIDDEN_UNITS + j];
                model->W1[i * HIDDEN_UNITS + j] -= lr * grad;
            }
        }

        for (int j = 0; j < HIDDEN_UNITS; j++) {
            model->b1[j] -= lr * dz1[j];
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Dataset Management
// ─────────────────────────────────────────────────────────────

Dataset* load_dataset(const char *filename, int max_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->X = (float*)malloc(max_samples * INPUT_SIZE * sizeof(float));
    dataset->y = (float*)malloc(max_samples * sizeof(float));
    dataset->n_samples = 0;
    
    char line[10000];
    while (fgets(line, sizeof(line), file) && dataset->n_samples < max_samples) {
        int label = atoi(strtok(line, ","));
        dataset->y[dataset->n_samples] = label;
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            char *token = strtok(NULL, ",");
            dataset->X[dataset->n_samples * INPUT_SIZE + i] = atoi(token) / 255.0;
        }
        dataset->n_samples++;
    }
    
    fclose(file);
    return dataset;
}

void free_dataset(Dataset *dataset) {
    if (dataset) {
        free(dataset->X);
        free(dataset->y);
        free(dataset);
    }
}

// Fisher-Yates shuffle
void shuffle_indices(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// ─────────────────────────────────────────────────────────────
// Training
// ─────────────────────────────────────────────────────────────

float cross_entropy_loss(ForwardCache *cache, int true_label) {
    float p = cache->a3[true_label];
    if (p < 1e-8f) p = 1e-8f;
    return -logf(p);
}

float evaluate_top1_accuracy_on_indices(FashionModel *model, Dataset *dataset, int *indices, int n) {
    if (n <= 0) return 0.0f;

    int correct = 0;
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        int pred = predict(model, &dataset->X[idx * INPUT_SIZE]);
        if (pred == (int)dataset->y[idx]) correct++;
    }
    return 100.0f * correct / n;
}

void train_model(FashionModel *model, Dataset *dataset,
                 int *train_indices, int n_train,
                 int *val_indices, int n_val) {
    int *indices = (int*)malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = train_indices[i];

    printf("Training on %d sample(s).... Number of iteration = %d\n", n_train, NUM_ITERATIONS);
    if (n_val > 0) {
        printf("Validation split active: %d sample(s)\n", n_val);
    }

    for (int epoch = 0; epoch < NUM_ITERATIONS; epoch++) {
        float lr = LEARNING_RATE / (1.0 + LR_DECAY_RATE * epoch);
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        shuffle_indices(indices, n_train);

        for (int i = 0; i < n_train; i++) {
            int idx = indices[i];
            float *x = &dataset->X[idx * INPUT_SIZE];
            int label = (int)dataset->y[idx];

            ForwardCache *cache = forward_pass(model, x);

            int pred = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (cache->a3[j] > cache->a3[pred]) pred = j;
            }
            if (pred == label) epoch_correct++;
            epoch_loss += cross_entropy_loss(cache, label);

            backward_pass(model, cache, x, label, lr);
            free_cache(cache);
        }

        if ((epoch + 1) % LOG_EVERY == 0 || epoch + 1 == NUM_ITERATIONS) {
            float train_loss = epoch_loss / n_train;
            float train_acc = 100.0f * epoch_correct / n_train;
            printf("Iteration....%d (lr=%.6f) loss=%.4f train_acc=%.2f%%",
                   epoch + 1, lr, train_loss, train_acc);

            if (n_val > 0) {
                float val_acc = evaluate_top1_accuracy_on_indices(model, dataset, val_indices, n_val);
                printf(" val_acc=%.2f%%", val_acc);
            }

            printf("\n");
        }
    }

    free(indices);
}

// ─────────────────────────────────────────────────────────────
// Inference & Evaluation
// ─────────────────────────────────────────────────────────────

int predict(FashionModel *model, float *x) {
    ForwardCache *cache = forward_pass(model, x);
    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (cache->a3[i] > cache->a3[best]) best = i;
    }
    free_cache(cache);
    return best;
}

void top3_from_probs(float *probs, int *top1, int *top2, int *top3) {
    *top1 = 0;
    *top2 = 1;
    *top3 = 2;

    if (probs[*top2] > probs[*top1]) {
        int tmp = *top1;
        *top1 = *top2;
        *top2 = tmp;
    }
    if (probs[*top3] > probs[*top2]) {
        int tmp = *top2;
        *top2 = *top3;
        *top3 = tmp;
    }
    if (probs[*top2] > probs[*top1]) {
        int tmp = *top1;
        *top1 = *top2;
        *top2 = tmp;
    }

    for (int i = 3; i < OUTPUT_SIZE; i++) {
        if (probs[i] > probs[*top1]) {
            *top3 = *top2;
            *top2 = *top1;
            *top1 = i;
        } else if (probs[i] > probs[*top2]) {
            *top3 = *top2;
            *top2 = i;
        } else if (probs[i] > probs[*top3]) {
            *top3 = i;
        }
    }
}

void evaluate(FashionModel *model, Dataset *dataset) {
    int top1_correct = 0;
    int top2_correct = 0;
    int top3_correct = 0;
    int per_class_correct[OUTPUT_SIZE] = {0};
    int per_class_total[OUTPUT_SIZE] = {0};
    
    for (int i = 0; i < dataset->n_samples; i++) {
        float *x = &dataset->X[i * INPUT_SIZE];
        int label = (int)dataset->y[i];

        ForwardCache *cache = forward_pass(model, x);
        int t1, t2, t3;
        top3_from_probs(cache->a3, &t1, &t2, &t3);

        if (t1 == label) top1_correct++;
        if (t1 == label || t2 == label) top2_correct++;
        if (t1 == label || t2 == label || t3 == label) top3_correct++;

        per_class_correct[label] += (t1 == label);
        per_class_total[label]++;

        free_cache(cache);
    }
    
    printf("\nTop-1 accuracy: %d / %d = %.2f%%\n", top1_correct, dataset->n_samples,
           100.0 * top1_correct / dataset->n_samples);
    printf("Top-2 accuracy: %d / %d = %.2f%%\n", top2_correct, dataset->n_samples,
           100.0 * top2_correct / dataset->n_samples);
    printf("Top-3 accuracy: %d / %d = %.2f%%\n\n", top3_correct, dataset->n_samples,
           100.0 * top3_correct / dataset->n_samples);
    
    printf("Per-class accuracy:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  Class %d (%s): %4d / %5d = %.2f%%\n", i, fashion_labels[i],
               per_class_correct[i], per_class_total[i],
               100.0 * per_class_correct[i] / per_class_total[i]);
    }
}

// ─────────────────────────────────────────────────────────────
// Model Persistence
// ─────────────────────────────────────────────────────────────

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t input_size;
    uint32_t hidden1_units;
    uint32_t hidden2_units;
    uint32_t output_size;
} ModelHeader;

#define MODEL_MAGIC 0x464D4F44u  // "FMOD"
#define MODEL_VERSION 1u

void save_model(FashionModel *model, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("save_model");
        return;
    }

    ModelHeader header;
    header.magic = MODEL_MAGIC;
    header.version = MODEL_VERSION;
    header.input_size = INPUT_SIZE;
    header.hidden1_units = HIDDEN_UNITS;
    header.hidden2_units = HIDDEN2_UNITS;
    header.output_size = OUTPUT_SIZE;

    fwrite(&header, sizeof(ModelHeader), 1, f);
    
    fwrite(model->W1, sizeof(float), INPUT_SIZE * HIDDEN_UNITS, f);
    fwrite(model->b1, sizeof(float), HIDDEN_UNITS, f);

    if (HIDDEN2_UNITS > 0) {
        fwrite(model->W2, sizeof(float), HIDDEN_UNITS * HIDDEN2_UNITS, f);
        fwrite(model->b2, sizeof(float), HIDDEN2_UNITS, f);
        fwrite(model->W3, sizeof(float), HIDDEN2_UNITS * OUTPUT_SIZE, f);
        fwrite(model->b3, sizeof(float), OUTPUT_SIZE, f);
    } else {
        fwrite(model->W2, sizeof(float), HIDDEN_UNITS * OUTPUT_SIZE, f);
        fwrite(model->b2, sizeof(float), OUTPUT_SIZE, f);
    }
    
    fclose(f);
    printf("Training complete. Saving model to %s\n", filename);
}

void save_model_metadata(const char *model_filename,
                         unsigned int seed,
                         int seed_set,
                         double val_split,
                         int n_train,
                         int n_val) {
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s.meta.txt", model_filename);

    FILE *f = fopen(meta_path, "w");
    if (!f) {
        perror("save_model_metadata");
        return;
    }

    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char ts[64] = "unknown";
    if (tm_info) {
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S %Z", tm_info);
    }

    fprintf(f, "model_path=%s\n", model_filename);
    fprintf(f, "created_at=%s\n", ts);
    fprintf(f, "seed=%u\n", seed);
    fprintf(f, "seed_mode=%s\n", seed_set ? "fixed(--seed)" : "auto(time)");
    fprintf(f, "train_samples=%d\n", n_train);
    fprintf(f, "val_samples=%d\n", n_val);
    fprintf(f, "val_split=%.4f\n", val_split);
    fprintf(f, "iterations=%d\n", NUM_ITERATIONS);
    fprintf(f, "log_every=%d\n", LOG_EVERY);
    fprintf(f, "learning_rate=%.6f\n", LEARNING_RATE);
    fprintf(f, "lr_decay=%.6f\n", LR_DECAY_RATE);
    fprintf(f, "l2_lambda=%.6f\n", g_l2_lambda);
    fprintf(f, "input_size=%d\n", INPUT_SIZE);
    fprintf(f, "hidden_units=%d\n", HIDDEN_UNITS);
    fprintf(f, "hidden2_units=%d\n", HIDDEN2_UNITS);
    fprintf(f, "output_size=%d\n", OUTPUT_SIZE);

    fclose(f);
    printf("Saved training metadata to %s\n", meta_path);
}

FashionModel* load_model_file(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("load_model");
        return NULL;
    }

    ModelHeader header;
    size_t nread = fread(&header, sizeof(ModelHeader), 1, f);
    int has_header = (nread == 1 && header.magic == MODEL_MAGIC);

    if (has_header) {
        if (header.version != MODEL_VERSION ||
            header.input_size != INPUT_SIZE ||
            header.hidden1_units != HIDDEN_UNITS ||
            header.hidden2_units != HIDDEN2_UNITS ||
            header.output_size != OUTPUT_SIZE) {
            fprintf(stderr,
                    "Model architecture mismatch. File has [%u,%u,%u,%u], binary expects [%d,%d,%d,%d].\n",
                    header.input_size, header.hidden1_units, header.hidden2_units, header.output_size,
                    INPUT_SIZE, HIDDEN_UNITS, HIDDEN2_UNITS, OUTPUT_SIZE);
            fclose(f);
            return NULL;
        }
    } else {
        // Backward compatibility: legacy format without header (single hidden layer only).
        if (HIDDEN2_UNITS > 0) {
            fprintf(stderr, "Legacy model format supports only single-hidden architecture.\n");
            fclose(f);
            return NULL;
        }
        rewind(f);
    }
    
    FashionModel *model = init_model();
    
    fread(model->W1, sizeof(float), INPUT_SIZE * HIDDEN_UNITS, f);
    fread(model->b1, sizeof(float), HIDDEN_UNITS, f);

    if (HIDDEN2_UNITS > 0) {
        fread(model->W2, sizeof(float), HIDDEN_UNITS * HIDDEN2_UNITS, f);
        fread(model->b2, sizeof(float), HIDDEN2_UNITS, f);
        fread(model->W3, sizeof(float), HIDDEN2_UNITS * OUTPUT_SIZE, f);
        fread(model->b3, sizeof(float), OUTPUT_SIZE, f);
    } else {
        fread(model->W2, sizeof(float), HIDDEN_UNITS * OUTPUT_SIZE, f);
        fread(model->b2, sizeof(float), OUTPUT_SIZE, f);
    }
    
    fclose(f);
    printf("Model successfully loaded from %s\n", filename);
    return model;
}

// ─────────────────────────────────────────────────────────────
// Testing & Visualization
// ─────────────────────────────────────────────────────────────

void print_model_card(const char *model_path, unsigned int seed, int seed_set, double val_split) {
    int64_t params = (int64_t)INPUT_SIZE * HIDDEN_UNITS + HIDDEN_UNITS;
    if (HIDDEN2_UNITS > 0) {
        params += (int64_t)HIDDEN_UNITS * HIDDEN2_UNITS + HIDDEN2_UNITS;
        params += (int64_t)HIDDEN2_UNITS * OUTPUT_SIZE + OUTPUT_SIZE;
    } else {
        params += (int64_t)HIDDEN_UNITS * OUTPUT_SIZE + OUTPUT_SIZE;
    }

    char arch_buf[64];
    if (HIDDEN2_UNITS > 0) {
        snprintf(arch_buf, sizeof(arch_buf), "784 -> %d -> %d -> 10", HIDDEN_UNITS, HIDDEN2_UNITS);
    } else {
        snprintf(arch_buf, sizeof(arch_buf), "784 -> %d -> 10", HIDDEN_UNITS);
    }

    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           FASHION-MNIST CLASSIFIER — Model Card               ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Architecture:  %-47s║\n", arch_buf);
    printf("║ Parameters:    %-47lld║\n", (long long)params);
    printf("║ Training:      50 iterations, LR=0.0010 (with decay)          ║\n");
    printf("║ Dataset:       Fashion-MNIST (28×28px, 10 clothing classes)   ║\n");
    printf("║ Expected Acc:  ~92-94%% (harder than MNIST digits)             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    printf("Config: model=%s\n", model_path);
    if (seed_set) {
        printf("Config: seed=%u (fixed)\n", seed);
    } else {
        printf("Config: seed=auto (time-based)\n");
    }
    printf("Config: val_split=%.3f\n\n", val_split);
    printf("Config: l2=%.6f\n\n", g_l2_lambda);
}

void test_sample(FashionModel *model, Dataset *dataset, int idx) {
    float *x = &dataset->X[idx * INPUT_SIZE];
    int true_label = (int)dataset->y[idx];
    
    ForwardCache *cache = forward_pass(model, x);
    int pred = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (cache->a3[i] > cache->a3[pred]) pred = i;
    }
    
    printf("Sample %d: True=%s, Predicted=%s (%.1f%%) %s\n",
           idx, fashion_labels[true_label], fashion_labels[pred],
           100.0 * cache->a3[pred],
           pred == true_label ? "✓" : "✗");
    
    free_cache(cache);
}

void confusion_matrix(FashionModel *model, Dataset *dataset, int num_samples) {
    int matrix[OUTPUT_SIZE][OUTPUT_SIZE];
    memset(matrix, 0, sizeof(matrix));
    
    int n = (num_samples < dataset->n_samples) ? num_samples : dataset->n_samples;
    
    for (int i = 0; i < n; i++) {
        int true_label = (int)dataset->y[i];
        int pred = predict(model, &dataset->X[i * INPUT_SIZE]);
        matrix[true_label][pred]++;
    }
    
    printf("\n  Confusion Matrix (rows=true, cols=predicted):\n");
    printf("  true \\ pred");
    for (int j = 0; j < OUTPUT_SIZE; j++) printf(" %4d", j);
    printf("\n");
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  class %d     ", i);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf(" %4d", matrix[i][j]);
        }
        printf("\n");
    }
}

void usage() {
    printf("Fashion-MNIST Classifier\n\n");
    printf("Usage:\n");
    printf("  ./fashion_model [--model <path>]                      Train on fashion_train.csv\n");
    printf("  ./fashion_model [--model <path>] eval <csv>           Evaluate on dataset\n");
    printf("  ./fashion_model [--model <path>] test <csv> [n]       Test n samples\n");
    printf("  ./fashion_model [--model <path>] info                 Show model info\n\n");
    printf("Options:\n");
    printf("  --model <path>     Model file path (default: %s)\n", DEFAULT_MODEL_PATH);
    printf("  --val-split <f>    Holdout fraction for validation during training (e.g. 0.1)\n");
    printf("  --l2 <f>           L2 regularization strength (default: %.6f)\n", DEFAULT_L2_LAMBDA);
    printf("  --seed <n>         Fixed RNG seed for reproducible training\n\n");
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    const char *model_path = DEFAULT_MODEL_PATH;
    double val_split = 0.0;
    unsigned int seed = 0;
    int seed_set = 0;
    int cmd_idx = 1;

    while (cmd_idx < argc && strncmp(argv[cmd_idx], "--", 2) == 0) {
        if (strcmp(argv[cmd_idx], "--model") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --model\n\n");
                usage();
                return 1;
            }
            model_path = argv[cmd_idx + 1];
            cmd_idx += 2;
            continue;
        }

        if (strcmp(argv[cmd_idx], "--val-split") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --val-split\n\n");
                usage();
                return 1;
            }
            val_split = strtod(argv[cmd_idx + 1], NULL);
            if (val_split < 0.0 || val_split >= 1.0) {
                fprintf(stderr, "Invalid --val-split value: %s (expected 0.0 <= f < 1.0)\n\n", argv[cmd_idx + 1]);
                usage();
                return 1;
            }
            cmd_idx += 2;
            continue;
        }

        if (strcmp(argv[cmd_idx], "--seed") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --seed\n\n");
                usage();
                return 1;
            }

            char *endptr = NULL;
            unsigned long parsed = strtoul(argv[cmd_idx + 1], &endptr, 10);
            if (argv[cmd_idx + 1][0] == '\0' || (endptr && *endptr != '\0')) {
                fprintf(stderr, "Invalid --seed value: %s\n\n", argv[cmd_idx + 1]);
                usage();
                return 1;
            }

            seed = (unsigned int)parsed;
            seed_set = 1;
            cmd_idx += 2;
            continue;
        }

        if (strcmp(argv[cmd_idx], "--l2") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Missing value for --l2\n\n");
                usage();
                return 1;
            }

            char *endptr = NULL;
            double parsed_l2 = strtod(argv[cmd_idx + 1], &endptr);
            if (argv[cmd_idx + 1][0] == '\0' || (endptr && *endptr != '\0') || parsed_l2 < 0.0) {
                fprintf(stderr, "Invalid --l2 value: %s (expected non-negative float)\n\n", argv[cmd_idx + 1]);
                usage();
                return 1;
            }

            g_l2_lambda = (float)parsed_l2;
            cmd_idx += 2;
            continue;
        }

        if (strcmp(argv[cmd_idx], "-h") == 0 || strcmp(argv[cmd_idx], "--help") == 0) {
            print_model_card(model_path, seed, seed_set, val_split);
            usage();
            return 0;
        }

        fprintf(stderr, "Unknown option: %s\n\n", argv[cmd_idx]);
        usage();
        return 1;
    }

    if (!seed_set) {
        seed = (unsigned int)time(NULL);
    }
    srand(seed);
    
    if (cmd_idx < argc) {
        if (strcmp(argv[cmd_idx], "info") == 0) {
            print_model_card(model_path, seed, seed_set, val_split);
            return 0;
        }

        if (strcmp(argv[cmd_idx], "eval") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Usage: %s [--model <path>] eval <csv_file>\n", argv[0]);
                return 1;
            }
            FashionModel *model = load_model_file(model_path);
            if (!model) return 1;

            Dataset *dataset = load_dataset(argv[cmd_idx + 1], MAX_SAMPLES);
            if (!dataset) return 1;

            printf("Evaluating model on %d samples from %s...\n", dataset->n_samples, argv[cmd_idx + 1]);
            evaluate(model, dataset);

            free_dataset(dataset);
            free_model(model);
            return 0;
        }

        if (strcmp(argv[cmd_idx], "test") == 0) {
            if (cmd_idx + 1 >= argc) {
                fprintf(stderr, "Usage: %s [--model <path>] test <csv_file> [num_samples]\n", argv[0]);
                return 1;
            }

            FashionModel *model = load_model_file(model_path);
            if (!model) return 1;

            Dataset *dataset = load_dataset(argv[cmd_idx + 1], MAX_SAMPLES);
            if (!dataset) return 1;

            int num_test = (cmd_idx + 2 < argc) ? atoi(argv[cmd_idx + 2]) : 10;
            if (num_test > dataset->n_samples) num_test = dataset->n_samples;

            print_model_card(model_path, seed, seed_set, val_split);

            printf("Testing %d sample(s)...\n\n", num_test);
            for (int i = 0; i < num_test; i++) {
                test_sample(model, dataset, i);
            }

            confusion_matrix(model, dataset, num_test);

            free_dataset(dataset);
            free_model(model);
            return 0;
        }

        fprintf(stderr, "Unknown command: %s\n\n", argv[cmd_idx]);
        usage();
        return 1;
    }
    
    // Default: train
    Dataset *dataset = load_dataset("fashion_train.csv", MAX_SAMPLES);
    if (!dataset) {
        fprintf(stderr, "Cannot load training data. Run: ./setup.sh --data\n");
        return 1;
    }
    
    FashionModel *model = init_model();

    int *all_indices = (int*)malloc(dataset->n_samples * sizeof(int));
    for (int i = 0; i < dataset->n_samples; i++) all_indices[i] = i;
    shuffle_indices(all_indices, dataset->n_samples);

    int n_val = (int)(dataset->n_samples * val_split);
    if (n_val >= dataset->n_samples) n_val = dataset->n_samples - 1;
    int n_train = dataset->n_samples - n_val;

    int *val_indices = all_indices;
    int *train_indices = &all_indices[n_val];

    if (n_train <= 0) {
        fprintf(stderr, "Validation split leaves no training samples. Reduce --val-split.\n");
        free(all_indices);
        free_dataset(dataset);
        free_model(model);
        return 1;
    }

    train_model(model, dataset, train_indices, n_train, val_indices, n_val);
    save_model(model, model_path);
    save_model_metadata(model_path, seed, seed_set, val_split, n_train, n_val);
    
    free(all_indices);
    free_dataset(dataset);
    free_model(model);
    return 0;
}
