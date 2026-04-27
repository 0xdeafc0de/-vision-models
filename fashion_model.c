#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

#define INPUT_SIZE 784
#define HIDDEN_UNITS 32
#define OUTPUT_SIZE 10
#define MAX_SAMPLES 70000
#define NUM_ITERATIONS 50
#define LEARNING_RATE 0.001
#define LR_DECAY_RATE 1e-5
#define L2_LAMBDA 0.0

typedef struct {
    float *W1, *b1;  // Input → Hidden
    float *W2, *b2;  // Hidden → Output
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
    
    model->W2 = (float*)malloc(HIDDEN_UNITS * OUTPUT_SIZE * sizeof(float));
    model->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    
    // He initialization for ReLU hidden layer
    he_init(model->W1, INPUT_SIZE, HIDDEN_UNITS);
    
    // Xavier for softmax output
    xavier_init(model->W2, HIDDEN_UNITS, OUTPUT_SIZE);
    
    return model;
}

void free_model(FashionModel *model) {
    if (model) {
        free(model->W1);
        free(model->b1);
        free(model->W2);
        free(model->b2);
        free(model);
    }
}

// ─────────────────────────────────────────────────────────────
// Forward Pass
// ─────────────────────────────────────────────────────────────

typedef struct {
    float *z1, *a1;  // Hidden layer
    float *z2, *a2;  // Output layer
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
    
    cache->z2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    cache->a2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Output layer: Softmax
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        cache->z2[j] = model->b2[j];
        for (int i = 0; i < HIDDEN_UNITS; i++) {
            cache->z2[j] += model->W2[i * OUTPUT_SIZE + j] * cache->a1[i];
        }
        cache->a2[j] = cache->z2[j];
    }
    softmax(cache->a2, OUTPUT_SIZE);
    
    return cache;
}

void free_cache(ForwardCache *cache) {
    if (cache) {
        free(cache->z1);
        free(cache->a1);
        free(cache->z2);
        free(cache->a2);
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
        dz2[i] = cache->a2[i];
        if (i == true_label) dz2[i] -= 1;
    }
    
    // Update W2 and b2
    for (int i = 0; i < HIDDEN_UNITS; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float grad = cache->a1[i] * dz2[j] + L2_LAMBDA * model->W2[i * OUTPUT_SIZE + j];
            model->W2[i * OUTPUT_SIZE + j] -= lr * grad;
        }
    }
    
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        model->b2[j] -= lr * dz2[j];
    }
    
    // Hidden layer gradients
    float dz1[HIDDEN_UNITS];
    for (int i = 0; i < HIDDEN_UNITS; i++) {
        float grad_sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad_sum += model->W2[i * OUTPUT_SIZE + j] * dz2[j];
        }
        dz1[i] = grad_sum * relu_derivative(cache->z1[i]);
    }
    
    // Update W1 and b1
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_UNITS; j++) {
            float grad = x[i] * dz1[j] + L2_LAMBDA * model->W1[i * HIDDEN_UNITS + j];
            model->W1[i * HIDDEN_UNITS + j] -= lr * grad;
        }
    }
    
    for (int j = 0; j < HIDDEN_UNITS; j++) {
        model->b1[j] -= lr * dz1[j];
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

void train_model(FashionModel *model, Dataset *dataset) {
    int *indices = (int*)malloc(dataset->n_samples * sizeof(int));
    for (int i = 0; i < dataset->n_samples; i++) indices[i] = i;
    
    printf("Training on %d samples.... Number of iteration = %d\n", dataset->n_samples, NUM_ITERATIONS);
    
    for (int epoch = 0; epoch < NUM_ITERATIONS; epoch++) {
        float lr = LEARNING_RATE / (1.0 + LR_DECAY_RATE * epoch);
        
        shuffle_indices(indices, dataset->n_samples);
        
        for (int i = 0; i < dataset->n_samples; i++) {
            int idx = indices[i];
            float *x = &dataset->X[idx * INPUT_SIZE];
            int label = (int)dataset->y[idx];
            
            ForwardCache *cache = forward_pass(model, x);
            
            backward_pass(model, cache, x, label, lr);
            free_cache(cache);
        }
        
        if ((epoch + 1) % 10 == 0) {
            printf("Iteration....%d (lr=%.6f)\n", epoch, lr);
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
        if (cache->a2[i] > cache->a2[best]) best = i;
    }
    free_cache(cache);
    return best;
}

void evaluate(FashionModel *model, Dataset *dataset) {
    int correct = 0;
    int per_class_correct[OUTPUT_SIZE] = {0};
    int per_class_total[OUTPUT_SIZE] = {0};
    
    for (int i = 0; i < dataset->n_samples; i++) {
        float *x = &dataset->X[i * INPUT_SIZE];
        int label = (int)dataset->y[i];
        int pred = predict(model, x);
        
        if (pred == label) correct++;
        per_class_correct[label] += (pred == label);
        per_class_total[label]++;
    }
    
    printf("\nOverall accuracy: %d / %d = %.2f%%\n\n", correct, dataset->n_samples,
           100.0 * correct / dataset->n_samples);
    
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

void save_model(FashionModel *model, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("save_model");
        return;
    }
    
    fwrite(model->W1, sizeof(float), INPUT_SIZE * HIDDEN_UNITS, f);
    fwrite(model->b1, sizeof(float), HIDDEN_UNITS, f);
    fwrite(model->W2, sizeof(float), HIDDEN_UNITS * OUTPUT_SIZE, f);
    fwrite(model->b2, sizeof(float), OUTPUT_SIZE, f);
    
    fclose(f);
    printf("Training complete. Saving model to %s\n", filename);
}

FashionModel* load_model_file(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("load_model");
        return NULL;
    }
    
    FashionModel *model = init_model();
    
    fread(model->W1, sizeof(float), INPUT_SIZE * HIDDEN_UNITS, f);
    fread(model->b1, sizeof(float), HIDDEN_UNITS, f);
    fread(model->W2, sizeof(float), HIDDEN_UNITS * OUTPUT_SIZE, f);
    fread(model->b2, sizeof(float), OUTPUT_SIZE, f);
    
    fclose(f);
    printf("Model successfully loaded from %s\n", filename);
    return model;
}

// ─────────────────────────────────────────────────────────────
// Testing & Visualization
// ─────────────────────────────────────────────────────────────

void print_model_card() {
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           FASHION-MNIST CLASSIFIER — Model Card               ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Architecture:  784 → 32 → 10  (Fully Connected, ReLU+Softmax) ║\n");
    printf("║ Parameters:    25450 total  (W1:25088 b1:32 | W2:320 b2:10)   ║\n");
    printf("║ Training:      50 iterations, LR=0.0010 (with decay)          ║\n");
    printf("║ Dataset:       Fashion-MNIST (28×28px, 10 clothing classes)   ║\n");
    printf("║ Expected Acc:  ~92-94%% (harder than MNIST digits)             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
}

void test_sample(FashionModel *model, Dataset *dataset, int idx) {
    float *x = &dataset->X[idx * INPUT_SIZE];
    int true_label = (int)dataset->y[idx];
    
    ForwardCache *cache = forward_pass(model, x);
    int pred = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (cache->a2[i] > cache->a2[pred]) pred = i;
    }
    
    printf("Sample %d: True=%s, Predicted=%s (%.1f%%) %s\n",
           idx, fashion_labels[true_label], fashion_labels[pred],
           100.0 * cache->a2[pred],
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
    printf("  ./fashion_model                      Train on fashion_train.csv\n");
    printf("  ./fashion_model eval <csv>           Evaluate on dataset\n");
    printf("  ./fashion_model test <csv> [n]       Test n samples\n");
    printf("  ./fashion_model info                 Show model info\n\n");
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        print_model_card();
        usage();
        return 0;
    }
    
    if (strcmp(argv[1], "info") == 0) {
        print_model_card();
        return 0;
    }
    
    if (strcmp(argv[1], "eval") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s eval <csv_file>\n", argv[0]);
            return 1;
        }
        FashionModel *model = load_model_file("model.bin");
        if (!model) return 1;
        
        Dataset *dataset = load_dataset(argv[2], MAX_SAMPLES);
        if (!dataset) return 1;
        
        printf("Evaluating model on %d samples from %s...\n", dataset->n_samples, argv[2]);
        evaluate(model, dataset);
        
        free_dataset(dataset);
        free_model(model);
        return 0;
    }
    
    if (strcmp(argv[1], "test") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s test <csv_file> [num_samples]\n", argv[0]);
            return 1;
        }
        
        FashionModel *model = load_model_file("model.bin");
        if (!model) return 1;
        
        Dataset *dataset = load_dataset(argv[2], MAX_SAMPLES);
        if (!dataset) return 1;
        
        int num_test = (argc > 3) ? atoi(argv[3]) : 10;
        if (num_test > dataset->n_samples) num_test = dataset->n_samples;
        
        print_model_card();
        
        printf("Testing %d sample(s)...\n\n", num_test);
        for (int i = 0; i < num_test; i++) {
            test_sample(model, dataset, i);
        }
        
        confusion_matrix(model, dataset, num_test);
        
        free_dataset(dataset);
        free_model(model);
        return 0;
    }
    
    // Default: train
    Dataset *dataset = load_dataset("fashion_train.csv", MAX_SAMPLES);
    if (!dataset) {
        fprintf(stderr, "Cannot load training data. Run: ./setup.sh --data\n");
        return 1;
    }
    
    FashionModel *model = init_model();
    train_model(model, dataset);
    save_model(model, "model.bin");
    
    free_dataset(dataset);
    free_model(model);
    return 0;
}
