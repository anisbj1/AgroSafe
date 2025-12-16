/* neural_network.c */
#include "neural_network.h"
#include <string.h>

// ===== ACTIVATION FUNCTIONS =====
float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float* x, int size) {
    float max = x[0];
    float sum = 0.0f;

    // Find max for numerical stability
    for(int i = 1; i < size; i++) {
        if(x[i] > max) max = x[i];
    }

    // Compute exponentials and sum
    for(int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    // Normalize
    for(int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ===== SCALE INPUTS =====
static float scale_value(float value, float mean, float scale) {
    return (value - mean) / scale;
}

// ===== NEURAL NETWORK FORWARD PASS =====
int predict_risk_level(float temperature, float humidity) {
    // Scale inputs
    float scaled_temp = scale_value(temperature, SCALER_MEAN_TEMP, SCALER_SCALE_TEMP);
    float scaled_hum = scale_value(humidity, SCALER_MEAN_HUM, SCALER_SCALE_HUM);

    float inputs[2] = {scaled_temp, scaled_hum};

    // ===== LAYER 1 =====
    float hidden1[8] = {0};

    // Compute: hidden1 = inputs * WEIGHTS_IN_HIDDEN1 + BIAS_HIDDEN1
    for(int j = 0; j < 8; j++) {
        // Start with bias
        hidden1[j] = BIAS_HIDDEN1[j];

        // Add weighted inputs
        for(int i = 0; i < 2; i++) {
            hidden1[j] += inputs[i] * WEIGHTS_IN_HIDDEN1[i][j];
        }

        // Apply ReLU activation
        hidden1[j] = relu(hidden1[j]);
    }

    // ===== LAYER 2 =====
    float hidden2[8] = {0};

    // Compute: hidden2 = hidden1 * WEIGHTS_HIDDEN1_HIDDEN2 + BIAS_HIDDEN2
    for(int j = 0; j < 8; j++) {
        // Start with bias
        hidden2[j] = BIAS_HIDDEN2[j];

        // Add weighted activations from previous layer
        for(int i = 0; i < 8; i++) {
            hidden2[j] += hidden1[i] * WEIGHTS_HIDDEN1_HIDDEN2[i][j];
        }

        // Apply ReLU activation
        hidden2[j] = relu(hidden2[j]);
    }

    // ===== OUTPUT LAYER =====
    float output[3] = {0};

    // Compute: output = hidden2 * WEIGHTS_HIDDEN2_OUTPUT + BIAS_OUTPUT
    for(int j = 0; j < 3; j++) {
        // Start with bias
        output[j] = BIAS_OUTPUT[j];

        // Add weighted activations
        for(int i = 0; i < 8; i++) {
            output[j] += hidden2[i] * WEIGHTS_HIDDEN2_OUTPUT[i][j];
        }
    }

    // Apply softmax to get probabilities
    softmax(output, 3);

    // Find class with highest probability
    int max_idx = 0;
    for(int i = 1; i < 3; i++) {
        if(output[i] > output[max_idx]) {
            max_idx = i;
        }
    }

    return max_idx;  // 0 = safe, 1 = warning, 2 = danger
}

// ===== RISK LEVEL DESCRIPTIONS =====
const char* get_risk_string(int risk_level) {
    switch(risk_level) {
        case 0: return "Faible";
        case 1: return "Moyen";
        case 2: return "Eleve";
        default: return "Inconnu";
    }
}
