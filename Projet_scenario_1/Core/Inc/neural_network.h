/* neural_network.h */
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>
#include <stdint.h>

// ===== SCALER PARAMETERS (Replace with your actual values!) =====
// These come from your Python script output
#define SCALER_MEAN_TEMP 33.801724f     // Replace with actual mean_temp
#define SCALER_MEAN_HUM  54.493103f     // Replace with actual mean_hum
#define SCALER_SCALE_TEMP 7.710517f    // Replace with actual scale_temp
#define SCALER_SCALE_HUM  19.206329f    // Replace with actual scale_hum

// ===== NEURAL NETWORK WEIGHTS =====
// Replace these with your actual weights from Python output!

// Layer 1 weights (input -> hidden1, 2x8)
extern const float WEIGHTS_IN_HIDDEN1[2][8];
// Layer 1 biases (8)
extern const float BIAS_HIDDEN1[8];

// Layer 2 weights (hidden1 -> hidden2, 8x8)
extern const float WEIGHTS_HIDDEN1_HIDDEN2[8][8];

// Layer 2 biases (8)
extern const float BIAS_HIDDEN2[8];

// Output layer weights (hidden2 -> output, 8x3)
extern const float WEIGHTS_HIDDEN2_OUTPUT[8][3];
// Output biases (3)
extern const float BIAS_OUTPUT[3];

// ===== FUNCTION DECLARATIONS =====
float relu(float x);
void softmax(float* x, int size);
int predict_risk_level(float temperature, float humidity);
const char* get_risk_string(int risk_level);

#endif // NEURAL_NETWORK_H
