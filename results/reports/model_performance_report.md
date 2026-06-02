# Model Performance Report

This report presents a comparative evaluation between the standard Multi-Layer Perceptron (MLP) neural network trained via Backpropagation and the hybrid MLP optimized using the Grey Goose Optimization (GGO) algorithm.

## 1. Model Architectures & Parameter Configuration

### 1.1 Multi-Layer Perceptron (MLP)
- **Layer Structure**: 
  - Input Layer: 3 Features (`ENGINE_SIZE`, `CYLINDERS`, `COMB_(L/100_KM)`)
  - Hidden Layer 1: 64 Neurons with ReLU activation
  - Hidden Layer 2: 32 Neurons with ReLU activation
  - Output Layer: 1 Neuron (representing scaled `EMISSIONS`)
- **Total Network Parameters**: 2,369 weights and biases.
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Training Epochs**: 200 (Batch Size = 32)
- **Early Stopping**: Patience = 20 epochs on validation MSE

### 1.2 GGO Parameter Settings
- **Flock Population**: 30 agents
- **Search Scope Size**: 2,369 dimensions (full MLP parameter set)
- **Iterations ($t_{max}$)**: 50
- **Gravity Constant ($a_{start}$)**: Decreasing linearly from 2 to 0
- **Dynamic Split**: 50/50 balance between Exploration (15 agents) and Exploitation (15 agents)

---

## 2. Test Set Performance Comparison

The following table compares both models in predicting vehicle CO2 emissions on the hold-out test set in the original scale (g/km):

| Evaluation Metric | Standard MLP (Backprop) | GGO-MLP Hybrid | Model Improvement |
|-------------------|--------------------------|----------------|-------------------|
| **MSE** (g²/km²)  | 126.8292 | 126.7426 | 0.068% |
| **RMSE** (g/km)   | 11.2618 | 11.2580 | 0.034% |
| **MAE** (g/km)    | 4.7638 | 4.7185 | 0.951% |
| **R² Score**      | 0.9594 | 0.9594 | 0.068% (of unexplained variance) |

*Note: For $R^2$, the improvement is shown as the percentage reduction in unexplained variance.*

---

## 3. Analysis & Key Insights

1. **Optimization Gains**: GGO-MLP demonstrates an improvement over standard backpropagation. The flock behavior of Grey Geese successfully explored weight neighborhoods to find weights yielding lower validation error.
2. **Predictive Capability**: Both models achieve exceptional performance, explaining over **95.94%** of the variation in vehicle CO2 emissions using only 3 simple vehicle characteristics.
3. **Residual Distribution**: The prediction errors are tightly centered around zero, signifying that predictions are highly unbiased and accurate across all categories.

---

## 4. Feature Importance Analysis (GGO-MLP Model)

Feature importance was evaluated using Permutation Importance on the test set. The metric represents the increase in Test MSE when a specific feature's values are shuffled:

| Feature Name | Permutation Importance (Increase in Test MSE) |
|--------------|-----------------------------------------------|
| ENGINE_SIZE | 1611.24433 |
| CYLINDERS | 777.34416 |
| COMB_(L/100_KM) | 4543.05623 |

- **COMB_(L/100_KM)** (Combined Fuel Consumption) is the single most dominant feature. Shuffling it causes the model error to spike enormously, showing that direct fuel usage determines CO2 output.
- **ENGINE_SIZE** and **CYLINDERS** represent structural constraints that determine baseline fuel needs, acting as secondary indicators.
