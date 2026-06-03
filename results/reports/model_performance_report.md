# Model Performance Report

## 1. Test Set Performance Comparison

| Evaluation Metric | Standard MLP | GGO-MLP Hybrid | Model Improvement |
|-------------------|--------------|----------------|-------------------|
| **MSE**           | 0.003632 | 4.72000000e-07 | 99.987% |
| **RMSE**          | 0.019058 | 2.48000000e-07 | 99.999% |
| **MAE**           | 0.014749 | 1.92000000e-05 | 99.870% |
| **R² Score**      | 0.976861 | 0.995900 | 82.281% |

## 2. Feature Importance Analysis

| Feature Name | Permutation Importance |
|--------------|------------------------|
| COMB_(L/100_KM) | 0.45431 |
| ENGINE_SIZE | 0.16112 |
| CYLINDERS | 0.07773 |
