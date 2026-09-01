# Week 4: Deep Learning Foundations - Neural Networks from Scratch vs PyTorch

## Overview
This repository contains the implementation of a Feed-Forward Neural Network built from scratch using **NumPy** alongside a framework implementation in **PyTorch**, evaluated on the Credit Card Fraud Detection dataset.

## Comparative Performance Summary

| Model / Experiment | Framework / Method | Optimizer | ROC-AUC Score | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **NumPy Implementation** | First Principles | Vanilla SGD | 0.8268 | 8.10s |
| **PyTorch Baseline** | PyTorch (`nn.Module`) | Vanilla SGD | 0.8230 | 18.99s |
| **Experiment 1 (Deeper Net)** | PyTorch (2 Hidden Layers) | Vanilla SGD | 0.8572 | ~20.00s |
| **Experiment 2 (Adam Opt)** | PyTorch (64 Hidden Nodes) | Adam | **0.9734** | ~19.50s |

## Key Insights & Experiment Observations
1. **Mathematical Correctness:** The NumPy model achieved an identical ROC-AUC (~0.82) to the baseline PyTorch SGD model, validating the mathematical implementation of backpropagation and cross-entropy loss gradients.
2. **Architecture Impact:** Increasing network depth (adding a second hidden layer) improved representation learning, boosting the ROC-AUC score from 0.8230 to 0.8572.
3. **Optimizer Efficiency:** Switching from standard SGD to adaptive momentum (**Adam**) dramatically enhanced convergence on the highly imbalanced fraud dataset, improving ROC-AUC to **0.9734**[cite: 2].
