# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network to predict **customer churn** — whether a customer is likely to leave a service or stay. The model demonstrates how neural networks learn through forward pass, backpropagation, and parameter updates.

## Dataset
- **File:** `customer_churn_nn.csv`
- **Source:** [Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Target Column:** `churn` (1 = churned, 0 = retained)
- **Features:** tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, referrals, region, plan_type, contract_type, payment_method

## Steps Performed
1. **Dataset Understanding** — Explored shape, types, missing values, and class distribution
2. **Data Preprocessing** — Dropped ID, encoded categoricals, scaled numericals, train-test split
3. **Neural Network Model** — Built a feed-forward network with 2 hidden layers using ReLU
4. **Training and Evaluation** — Trained for 30 epochs, evaluated with accuracy and confusion matrix
5. **Hyperparameter Experiments** — Ran 4 configurations, compared results in a table
6. **Reflection** — Analyzed weights, activation functions, learning rate effects, and overfitting

## Results
| Metric | Value |
|--------|-------|
| Training Accuracy | See notebook |
| Testing Accuracy | See notebook |
| Evaluation | Confusion matrix + classification report |

## Model Architecture
```
Input Layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** Binary classification (churn or not)

## Hyperparameter Comparison Table
See `results/model_comparison_table.csv` for full experiment comparison.

## Reflection

### What role do weights and biases play?
Weights control how strongly each input feature influences the prediction. Biases allow the model to shift the activation, helping the neuron fire even when inputs are zero. Together, they are the "parameters" the model learns during training.

### Why is an activation function required?
Without activation functions, a neural network is just a linear regression — no matter how many layers you add. Activation functions like ReLU introduce non-linearity, allowing the network to learn complex patterns.

### What happens when learning rate is too high or too low?
- **Too high:** The model overshoots the optimal solution — loss bounces or diverges
- **Too low:** Training becomes very slow — the model takes too long to converge

### Underfitting vs Overfitting
- **Underfitting:** If training accuracy is also low — the model is too simple
- **Overfitting:** If training accuracy >> testing accuracy — the model memorized training data
- See notebook for actual observations with this dataset.

## How to Run
1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `customer_churn_nn.csv` from the dataset folder
3. Run all cells from top to bottom

## Requirements
See `requirements.txt` for all dependencies.
