import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def huber_loss(y_true, y_pred, delta=1.0):
    residual = torch.abs(y_true - y_pred)
    loss = torch.where(residual < delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))
    return torch.mean(loss)  # Take the mean of the loss across the mini-batch

def k_fold_evaluation(X, y, num_epochs, k=5, device=torch.device("cpu")):
    kf = KFold(n_splits=k, shuffle=True)
    mse_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LinearRegressionModel(input_dim=X_train.shape[1])
        criterion = huber_loss
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        
        # Move model and data to GPU if available
        model.to(device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # Training
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_tensor)
            loss = criterion(y_train_tensor.unsqueeze(1), y_pred)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            mse_score = mean_squared_error(y_test, y_pred.cpu().numpy())
            print("Fold", fold, "Test Loss (MSE):", mse_score)
            mse_scores.append(mse_score)
    
    avg_kfold_mse = np.mean(mse_scores)
    std_kfold_mse = np.std(mse_scores)
    return avg_kfold_mse, std_kfold_mse

def train_on_entire_dataset(model, X, y, num_epochs, device=torch.device("cpu")):
    criterion = huber_loss
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Move model and data to GPU if available
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Training
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_tensor.unsqueeze(1), y_pred)
        loss.backward()
        optimizer.step()
    
    training_time = time.time() - start_time
    return model, training_time

if __name__ == "__main__":
    # GPU Setup
    gpu_number = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    filepath = 'data/stainless_steel_304_standardized.xlsx'
    num_epochs = 5000
    
    X, y = load_data(filepath)
    
    # K-Fold Evaluation
    avg_kfold_mse, std_kfold_mse = k_fold_evaluation(X, y, num_epochs, k=5, device=device)
    print("Average K-Fold MSE:", avg_kfold_mse)
    print("Standard Deviation of K-Fold MSE:", std_kfold_mse)
    
    # Create a new model for training on the entire dataset
    model = LinearRegressionModel(input_dim=X.shape[1])
    
    # Training on Entire Dataset
    model, training_time = train_on_entire_dataset(model, X, y, num_epochs, device=device)
    print("Training Time:", training_time)
    
    # Test Loss on Entire Dataset
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X_tensor)
        test_loss = mean_squared_error(y, y_pred.cpu().numpy())
    print("Test Loss (MSE) on Entire Dataset:", test_loss)

    # Save Results to Excel
    results = {
        "Average K-Fold MSE": [avg_kfold_mse],
        "Standard Deviation of K-Fold MSE": [std_kfold_mse],
        "Training Time": [training_time],
        "Test Loss (MSE)": [test_loss]
    }
    df_results = pd.DataFrame(results)
    df_results.to_excel("data/robust_linear_regression_results.xlsx", index=False, float_format="%.6f")
