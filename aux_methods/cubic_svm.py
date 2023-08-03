import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def k_fold_evaluation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    mse_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = SVR(kernel='poly', degree=3)  # Use polynomial kernel with degree=3 for cubic SVM
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluation
        y_pred = model.predict(X_test)
        mse_score = mean_squared_error(y_test, y_pred)
        print("Fold", fold, "Test Loss (MSE):", mse_score)
        mse_scores.append(mse_score)
    
    avg_kfold_mse = np.mean(mse_scores)
    std_kfold_mse = np.std(mse_scores)
    return avg_kfold_mse, std_kfold_mse, training_time

if __name__ == "__main__":
    filepath = 'data/stainless_steel_304_standardized.xlsx'
    
    X, y = load_data(filepath)
    
    # K-Fold Evaluation
    avg_kfold_mse, std_kfold_mse, _ = k_fold_evaluation(X, y, k=5)
    print("Average K-Fold MSE:", avg_kfold_mse)
    print("Standard Deviation of K-Fold MSE:", std_kfold_mse)

    # Create a new model for the entire dataset
    model = SVR(kernel='poly', degree=3)  # Use polynomial kernel with degree=3 for cubic SVM
    
    # Training on Entire Dataset
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    print("Training Time:", training_time)
    
    # Test Loss on Entire Dataset
    y_pred = model.predict(X)
    test_loss = mean_squared_error(y, y_pred)
    print("Test Loss (MSE) on Entire Dataset:", test_loss)

    # Save Results to Excel in the "data/" directory
    os.makedirs("data", exist_ok=True)
    results = {
        "Average K-Fold MSE": [avg_kfold_mse],
        "Standard Deviation of K-Fold MSE": [std_kfold_mse],
        "Training Time (Seconds)": [training_time],
        "Test Loss (MSE)": [test_loss]
    }
    df_results = pd.DataFrame(results)
    df_results.to_excel("data/cubic_svm_results.xlsx", index=False, float_format="%.6f")
