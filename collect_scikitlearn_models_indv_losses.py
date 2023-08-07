from joblib import load
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import torch.nn as nn
import torch

# Function to load data
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

# Get a list of all files in the "models" directory with the "joblib" extension
models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

# Iterate over each model file
for model_file in model_files:
    # Load the model from the file
    model_path = os.path.join(models_dir, model_file)
    model = load(model_path)

    # Load the data (You can change this to load different data files for each model)
    data_path = 'data/stainless_steel_304_standardized.xlsx'
    X, y = load_data(data_path)

    # Predict the target for the entire dataset
    if "linear_regression" in model_path:
        
        model = model.cpu()
        X = torch.tensor(X, dtype=torch.float32).cpu()
        with torch.no_grad():
            y_pred = model(X)
        y_pred = y_pred.numpy()
    else:
        y_pred = model.predict(X)

    # Calculate the loss for each data point
    individual_losses = [(y_true - y_pred)**2 for y_true, y_pred in zip(y, y_pred)]

    # Calculate the overall mean squared error
    overall_mse = mean_squared_error(y, y_pred)

    print("Model:", model_file)
    print("Overall Test Loss (MSE):", overall_mse)

    # Create a filename that includes the model name without the "data/" part
    model_name = os.path.basename(model_file).split('.')[0]
    save_name = os.path.join("data", model_name + "_individual_losses.xlsx")

    # Save the individual losses to an Excel file
    df_individual_losses = pd.DataFrame(individual_losses, columns=['Individual Loss'])
    df_individual_losses.to_excel(save_name, index=False)
