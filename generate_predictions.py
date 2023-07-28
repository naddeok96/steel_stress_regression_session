import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from mlp import MLP
import os

# Load the standardized data
def load_standardized_data(filename):
    data = pd.read_excel(filename)
    return torch.tensor(data.values).float()

def load_standardization_values(filename):
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(filename, index_col=0)

    # Extract the rows with "mean" and "std" from the "Stress" column
    stress_column = df['Stress']

    # Extract the corresponding values
    mean_value = stress_column["mean"]
    std_value = stress_column["std"]
    
    return mean_value, std_value

# Unstandardize the predictions based on the Stress column's mean and std
def unstandardize_predictions(predictions, mean_values, std_values):
    return ((predictions * std_values) + mean_values).squeeze()

if __name__ == "__main__":
    # Set the path to your data files
    model_name = "models/model_8.00_hidden_layers_kfold_loss_0.0007.pt"
    hidden_size=8
    data_file = "data/stainless_steel_304.xlsx"
    standardized_data_file = "data/stainless_steel_304_standardized.xlsx"
    standardization_values_file = "data/stainless_steel_304_standardization_values.xlsx"

    model_name_without_extension = os.path.splitext(os.path.basename(model_name))[0]
    data_file_without_extension = os.path.splitext(os.path.basename(data_file))[0]

    # Load the standardized data
    standardized_data = load_standardized_data(standardized_data_file)

    # Load the standardization values (mean and std)
    mean_values, std_values = load_standardization_values(standardization_values_file)
    
    # Load the neural network model
    model = MLP(hidden_size=hidden_size)

    # Load the trained model state dict
    model_state_dict = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)

    # Set the device for prediction (CPU or GPU)
    device = torch.device("cpu")
    model.to(device)

    # Prepare the input data for prediction
    X = standardized_data[:, :-1].to(device)

    # Predict using the model
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu()

    # Unstandardize the predictions
    unstandardized_predictions = unstandardize_predictions(predictions, mean_values, std_values)

    # Calculate the error (residual) per row
    y_true = torch.tensor(pd.read_excel(data_file, index_col=0)["Stress"].values).float()  # Load the true target values

    # Make sure y_true and unstandardized_predictions have the same shape
    assert y_true.shape == unstandardized_predictions.shape, "Shapes of y_true and predictions don't match"

    error_per_row = y_true - unstandardized_predictions.squeeze()

    # Calculate absolute error per row
    abs_error_per_row = torch.abs(error_per_row)

    # Calculate mean absolute error
    mean_abs_error = torch.mean(abs_error_per_row)

    # Calculate standard deviation of absolute error
    std_abs_error = torch.std(abs_error_per_row)

    # Determine outliers based on quantiles
    q1 = torch.quantile(abs_error_per_row, 0.25)
    q3 = torch.quantile(abs_error_per_row, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((abs_error_per_row < lower_bound) | (abs_error_per_row > upper_bound))

    # Calculate number of outliers
    num_outliers = torch.sum(outliers)

    # Save the results with additional columns for unstandardized predictions and error per row
    result_df = pd.read_excel(data_file)
    result_df["Unstandardized Prediction"] = unstandardized_predictions.numpy()
    result_df["Error per Row"] = error_per_row.numpy()
    result_df["Absolute Error per Row"] = abs_error_per_row.numpy()

    # Create a DataFrame for the summary statistics
    summary_df = pd.DataFrame({
        "Mean Absolute Error": [mean_abs_error.item()],
        "Std Absolute Error": [std_abs_error.item()],
        "Number of Outliers": [num_outliers.item()]
    })

    # Add the summary stats to the result DataFrame
    result_df = pd.concat([result_df, summary_df], axis=1)

    result_df.to_excel("data/" + model_name_without_extension + "_" + data_file_without_extension + ".xlsx", index=False)
