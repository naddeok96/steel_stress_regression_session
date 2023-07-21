import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from mlp import MLP

# Load the standardized data
def load_standardized_data(filename):
    data = pd.read_excel(filename, index_col=0)
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
    data_file = "data/stainless_steel_304.xlsx"
    standardized_data_file = "data/stainless_steel_304_standardized.xlsx"
    standardization_values_file = "data/stainless_steel_304_standardization_values.xlsx"

    # Load the standardized data
    standardized_data = load_standardized_data(standardized_data_file)

    # Load the standardization values (mean and std)
    mean_values, std_values = load_standardization_values(standardization_values_file)
    
    # Load the neural network model
    model = MLP()

    # Load the trained model state dict
    model_state_dict = torch.load("models/trained_model.pt", map_location=torch.device('cpu'))
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

    # Save the results with additional columns for unstandardized predictions and error per row
    result_df = pd.read_excel(data_file, index_col=0)
    result_df["Unstandardized Prediction"] = unstandardized_predictions.numpy()
    result_df["Error per Row"] = error_per_row.numpy()
    result_df.to_excel("data/stainless_steel_304_with_predictions.xlsx", index=False)
