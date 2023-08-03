import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from mlp import MLP
import os
import itertools
from tqdm import tqdm

# Load the standardized data
def load_standardized_data(filename):
    data = pd.read_excel(filename)
    return torch.tensor(data.values).float()

def load_standardization_values(filename):
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(filename)

    # Filter the rows based on "mean" and "std" values in the "Unnamed: 0" column
    mean_values = df[df['Unnamed: 0'] == 'mean']
    std_values = df[df['Unnamed: 0'] == 'std']

    # Drop the 'Unnamed: 0' column as it's not needed anymore
    mean_values = mean_values.drop(columns=['Unnamed: 0'])
    std_values = std_values.drop(columns=['Unnamed: 0'])

    # If you want to get the values as variables:
    mean_values = mean_values.values.flatten()
    std_values = std_values.values.flatten()
    
    return mean_values, std_values

# Function to unstandardize the input combinations based on mean and std values
def unstandardize_input_combinations(input_combinations, mean_values, std_values):
    return (input_combinations * std_values) + mean_values

# Unstandardize the predictions based on the mean and std values
def unstandardize_predictions(predictions, mean_values, std_values):
    return ((predictions * std_values) + mean_values).squeeze()

# Function to generate equally spaced data points for each column based on min and max range
def generate_equally_spaced_data_points(min_val, max_val, num_points=100):
    return np.linspace(min_val, max_val, num_points)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the path to your data files
    model_name = "models/leitner_model_16_hidden_layers_kfold_loss_0_0008.pt"
    hidden_size = 16
    data_file = "data/stainless_steel_304.xlsx"
    standardized_data_file = "data/stainless_steel_304_standardized.xlsx"
    standardization_values_file = "data/stainless_steel_304_standardization_values.xlsx"

    model_name_without_extension = os.path.splitext(os.path.basename(model_name))[0]
    data_file_without_extension = os.path.splitext(os.path.basename(data_file))[0]

    # Load the standardized data
    standardized_data = load_standardized_data(standardized_data_file)

    # Load the standardization values (mean and std) for each column
    mean_values, std_values = load_standardization_values(standardization_values_file)
    
    # Load the neural network model
    model = MLP(hidden_size=hidden_size)

    # Load the trained model state dict
    model_state_dict = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)

    # Set the device for prediction (CPU or GPU)
    model.to(device)

    # Extract the minimum and maximum values for each column from the standardized data
    min_vals = standardized_data.min(dim=0)[0][:-1]  # Exclude the last column (Stress) which is the target
    max_vals = standardized_data.max(dim=0)[0][:-1]  # Exclude the last column (Stress) which is the target

    # Generate equally spaced data points for each column based on min and max range
    num_points_per_column = 10
    input_combinations = itertools.product(*[
        generate_equally_spaced_data_points(min_val, max_val, num_points_per_column)
        for min_val, max_val in zip(min_vals, max_vals)
    ])

    # Convert input_combinations to a list of NumPy arrays and stack them vertically
    input_combinations = np.vstack(list(input_combinations))

    # Convert the NumPy array to a PyTorch tensor
    input_combinations_tensor = torch.tensor(input_combinations).float().to(device)

    # Unstandardize the input combinations
    unstandardized_input_combinations = unstandardize_input_combinations(input_combinations, mean_values[:-1], std_values[:-1])

    # Convert the NumPy array to a PyTorch tensor
    unstandardized_input_combinations_tensor = torch.tensor(unstandardized_input_combinations).float()

    # Create a DataLoader to iterate over batches
    batch_size = int(2**8)
    data_loader = DataLoader(TensorDataset(input_combinations_tensor), batch_size=batch_size, shuffle=False)

    # Make predictions for each batch of input combinations
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches", leave=True):
            batch_inputs = batch[0].to(device)
            batch_predictions = model(batch_inputs).cpu()
            all_predictions.append(batch_predictions)

    # Concatenate all the predictions
    all_predictions = torch.cat(all_predictions, dim=0)

    # Unstandardize the predictions
    exhaustive_predictions = unstandardize_predictions(all_predictions, mean_values[-1], std_values[-1])

    # Create a DataFrame to store the results
    exhaustive_results_df = pd.DataFrame(unstandardized_input_combinations_tensor, columns=["Temp", "Strain", "Strain Rate"])
    exhaustive_results_df["Predictions"] = exhaustive_predictions.numpy()

    # Save the exhaustive predictions to an Excel file
    exhaustive_results_df.to_excel("data/" + model_name_without_extension + "_" + data_file_without_extension + "_exhaustive_predictions_" + str(num_points_per_column) + ".xlsx", index=False)
