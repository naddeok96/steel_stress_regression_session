import pandas as pd
import torch

def open_data(filename):
    # Open data and return it 
    data = pd.read_excel(filename)
    return data

def clean_data(data):
    # check for any values that are not numbers
    tensor_data = torch.tensor(data.values).float()  # convert DataFrame to PyTorch Tensor
    nan_locations = torch.isnan(tensor_data)
    if nan_locations.any():
        print("NaN values found at:")
        rows, cols = torch.where(nan_locations)
        for row, col in zip(rows, cols):
            print(f"Row {row.item()}, Column {col.item()}")
    tensor_data[nan_locations] = 0  # replace NaNs with 0
    return tensor_data

def calculate_standardization_values(data, filename, headers):
    # calculate mean and std of each col and save in an xlsx and print save name and return values
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)
    mean_std_values = torch.stack([mean_values, std_values])
    df = pd.DataFrame(mean_std_values.numpy(), columns=headers, index=['mean', 'std'])
    df.to_excel(filename.replace('.xlsx', '_standardization_values.xlsx'))  # save with filename reference
    print(f"Standardization values saved as {filename.replace('.xlsx', '_standardization_values.xlsx')}")
    return mean_values, std_values

def standardize_data(data, mean_values, std_values):
    # standardize each column by subtracting mean and dividing by std
    standardized_data = (data - mean_values) / std_values
    return standardized_data

def save_data(data, filename):
    # save as xlsx in same folder
    pd.DataFrame(data.numpy()).to_excel(filename)

def process_data(filename, standardization_values=None):
    df = open_data(filename)
    data = clean_data(df)
    save_data(data, filename.replace('.xlsx', '_cleaned.xlsx'))

    if standardization_values is None:
        mean_values, std_values = calculate_standardization_values(data, filename, df.columns)
    else:
        mean_values, std_values = standardization_values

    standardized_data = standardize_data(data, mean_values, std_values)
    save_data(standardized_data, filename.replace('.xlsx', '_standardized.xlsx'))

    return standardized_data

if "__main__" == __name__:
    process_data("data/stainless_steel_304.xlsx", standardization_values=None)
