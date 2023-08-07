import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add parent directory to the system path
sys.path.append(os.path.abspath('..'))  # Assuming the parent directory contains the 'mlp.py' file

from mlp import MLP  # Now you can import the MLP class

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hypers
    hidden_layer = 16

    # Load data
    data_path = 'data/stainless_steel_304_standardized.xlsx'
    X, y = load_data(data_path)
    test_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    
    # List of model paths
    model_paths = ['models/leitner_model_16_hidden_layers_5000_epochs_kfold_loss_0_0001.pt',
                   'models/model_16_hidden_layers_5000_epochs_kfold_loss_0_000375.pt']
    
    for model_path in model_paths:
        # Instantiate the model
        model = MLP(hidden_size=hidden_layer).to(device)
        
        # Load the trained model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Collect individual losses for the test data
        individual_losses = []
        dataloader = DataLoader(test_data, batch_size=1)
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch, (X_batch, y_batch) in enumerate(dataloader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(-1))
                individual_losses.append(loss.cpu().numpy())

        # Save the individual losses to an Excel file
        model_name = os.path.basename(model_path).split('.')[0]
        save_name = f'data/{model_name}_individual_losses.xlsx'
        df_individual_losses = pd.DataFrame(individual_losses, columns=['Individual Loss'])
        df_individual_losses.to_excel(save_name, index=False)
