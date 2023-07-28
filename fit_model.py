import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from mlp import MLP  # Make sure mlp.py is in the same directory

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def train_model(model, dataloader, epochs, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def test_model(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(-1).to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            print("Test Loss ", total_loss)
            
    avg_test_loss = total_loss / len(dataloader)
    print(f'Avg Test Loss: {avg_test_loss}')
    return total_loss / len(dataloader)

def k_fold_validation(model, X, y, epochs, criterion, optimizer, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    losses = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        train_dataloader = DataLoader(train_data, batch_size=64)
        test_dataloader = DataLoader(test_data, batch_size=64)

        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_dataloader, epochs, criterion, optimizer)
        loss = test_model(model, criterion, test_dataloader)
        losses.append(loss)

    return sum(losses) / len(losses)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hypers
    epochs = 50
    hidden_layer = 8
    
    # Instantiate the model
    model = MLP(hidden_size=hidden_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data
    X, y = load_data('data/stainless_steel_304_standardized.xlsx')

    # K Fold cross validation
    avg_loss = k_fold_validation(model, X, y, epochs, criterion, optimizer)

    # Train on full data
    full_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    full_dataloader = DataLoader(full_data, batch_size=64)

    train_model(model, full_dataloader, epochs, criterion, optimizer)
    print("Average Loss from K-Fold Cross Validation:", avg_loss)

    # Save model with kfold score in filename
    torch.save(model.state_dict(), f'models/model_{hidden_layer:.2f}_hidden_layers_kfold_loss_{avg_loss:.2f}.pt')
