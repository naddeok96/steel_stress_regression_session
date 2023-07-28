import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from mlp import MLP  # Make sure mlp.py is in the same directory
from leitner_mining import LeitnerOHEM
import numpy as np

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def leitner_train_model(model, train_data, max_updates, criterion, optimizer, device, fold=None):
    model.train()
    dataloader = DataLoader(train_data, batch_size=64)
    
    ohem = LeitnerOHEM(model, criterion, dataloader, device)
    loss_std_threshold = 0.01
    
    
    update_count = 0

    while update_count < max_updates:
        std_all_losses = ohem.update_piles(dataloader)
        print("The std for all losses is: ", std_all_losses)

        # Check if std_all_losses is under the threshold
        if std_all_losses < loss_std_threshold:
            break

        max_pile_3 = max(ohem.calculate_losses(ohem.piles[3]))
        for pile in [2, 1]:  # start with pile 2
            inner_update_count = 0
            pile_size = len(ohem.piles[pile])
            while pile_size > 0 and inner_update_count < max_updates:
                indices = ohem.piles[pile]
                np.random.shuffle(indices)

                # Train on the whole batch
                X, y = train_data[indices]
                X, y = X.to(device), y.unsqueeze(-1).to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                # Calculate individual losses
                with torch.no_grad():
                    individual_losses = []
                    for idx in indices:
                        X_single, y_single = train_data[idx]
                        X_single, y_single = X_single.to(device).unsqueeze(0), y_single.to(device).unsqueeze(0).unsqueeze(-1)
                        output_single = model(X_single)
                        loss_single = criterion(output_single, y_single)
                        individual_losses.append(loss_single.item())
                indices_to_remove = [idx for idx, loss in zip(indices, individual_losses) if loss <= max_pile_3]

                prev_pile_size = pile_size
                for idx in indices_to_remove:
                    ohem.piles[pile].remove(idx)
                pile_size = len(ohem.piles[pile])
                
                if prev_pile_size != pile_size:
                    if fold is None:
                        print(f'Final Train. Pile Update {inner_update_count+1} in pile {pile}. Number of data points in pile: {pile_size}. Maximum loss in current pile: {max(individual_losses)}. Max loss in pile 3: {max_pile_3}')
                    else:
                        print(f'Fold: {fold}. Pile Update {inner_update_count+1} in pile {pile}. Number of data points in pile: {pile_size}. Maximum loss in current pile: {max(individual_losses)}. Max loss in pile 3: {max_pile_3}')
                inner_update_count += 1

        # Take a gradient step on the whole dataset
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        update_count += 1


        print(f'Update {update_count+1}/{max_updates}, Loss: {loss.item()}')

    return model


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

def k_fold_validation(base_model, X, y, max_updates, criterion, optimizer, device, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    losses = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        test_dataloader = DataLoader(test_data, batch_size=64)

        model = type(base_model)(hidden_size=base_model.layers[2].in_features).to(device)  # create a new instance of the same model
        model.load_state_dict(base_model.state_dict())
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        leitner_model = leitner_train_model(model, train_data, max_updates, criterion, optimizer, device, fold)
        loss = test_model(leitner_model, criterion, test_dataloader)
        losses.append(loss)

    return sum(losses) / len(losses)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hypers
    max_updates = 5000
    hidden_layer = 16
    
    # Instantiate the model
    model = MLP(hidden_size=hidden_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data
    X, y = load_data('data/stainless_steel_304_standardized.xlsx')

    # K Fold cross validation
    avg_loss = k_fold_validation(model, X, y, max_updates, criterion, optimizer, device)

    # Train on full data
    full_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    leitner_model = leitner_train_model(model, full_data, max_updates, criterion, optimizer, device)
    print("Average Loss from K-Fold Cross Validation:", avg_loss)

    # Save model with kfold score in filename
    torch.save(leitner_model.state_dict(), f"models/leitner_model_{hidden_layer:.0f}_hidden_layers_kfold_loss_{avg_loss:.4f}".replace('.', '_') + ".pt")
