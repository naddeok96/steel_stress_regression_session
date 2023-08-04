import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from mlp import MLP  # Make sure mlp.py is in the same directory
from leitner_mining import LeitnerOHEM
import numpy as np
from time import time

def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def leitner_train_model(model, train_data, epochs, criterion, optimizer, device, fold=None, folds=None):
    model.train()
    dataloader = DataLoader(train_data, batch_size=64)
    
    ohem = LeitnerOHEM(model, criterion, dataloader, device)
    mean_loss_threshold = 0.0001
    loss_std_threshold = 0.001
    
    
    update_count = 0

    while update_count < epochs:
        mean_all_losses, std_all_losses = ohem.update_piles(dataloader)
        print("The mean/std for all losses is: ", mean_all_losses, "/", std_all_losses)

        # Check if std_all_losses is under the threshold
        if mean_all_losses < mean_loss_threshold and std_all_losses < loss_std_threshold:
            break

        max_pile_3 = max(ohem.calculate_losses(ohem.piles[3]))
        for pile in [2, 1]:  # start with pile 2
            inner_update_count = 0
            pile_size = len(ohem.piles[pile])
            while pile_size > 0 and inner_update_count < epochs:
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
                
                # if prev_pile_size != pile_size:
                #     if fold is None:
                #         print(f'Final Train. Pile Update {inner_update_count+1} in pile {pile}. Number of data points in pile: {pile_size}. Maximum loss in current pile: {max(individual_losses)}. Max loss in pile 3: {max_pile_3}')
                #     else:
                #         print(f'Fold: {fold+1}/{folds}. Pile Update {inner_update_count+1} in pile {pile}. Number of data points in pile: {pile_size}. Maximum loss in current pile: {max(individual_losses)}. Max loss in pile 3: {max_pile_3}')
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

        if fold is None:
            print(f'Final Train. Update {update_count+1}/{epochs}, Loss: {loss.item()}')
        else:
            print(f'Fold: {fold+1}/{folds}. Update {update_count+1}/{epochs}, Loss: {loss.item()}')

    return model


def test_model(model, criterion, data):
    model.eval()
    total_loss = 0
    total_samples = 0
    dataloader = DataLoader(data, batch_size=64)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(-1).to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * len(X) # Multiply by the number of samples in the batch
            total_samples += len(X) # Keep track of the total number of samples
            print("Test Loss ", total_loss)
            
    avg_test_loss = total_loss / total_samples # Divide by the total number of samples
    print(f'Avg Test Loss: {avg_test_loss}')
    return avg_test_loss

def k_fold_validation(base_model, X, y, epochs, criterion, optimizer, device, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    losses = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        model = type(base_model)(hidden_size=base_model.layers[2].in_features).to(device)  # create a new instance of the same model
        model.load_state_dict(base_model.state_dict())
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        leitner_model = leitner_train_model(model, train_data, epochs, criterion, optimizer, device, fold, k)
        loss = test_model(leitner_model, criterion, test_data)
        losses.append(loss)

    return np.mean(losses), np.std(losses)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hypers
    epochs = 100
    hidden_layer = 16
    
    # Instantiate the model
    model = MLP(hidden_size=hidden_layer).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load data
    X, y = load_data('data/stainless_steel_304_standardized.xlsx')

    # K Fold cross validation
    avg_kfold_mse, std_kfold_mse = k_fold_validation(model, X, y, epochs, criterion, optimizer, device)

    # Train on full data
    full_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    start_time = time()
    leitner_model = leitner_train_model(model, full_data, epochs, criterion, optimizer, device)
    training_time = time() - start_time
    print("Average Loss from K-Fold Cross Validation:", avg_kfold_mse)

    test_loss = test_model(model, criterion, full_data)

    results = {
        "Average K-Fold MSE": [avg_kfold_mse],
        "Standard Deviation of K-Fold MSE": [std_kfold_mse],
        "Training Time (Seconds)": [training_time],
        "Test Loss (MSE)": [test_loss]
    }
    df_results = pd.DataFrame(results)
    df_results.to_excel(f"data/leitner_model_{hidden_layer:.0f}_hidden_layers_{epochs:.0f}_epochs_kfold_loss_{avg_kfold_mse:.6f}".replace('.', '_') + "_results.xlsx", index=False, float_format="%.6f")


    # Save model with kfold score in filename
    torch.save(leitner_model.state_dict(), f"models/leitner_model_{hidden_layer:.0f}_hidden_layers_{epochs:.0f}_epochs_kfold_loss_{avg_kfold_mse:.4f}".replace('.', '_') + ".pt")
