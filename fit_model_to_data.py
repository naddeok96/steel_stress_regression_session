import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from preprocess import process_data
from mlp import MLP

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "1"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set the number of folds
    num_folds = 5

    # Set the path to your data file
    data_file = "data/stainless_steel_304.xlsx"

    # Perform preprocessing and obtain the standardized data
    standardized_data = process_data(data_file)

    # Extract the input variables and target variable
    X = standardized_data[:, :-1]
    y = standardized_data[:, -1]

    # Convert the tensors to numpy arrays for k-fold cross-validation
    X = X.numpy()
    y = y.numpy()

    # Define the neural network model
    input_size = X.shape[1]
    hidden_size = 16
    output_size = 1
    model = MLP(input_size, hidden_size, output_size)

    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Perform k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True)
    fold = 1
    kfold_scores = []

    for train_index, test_index in kf.split(X):
        print(f"Fold: {fold}")
        fold += 1

        # Split the data into training and testing sets for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert the numpy arrays back to PyTorch tensors and move to the device
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)

        # Create DataLoader for training set
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train the model and print training loss
        train_loss = train(model, train_dataloader, criterion, optimizer)
        print(f"Training Loss: {train_loss}")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = torch.sqrt(criterion(test_outputs, y_test))  # Calculate RMSE
            print(f"Test Loss (RMSE): {test_loss.item()}")

        # Store the k-fold score (RMSE)
        kfold_scores.append(test_loss.item())

    # Train the model on all data
    X_all = torch.from_numpy(X).float().to(device)
    y_all = torch.from_numpy(y).float().to(device)
    train_dataset_all = TensorDataset(X_all, y_all)
    train_dataloader_all = DataLoader(train_dataset_all, batch_size=32, shuffle=True)

    model.train()
    optimizer.zero_grad()
    outputs_all = model(X_all)
    loss_all = criterion(outputs_all, y_all)
    loss_all.backward()
    optimizer.step()

    # Save the trained model with average k-fold score (RMSE) under "models" folder
    kfold_score_avg = sum(kfold_scores) / len(kfold_scores)
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)
    model_save_path = os.path.join(models_folder, "trained_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved with average k-fold score (RMSE): {kfold_score_avg}")
