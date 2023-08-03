import os
import torch
from torch import nn
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Initialize GPU usage
    gpu_number = "1"
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Instantiate the model and move it to the device
    model = MLP(hidden_size=16).to(device)
    
    # Print the model summary
    summary(model, input_size=(3,))
