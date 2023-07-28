import torch
import numpy as np

class LeitnerOHEM:
    def __init__(self, model, criterion, data_loader, device):
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device
        self.piles = {1: [], 2: [], 3: []}

    def calculate_losses(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                losses.extend(loss.tolist())
        self.model.train()
        return np.array(losses)

    def assign_to_piles(self, losses):
        thresholds = np.quantile(losses, [0.33, 0.66])
        for i, loss in enumerate(losses):
            if loss < thresholds[0]:
                self.piles[3].append(i)
            elif loss < thresholds[1]:
                self.piles[2].append(i)
            else:
                self.piles[1].append(i)

        # Print the mean and std of loss for each pile
        for pile in [1, 2, 3]:
            pile_losses = [losses[i] for i in self.piles[pile]]
            if pile_losses:
                print(f'Pile {pile}: Mean loss = {np.mean(pile_losses)}, Std loss = {np.std(pile_losses)}')

    def get_batch(self, pile_ratios):
        indices = []
        for pile, ratio in pile_ratios.items():
            pile_indices = np.random.choice(self.piles[pile], size=int(ratio * len(self.piles[pile])))
            indices.extend(pile_indices)
        return indices

    def update_piles(self):
        losses = self.calculate_losses()
        self.piles = {1: [], 2: [], 3: []}
        self.assign_to_piles(losses)
        
    def check_convergence(self, sigma):
        pile_1_max = max(self.calculate_losses(self.piles[1]))
        pile_3_min = min(self.calculate_losses(self.piles[3]))
        pile_2_std = np.std(self.calculate_losses(self.piles[2]))

        return abs(pile_1_max - pile_3_min) <= sigma * pile_2_std

    def calculate_losses(self, indices):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i in indices:
                inputs, targets = self.data_loader.dataset[i]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs.unsqueeze(0))
                loss = self.criterion(outputs, targets.unsqueeze(0))
                losses.append(loss.item())
        self.model.train()
        return np.array(losses)
