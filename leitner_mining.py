import torch
import numpy as np

class LeitnerOHEM:
    def __init__(self, model, criterion, data_loader, device):
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device
        self.piles = {1: [], 2: [], 3: []}

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
            pile_indices = self.piles[pile]
            pile_losses = self.calculate_losses(pile_indices)
            # if len(pile_losses) > 0:
                # print(f'Pile {pile}: Mean loss = {np.mean(pile_losses)}, Std loss = {np.std(pile_losses)}')


    def get_batch(self, pile_ratios):
        indices = []
        for pile, ratio in pile_ratios.items():
            pile_indices = np.random.choice(self.piles[pile], size=int(ratio * len(self.piles[pile])))
            indices.extend(pile_indices)
        return indices

    def update_piles(self, dataloader):
        all_indices = list(range(len(dataloader.dataset)))
        losses = self.calculate_losses(all_indices)
        self.piles = {1: [], 2: [], 3: []}
        self.assign_to_piles(losses)

        # Calculate the standard deviation of the losses
        losses_mean = np.mean(losses)
        losses_std = np.std(losses)

        return losses_mean, losses_std

    def calculate_losses(self, indices):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i in indices:
                inputs, targets = self.data_loader.dataset[i]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs.unsqueeze(0))
                loss = self.criterion(outputs, targets.unsqueeze(0).unsqueeze(-1))
                losses.append(loss.item())
        self.model.train()
        return np.array(losses)
