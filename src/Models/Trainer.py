#!/usr/bin/env python
# coding: utf-8

import torch, time

import numpy as np

from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from .TrainHelper import LossAccuracyKeeper
from Data.SnakeDataset import SnakeDataset

# Trains models
class Trainer():
    def __init__(self, model, transforms, criterion, optimizer):
        self.train_loader, self.validation_loader, self.test_loader = self.get_loaders(transforms)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.criterion, self.optimizer = criterion, optimizer
    
    def get_loaders(self, transforms=transforms.ToTensor()):
        # Train - fit model, Validation - tune hyperparameters, Test - final results
        train_data = SnakeDataset("../train", "../train_labels.csv", transforms["train"])
        validation_data = SnakeDataset("../train", "../train_labels.csv", transforms["validation"])
        test_data = SnakeDataset("../train", "../train_labels.csv", transforms["validation"])

        num_images = len(train_data)
        shuffle = np.random.permutation(num_images)

        # Position of where to start and end splitting data
        # Extra duplicates which don't use full dataset are for testing
        train_end = int(num_images * 0.95)
        validation_start = int(num_images * 0.95)
        validation_end = int(num_images * 0.975)
        test_start = int(num_images * 0.975)

        #train_end = int(num_images * 0.001)
        #validation_start = int(num_images * 0.001)
        #validation_end = int(num_images * 0.002)
        #test_start = int(num_images * 0.0002)
        #test_end = int(num_images * 0.0003)

        train_sampler = SubsetRandomSampler(shuffle[:train_end])
        validation_sampler = SubsetRandomSampler(shuffle[validation_start:validation_end])
        test_sampler = SubsetRandomSampler(shuffle[test_start:])
        #test_sampler = SubsetRandomSampler(shuffle[test_start:test_end])

        train_loader = DataLoader(train_data, sampler=train_sampler)
        validation_loader = DataLoader(validation_data, sampler=validation_sampler)
        test_loader = DataLoader(test_data, sampler=test_sampler)

        return train_loader, validation_loader, test_loader

    def train(self, save_path, max_epoch_stop=5, n_epochs=30, print_every=2):
        try:
            print("self.model has been trained for: " + self.model.epochs + " epochs")
        except:
            self.model.epochs = 0
            print("Starting training from scratch")

        overall_start = time.time()

        loss_accuracy_keeper = LossAccuracyKeeper()

        for epoch in range(n_epochs):
            loss_accuracy_keeper.reset()

            epoch_start_time = time.time()
            self.model.train()

            # Train loop
            for ii, (data, target) in enumerate(self.train_loader, 0):
                # Use GPU when available
                data, target = data.to(self.device), target.to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()
                output = self.model(data)

                # Backpropagate
                loss = self.criterion(output, target)
                loss.backward()

                # Update prameters
                self.optimizer.step()

                loss_accuracy_keeper.update_loss_acc(data, target, output, self.criterion, "train")

                # Training progress
                percent = 100 * (ii + 1) / len(self.train_loader)
                elapsed_time = time.time() - epoch_start_time
                print(
                    f"Epoch: {epoch}\t{percent:.2f}% complete. {elapsed_time:.2f} seconds elapsed",
                    end="\r"
                )

            # Validation
            else:
                self.model.epochs += 1

                # Don't track gradients
                with torch.no_grad():
                    # Evalutaion
                    self.model.eval()

                    # Validation loop
                    for data, target in self.validation_loader:
                        # Use GPU if available
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # Forward pass
                        output = self.model(data)

                        loss_accuracy_keeper.update_loss_acc(data, target, output, self.criterion, "validation")
                    
                    # Average losses
                    loss_accuracy_keeper.update_average_loss_acc(len(self.train_loader.dataset))

                    # Print results
                    if (epoch + 1) % print_every == 0:
                        loss_accuracy_keeper.print_progress(epoch)
                    
                    # Judges whether improvement has been made
                    # If not in set max number of epochs, stop early
                    if loss_accuracy_keeper.safe_improvement(max_epoch_stop) == [True, True]:
                        state = {
                            "epoch": epoch,
                            "state_dict": self.model.state_dict,
                            "optimizer": self.optimizer.state_dict(),
                            "validation_loss_min": loss_accuracy_keeper.validation_loss_min,
                        }
                        torch.save(state, save_path)
                        best_epoch = epoch
                    elif loss_accuracy_keeper.safe_improvement(max_epoch_stop) == [False, False]:
                        # Load state dict
                        #self.model.load_state_dict(torch.load(save_path))
                        break

        self.model.optimizer = self.optimizer
        total_time = time.time() - overall_start
        validation_acc = loss_accuracy_keeper.acc["validation"]
        print(
            f"\nBest epoch: {best_epoch} with loss: {loss_accuracy_keeper.validation_loss_min:.2f} and accuracy: {100 * validation_acc:.2f}%"
        )
        print(
            f"{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch"
        )
        return self.model, loss_accuracy_keeper.get_history_dataframe()

    def load(self, model, path, trainning=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if trainning == False:
            model.eval()
        else: model.train()
