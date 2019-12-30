#!/usr/bin/env python
# coding: utf-8

import torch

import numpy as np
import pandas as pd

class LossAccuracyKeeper():
    def __init__(self):
        self.history = []
        self.validation_loss_min = np.Inf
        self.epochs_no_improvement = 0

        self.loss = {
            "train": 0.0,
            "validation": 0.0
        }
        
        self.acc = {
            "train": 0.0,
            "validation": 0.0
        }
        
    def update_loss_acc(self, data, target, output, criterion, form: str):
        loss = criterion(output, target)
        self.loss[form] += loss.item() * data.size(0)
        
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        self.acc[form] += accuracy.item() * data.size(0)
    
    def update_average_loss_acc(self, data_length):
        # Average losses and accuracy
        self.loss = self.loss / len(data_length)
        self.acc = self.acc / len(data_length)

        if self.loss["validation"] < self.validation_loss_min:
            # Track improvement
            self.epochs_no_improvement = 0
            self.validation_loss_min = self.loss["validation"]
        else:
            self.epochs_no_improvement += 1
    
    # Returns array indicating whether improvement has been made:
    # Now and over a set number of epochs
    def safe_improvement(self, max_epoch_stop):
        if self.epochs_no_improvement == 0:
            return [True, True]
        elif self.epochs_no_improvement >= max_epoch_stop:
            return [False, True]
        else: return [False, False]

    
    def print_progress(self, epoch):
        # Workaround for inability to print value from dictionary
        train_loss, validation_loss = self.loss["train"], self.loss["validation"]
        train_acc, validation_acc = self.acc["train"], self.acc["validation"]
        
        print(
            f"\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {validation_loss:.4f}"
        )
        print(
            f"\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * validation_acc:.2f}%"
        )
    
    def update_history(self):
        self.history.append([self.loss["train"], self.loss["validation"], self.acc["train"], self.acc["validation"]])
    
    def get_history_dataframe(self):
        self.history = pd.DataFrame(
                        self.history,
                        columns=[
                            "train_loss", "validation_loss", "train_acc", "validation_acc"
                        ]
        )
        return self.history