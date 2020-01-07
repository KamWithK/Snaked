#!/usr/bin/env python
# coding: utf-8

import torch, os, time

import numpy as np

from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from Data.SnakeDataset import SnakeDataset

# Trains models
class Trainer():
    def __init__(self, model, transforms, criterion, optimizer, scheduler, path_saved="", trainning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        if not os.path.exists(path_saved):
            self.model.epoch = 0
            self.best_acc = 0.0
            self.epoch_no_change = 0

            if os.path.exists("Saved/DataLoaders"):
                self.data_loaders = torch.load("Saved/DataLoaders")
            else: self.data_loaders = self.get_loaders(transforms)
        else:
            print("Loading saved model")
            checkpoint = torch.load(path_saved)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model.epoch = checkpoint["epoch"] + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.data_loaders = checkpoint["loaders"]
            self.best_acc = checkpoint["acc"]
            self.epoch_no_change = checkpoint["epoch_no_change"]

            if trainning == False:
                self.model.eval()
            else: self.model.train()
        
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            nn.DataParallel(self.model)
    
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

        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=64, num_workers=10)
        validation_loader = DataLoader(validation_data, sampler=validation_sampler, batch_size=64, num_workers=10)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=64, num_workers=10)

        data_loaders = {
            "train": train_loader,
            "validation": validation_loader,
            "test": test_loader
        }

        return data_loaders

    def train(self, save_folder, n_epochs=30):
        self.writer = SummaryWriter(save_folder + "/TensorBoard")

        for epoch in range(self.model.epoch, n_epochs):
            print("Epoch {}/{}:".format(epoch, n_epochs - 1))
            start_time = time.time()

            for phase in ["train", "validation"]:
                running_loss = 0.0
                running_corrects = 0
                
                if phase == "train":
                    self.model.train()
                else: self.model.eval()

                for i, (inputs, labels) in enumerate(self.data_loaders[phase], 0):
                    progress = 100 * (i + 1) / len(self.data_loaders[phase])
                    
                    if i == 0:
                        time_left, formated_duration = 0, 0
                    else:
                        time_left = (100 / progress) * (time.time() - start_time)
                        formated_duration = time.strftime("%H:%M:%S", time.gmtime(time_left))

                    print(f"Phase: {phase}      Progress: {progress * 100}%       Time Left: +{formated_duration}", end="\r")

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(self.data_loaders[phase].sampler)
                epoch_acc = running_corrects.double() / len(self.data_loaders[phase].sampler)
                epoch_time = time.time() - start_time
                
                if phase == "validation":
                    self.scheduler.step(epoch_loss)

                self.writer.add_scalar(phase + "/loss", epoch_loss, epoch)
                self.writer.add_scalar(phase + "/acc", epoch_acc, epoch)

                self.writer.flush()

                print("\nPhase: {}, Loss: {:.4f}, Acc: {:.4f}, Time: {:.4f}".format(phase, epoch_loss, epoch_acc, epoch_time))

                if phase == "validation" and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    torch.save({
                        "epoch": epoch,
                        "epoch_no_change": self.epoch_no_change,
                        "acc": epoch_acc,
                        "loss": epoch_loss,
                        "loaders": self.data_loaders,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict()
                    }, save_folder + "/Model.tar")
                elif phase == "validation":
                    self.epoch_no_change += 1

                    if self.epoch_no_change >= 10:
                        break
                
            print()
        return self.model
