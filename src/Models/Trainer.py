#!/usr/bin/env python
# coding: utf-8

import torch, os, time, pandas

import matplotlib.pyplot as plt
import seaborn as sn

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from torch_lr_finder import LRFinder

# Trains models
class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, save_folder="", data_loaders=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_folder = save_folder

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True

        # Note that it's possible to load a saved model with new data (i.e. for testing/using a model)
        if not data_loaders == None:
            self.data_loaders = data_loaders
        elif os.path.exists("Saved/DataLoaders"):
            self.data_loaders = torch.load("Saved/DataLoaders")
        
        self.model = nn.DataParallel(self.model)
        
        if not os.path.exists(self.save_folder + "/Model.tar"):
            self.model.epoch = 0
            self.best_acc = 0.0
            self.epoch_no_change = 0
        else:
            print("Loading saved model")
            checkpoint = torch.load(self.save_folder + "/Model.tar", map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            self.model.epoch = checkpoint["epoch"] + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.data_loaders = checkpoint["loaders"]
            self.best_acc = checkpoint["acc"]
            self.epoch_no_change = checkpoint["epoch_no_change"]
    
    def find_lr(self):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, self.device)
        lr_finder.range_test(self.data_loaders["train"], end_lr=10, num_iter=1000)
        lr_finder.plot()
        plt.savefig(self.save_folder + "/LRvsLoss.png")
        plt.close()
    
    def train(self, n_epochs=100):
        self.writer = SummaryWriter(self.save_folder + "/TensorBoard")

        for epoch in range(self.model.epoch, n_epochs):
            print("Epoch {}/{}:".format(epoch, n_epochs - 1))
            start_time = time.time()

            for phase in ["train", "validation"]:
                running_loss = 0.0
                running_corrects = 0
                
                if phase == "train":
                    self.model.train()
                else: self.model.eval()

                for i, item in enumerate(self.data_loaders[phase], 0):
                    progress = 100 * (i + 1) / len(self.data_loaders[phase])
                    formated_duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    print(f"Phase: {phase}      Progress: {progress}%       Elapsed Time: +{formated_duration}", end="\r")

                    inputs, labels = item.img.to(self.device), item.species_number.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.data_loaders[phase].sampler)
                epoch_acc = running_corrects.double() / len(self.data_loaders[phase].sampler)
                epoch_time = time.time() - start_time
                
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
                    }, self.save_folder + "/Model.tar")
                elif phase == "validation":
                    self.epoch_no_change += 1

                    if self.epoch_no_change >= 10:
                        break
                
            print()
        return self.model
    
    def jitter(self):
        self.model.eval()
        example = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(self.model, example)
        traced_script_module.save(self.save_folder + "/TorchScriptModel.pt")

    def evaluate(self, phase="test", feedback=True):
        #self.writer = SummaryWriter(self.save_folder + "/TensorBoard")
        start_time = time.time()

        running_loss = 0.0
        running_corrects = 0

        preds_list = torch.zeros(0, dtype=torch.long, device="cpu")
        labels_list = torch.zeros(0, dtype=torch.long, device="cpu")
        
        self.model.eval()

        for i, item in enumerate(self.data_loaders[phase], 0):
            progress = 100 * (i + 1) / len(self.data_loaders[phase])
            formated_duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Phase: {phase}      Progress: {progress}%       Elapsed Time: +{formated_duration}", end="\r")

            inputs, labels = item.img.to(self.device), item.species_number.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                preds_list = torch.cat([preds_list, preds.view(-1).cpu()])
                labels_list = torch.cat([labels_list, labels.view(-1).cpu()])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / len(self.data_loaders[phase].sampler)
        test_acc = running_corrects.double() / len(self.data_loaders[phase].sampler)
        test_time = time.time() - start_time

        print("\nPhase: {}, Loss: {:.4f}, Acc: {:.4f}, Time: {:.4f}".format(phase, test_loss, test_acc, test_time))

        #self.writer.add_scalar(test + "/loss", test_loss, epoch)
        #self.writer.add_scalar(test + "/acc", test_acc, epoch)

        #self.writer.flush()

        if feedback == True:
            report = metrics.classification_report(labels_list.numpy(), preds_list.numpy(), output_dict=True)
            confusion_matrix = metrics.confusion_matrix(labels_list.numpy(), preds_list.numpy())

            pandas.DataFrame(report).transpose().to_csv(self.save_folder + "/Report.csv")
            sn.heatmap(confusion_matrix).get_figure().savefig(self.save_folder + "/Confusion Matrix.png")
