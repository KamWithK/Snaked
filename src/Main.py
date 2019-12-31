#!/usr/bin/env python
# coding: utf-8

import TransferLearning

import matplotlib.pyplot as plt
import seaborn as sns

model, history = TransferLearning.model, TransferLearning.history

plt.figure(1)
for c in ["train_loss", "validation_loss"]:
    plt.plot(history[c], label=c)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Average Negative Log Likelihood")
plt.title("Training and Validation Losses")
plt.savefig("Training and Validation Losses.png")

plt.figure(2)
for c in ["train_acc", "validation_acc"]:
    plt.plot(100 * history[c], label=c)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Average Accuracy")
plt.title("Training and Validation Accuracy")
plt.savefig("Training and Validation Accuracy")
