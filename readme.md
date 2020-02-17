# Snaked
## Purpose
5.4 million people are bitten by snakes every year and 81-138 million people die due to snake bites each year.
Preventing snake bites is clearly a major issue, in need of preventative measures to save lives.
This repository serves as a demonstration of the code required to identify snake species and whether poisonous from photographs.
A sample application/proof of concept is additionally available on the [Google Play Store](https://play.google.com/store/apps/details?id=com.kamwithk.snaked)!

## Challenge
* Each snake species varies in shape, colour, size, texture and more.
* Over 3000 snake species have already been discovered worldwide!
* Different snake species may look nearly identical, however, vary significantly

So it is clearly not an easy job to identify one snake from another, even though it's quintessential that we do.

## Frameworks and Methods Utilized
PyTorch is used for ALL deep learning code and Numpy for numerical computations.
The code is separated into the main file outlining the chosen algorithms (can be swapped/modified easily) and abstracted code to help train, evaluate and create an executable for any model easily.

There are several novel ideas/techniques used here which help to create neater/more readable pythonic code.
The three primary examples of new PyTorch techniques:
* The Item tuple class which allows modular and further extensible code (when extra data needs to be processed)
* Use of super-convergence/the one-cycle policy in pure PyTorch instead of a highly abstracted library (i.e. Fast.AI)/bare python/numpy
* Use of a dictionary to dictate how different data sets should be split up to allow easy modifications of data proportions (i.e. switching between full dataset for training and small batches for ensuring all code runs without errors)

Although several models were trialled out on the dataset, in the end, a MobileNetV2 model provided the best results, whilst also remaining relatively lightweight and so able to run on low-power devices like phones (essential for the app).
The final model was trained using LDAM loss instead of cross-entropy loss due to the dataset being imbalanced (some classes having far more samples than others).
Note that the codebase has support for classical rebalancing, however, experimental trials show that this method causes overfitting extremely early on.
The model manages to achieve around a 70% accuracy and F1 score.
More details about the [choice of model](https://www.kamwithk.com/modern-algorithms-choosing-a-model-ck6hwiovf004ndfs1529fcppy), [how it was improved](https://www.kamwithk.com/improving-your-computer-vision-model-ck6k3em3b0113dfs16bray6ee), [report](https://www.kamwithk.com/snake-classification-report-ck6oj5dg202fwdfs10qim8fwd) and [lessons learnt from this project](https://www.kamwithk.com/how-i-published-an-app-and-model-to-classify-85-snake-species-and-how-you-can-too-ck6jb8er400r0dfs1agw7d0y4) can be found on [The Data Science Swiss Army Knife blog](https://www.kamwithk.com/).

## Android Application
The android application created for this project was written with Kotlin, using Fotoapparat (for easy camera support) and PyTorch (for utilizing the chosen PyTorch model) libraries.
Due to the lightweight MobileNetV2 model no network is required to connect to a server (which would normally run the computations itself).
This is intentionally done to facilitate use within remote locations!
Please note that this is only a sample proof of concept app and if you're bitten consult a medical expert immediately.

## Sources
All data currently used for the project comes from [AIcrowd's Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge).
The images and labels have been used, however, geographic locations have been ignored to allow easy usage for any image, even if it hasn't been tagged.
The dataset allows 85 species to be labelled.
A Jupyter Notebook is also provided which demonstrates how to collect further data using Google Image searches!
All statistics used here, or within the repository are from the World Health Organisation (unless otherwise stated) or pertaining specifically to the Snaked source code.

For further information please see the following:
* [Snakebite envenoming](https://www.who.int/news-room/fact-sheets/detail/snakebite-envenoming)
* [Venomous Snakes Distribution and Species Risk Categories](http://apps.who.int/bloodproducts/snakeantivenoms/database/)
* [AIcrowd Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge)
