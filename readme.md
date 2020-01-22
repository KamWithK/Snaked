# Snaked
## Purpose
5.4 million people are bitten by snakes every year and 81-138 million people die due to snake bites each year.
Preventing snake bites is clearly a major issue, in need of preventitive measures to save lives.
This repository serves as a demonstration of the code required to identify snake species and whether poisonous from photographs.
A sample application/proof of concept will additionally be available via Google Play Store!

## Challenge
* Each snake species varies in shape, colour, size, texture and more.
* Over 3000 snake species have already been discovered worldwide!
* Different snake species may look nearly identical, however vary significantly

So it is clearly not an easy job to identify one snake from another, even though it's quintessential that we do.

## Frameworks and Methods Utilized
PyTorch is used for ALL deep learning code and Numpy for numerical computations.
The code is seperated into a main file outlining the chosen algorithms (can be swaped/modified easily) and abstracted code to help train, evaluate and create an executable for any model easily.

There are several novel ideas/techniques used here which help to create neater/more readable pythonic code.
The three primary examples of new PyTorch techniques:
* The Item tuple class which allows modular and further extensable code (when extra data needs to be processed)
* Use of super convergence/the one-cycle policy in pure PyTorch instead of a highly abstracted library (i.e. Fast.AI)/bare python/numpy
* Use of a dictionary to dictate how different data sets should be split up to allow easy modifications of data proportions (i.e. switching between full dataset for training and small batches for ensuring all code runs without errors)

## Sources
All data currently used for the project comes from AIcrowd's "Snake Species Identification Challenge".
A Jupyter Notebook is also provided which demonstrates how to collect further data using Google Image searches!
All statistics used here, or within the repository are from the World Health Organinsation (unless otherwise stated).

For further information please see the following:
* [Snakebite envenoming](https://www.who.int/news-room/fact-sheets/detail/snakebite-envenoming)
* [Venomous Snakes Distribution and Species Risk Categories](http://apps.who.int/bloodproducts/snakeantivenoms/database/)
* [AIcrowd Snake Species Identification Challenge](https://www.aicrowd.com/challenges/snake-species-identification-challenge)
