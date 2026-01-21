Assignment 4 – Flower Classification Using CNNs and Transfer Learning
===============================================================================================================
Overview

This project implements flower image classification using transfer learning with Convolutional Neural Networks (CNNs).
The goal is to classify images into 102 flower categories and output probabilistic predictions for each class.

Two mandatory pretrained models are used and compared:

.VGG19

YOLOv5 (classification variant)

Each model is trained and evaluated using two different random seeds to ensure robustness, as required by the assignment.
----------------------------------------------------------------------------------------------------------------
Models Used
================================================================================================================
1. VGG19
----------------------------------------------------------------------------------------------------------------
Pretrained on ImageNet

Backbone layers frozen

Final fully connected layer replaced to output 102 classes

Trained using standard PyTorch training loop


2. YOLOv5 (Classification Model)
----------------------------------------------------------------------------------------------------------------
Model used: yolov5s-cls.pt (Ultralytics classification variant)

Pretrained on ImageNet

Backbone frozen, classification head retrained for 102 classes

Used as a generic CNN classifier within a custom PyTorch training loop

Dataset
================================================================================================================
Primary Dataset (Required)

Oxford Flowers 102 Dataset
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Dataset Splits

For each experiment:

Training: 50%

Validation: 25%

Test: 25%

The dataset split is randomized and repeated twice using different seeds:

Seed 1

Seed 2

This satisfies the requirement to repeat the random split at least twice.

Preprocessing
===================================================================================================================

Images resized to 224 × 224

Normalization applied using ImageNet mean and standard deviation

Identical preprocessing used across VGG19 and YOLOv5 for consistency

Data loaders created separately for each seed to ensure reproducibility

Training Details
=========================================================================================================================
Loss Function: Cross-Entropy Loss

Optimizer: Adam

Learning Rate Scheduling: ReduceLROnPlateau

Metrics Tracked Per Epoch:

Accuracy (train / validation / test)

Cross-entropy loss (train / validation / test)

Each model is trained independently for Seed 1 and Seed 2.
