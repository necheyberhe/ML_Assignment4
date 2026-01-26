#  Flower Image Classification Using CNNs

### Students

| Name              |  email                    |
|-------------------|---------------------------|
| Omer Eliyahu      |  Omereliy@post.bgu.ac.il  |
| Nechi Berhe Weldu |  weldu@post.bgu.ac.il     |
| Amit Ner Gaon     |  amitner@post.bgu.ac.il   |


## Overview

This project implements flower image classification using transfer learning with Convolutional Neural Networks (CNNs).
The goal is to classify images into 102 flower categories and output probabilistic predictions for each class.

Three mandatory pretrained models are used and compared:

1. VGG19
2. YOLOv5 (classification variant)
3. Resnet50


Each model is trained and evaluated using two different random seeds to ensure robustness, as required by the assignment. 

The source code is available at: https://github.com/necheyberhe/Assignment4.git


## Models

### VGG19

* Pretrained on ImageNet.
* Backbone layers frozen.
* Output fully connected layer replaced to output 102 classes.
* Trained using standard PyTorch training loop.


### YOLOv5 (Classification Model)

* Model used: yolov5s-cls.pt (Ultralytics classification variant).
* Backbone frozen.
* Classification head retrained for 102 classes.


### Resnet50

* Backbone: torchvision.models.resnet50.
* Pretrained on ImageNet.
* Head: final fully-connected layer replaced with nn.Linear(in_features, 102).
* If freeze_backbone=True:
  * All pretrained backbone layers are frozen (requires_grad = False).
  * Only the final classifier layer model.fc is trainable.
* If freeze_backbone=False:
  * Entire network is trainable.



## Dataset

### Primary Dataset (Required)

Oxford Flowers 102 Dataset
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

#### Dataset Splits

* For each experiment: Training: 50%, Validation: 25%, Test: 25%.

* The dataset split is randomized and repeated twice using different seeds (Seed 1, Seed 2).

* This satisfies the requirement to repeat the random split at least twice.

### Preprocessing

* Images resized to 224 × 224.

* Normalization applied using ImageNet mean and standard deviation.

* Identical preprocessing used across VGG19 and YOLOv5 for consistency.

* Data loaders created separately for each seed to ensure reproducibility.


### Training Details

* Loss Function: Cross-Entropy Loss.

* Optimizer: Adam.

*  Rate Scheduling: ReduceLROnPlateau.
  
* Each model is trained independently for Seed 1 and Seed 2.


**Metrics Tracked Per Epoch:**

1. Accuracy (train / validation / test)

2. Cross-entropy loss (train / validation / test)

