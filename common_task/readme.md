# Electron/Photon Classification using ResNet-15

This repository contains a solution for the electron/photon classification task using a custom ResNet-15 architecture.

## Task Description

The task involves classifying 32x32 matrices representing detector data for two types of particles:
- Electrons 
- Photons

The original data has two channels:
1. Hit energy
2. Hit time

## Dataset

The data consists of detector images from:
- [Photons dataset](https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc)
- [Electrons dataset](https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA)

## Approach

### Feature Engineering

A key innovation in this solution is the addition of a third channel that captures the interaction between energy and time components:

```python
epsilon = 1e-6  # Avoid division by zero
new_channel = (data[:,0] * data[:,1]) / (np.sum(data[:,0], axis=(-2,-1), keepdims=True) + epsilon)
```

This new channel represents a normalized product of energy and time, highlighting areas where both significant energy and timing information coincide. This feature proved to be more important than the raw time channel for classification.

### Model Architecture

The classification model is based on a custom ResNet-15 architecture:
- Modified from ResNet-18 by removing layer4 completely
- Added two fully connected layers at the end
- Total of 15 layers with trainable parameters

The architecture breakdown:
- 1 initial convolution layer
- 4 convolutional layers in layer1 (2 blocks)
- 4 convolutional layers in layer2 (2 blocks)
- 4 convolutional layers in layer3 (2 blocks)
- 2 fully connected layers

### Training

- 80% of the data is used for training, 20% for validation
- Cross-entropy loss function
- Adam optimizer
- Learning rate scheduling for better convergence
- Data augmentation techniques to prevent overfitting

## Results

The model achieves 72% accuracy on the validation set without overfitting. This performance demonstrates the effectiveness of both the custom ResNet-15 architecture and the engineered third channel that captures the normalized product of energy and time information.