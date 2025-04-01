# Discovery of Hidden Symmetries and Conservation Laws

## Project Overview
This repository contains my submission for the GSoC 2025 project "Discovery of Hidden Symmetries and Conservation Laws" under ML4SCI. The project focuses on using deep learning techniques to discover symmetries and conservation laws in physical systems, with applications in particle physics.

## Repository Structure
- `/common_task/`: Contains the implementation for the electron/photon classification task using a ResNet-15 architecture
- `/specific_task/`: Contains the implementation for the symmetry discovery tasks
  - Task 1: Dataset preparation (MNIST rotation) and Variational Auto-Encoder implementation
  - Task 2: Supervised symmetry discovery using MLP on the latent space
  - Task 3: Unsupervised symmetry discovery preserving logits
  - Bonus Task: Rotation invariant network implementation using discovered symmetries

## Project Tasks

### Common Task: Electron/Photon Classification
Classification of particles (electrons vs photons) using detector data represented as 32x32 matrices with two channels (hit energy and time). Implemented using a ResNet-15 like architecture with PyTorch.

### Specific Task: Symmetry Discovery
1. **Dataset Preparation**: Rotating MNIST digits (focusing on digits 1 and 2 if computational budget is limited) in 30-degree increments and storing them in an appropriate data format.
2. **Latent Space Creation**: Building and training a Variational Auto-Encoder using the rotated dataset.
3. **Supervised Symmetry Discovery**: Learning transformations in latent space using MLP that map vectors to their rotated versions in image space.
4. **Unsupervised Symmetry Discovery**: Discovering symmetries in the MNIST dataset that preserve the logit, with rotation being one of the expected discoveries. Replicated the paper "Oracle-Preserving Latent Flows" with Variational auto Encoder.
5. **Bonus Task**: Building a rotation invariant network using the symmetries discovered in the previous tasks.

## Links
- [GSoC 2025 Project Description](https://ml4sci.org/gsoc/2025/proposal_E2E8.html)
- [Oracle-Preserving Latent Flows](https://arxiv.org/abs/2302.00806)