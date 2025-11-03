# WGAN-GP (Wasserstein GAN with Gradient Penalty) — PyTorch

This project implements a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** using **PyTorch**.  
The model can be trained on **MNIST** or **CelebA** datasets and generates realistic images from random noise.

---

## 🧠 Overview

**WGAN-GP** improves training stability over traditional GANs by replacing the discriminator with a **critic** and enforcing the **Lipschitz constraint** using a **gradient penalty** instead of weight clipping.

**Architecture Highlights:**
- Deep Convolutional Generator and Critic (based on DCGAN)
- Wasserstein loss function with Gradient Penalty
- Training on 64×64 grayscale images (MNIST) or RGB images (CelebA)
- TensorBoard integration for visualizing real and generated images

---

## 🚀 Features

- Stable GAN training using WGAN-GP loss  
- Uses **InstanceNorm** (Critic) and **BatchNorm** (Generator)  
- Customizable architecture through hyperparameters  
- Simple dataset switch between **MNIST** and **CelebA**  
- Real-time logging using **TensorBoard**

---

## 🧩 Requirements

Install the dependencies:

```bash
pip install torch torchvision tensorboard

