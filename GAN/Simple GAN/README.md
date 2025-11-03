# GAN on MNIST using PyTorch

This project implements a **Generative Adversarial Network (GAN)** from scratch using **PyTorch** to generate handwritten digits similar to the MNIST dataset.

---

## 🧠 Overview

A **GAN** consists of two neural networks — a **Generator** and a **Discriminator** — that compete in a zero-sum game:

- **Generator (G)**: Learns to create fake images from random noise that resemble real MNIST digits.
- **Discriminator (D)**: Learns to distinguish between real MNIST images and fake ones created by the generator.

Training improves both models until the generator produces realistic-looking digits.

---

## 🚀 Features

- Simple GAN architecture using fully connected layers
- Uses **LeakyReLU**, **Tanh**, and **Sigmoid** activations
- Trains on the **MNIST** dataset
- Tracks training progress with **TensorBoard**
- Visualizes real and generated images

---

## 🧩 Requirements

Install dependencies before running:

```bash
pip install torch torchvision tensorboard

