# WGAN (Wasserstein GAN) — PyTorch

This repository implements a **Wasserstein Generative Adversarial Network (WGAN)** using **PyTorch**, trained on the **MNIST** dataset.  
The implementation follows the original WGAN paper, using **weight clipping** to enforce the Lipschitz constraint.

---

## 🧠 Overview

The **Wasserstein GAN (WGAN)** improves the training stability of GANs by using the **Wasserstein (Earth-Mover) distance** as a measure of similarity between real and generated distributions.  
Instead of using a discriminator that classifies real/fake, WGAN uses a **critic** that scores how real an image looks.

### Key Idea

- **Critic loss:** maximize \( E[D(x)] - E[D(G(z))] \)
- **Generator loss:** minimize \( -E[D(G(z))] \)
- Enforce **Lipschitz continuity** using **weight clipping** instead of gradient penalty.

---

## 🚀 Features

- Fully convolutional **Generator** and **Critic** (based on DCGAN)
- Wasserstein loss with **weight clipping**
- Trains on **MNIST** (or **CelebA** with a single line change)
- Logs training results with **TensorBoard**
- Saves trained models (`generator.pth` and `discriminator.pth`)

---

## 🧩 Requirements

Install the required libraries:

```bash
pip install torch torchvision tensorboard
