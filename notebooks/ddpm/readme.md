# DDPM Implementation

# Setup
pip install -r requirements.txt

# Project Structure
- DDPM-CIFAR10
  - ddpm implementation with CIFAR10 dataset
- DDPM-MNIST
  - ddpm implementation with MNIST dataset with both noise schedulers
- Naive Diffusion
  - Naive Diffusion implementation with MNIST dataset with experiments to have the model to learn $x_0$ instead of $\epsilon_\theta$


# Sources used
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) 
- [Understanding Diffusion Models: A Unified Perspective (Luo, 2022)](https://arxiv.org/abs/2208.11970)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [LabML.ai Tutorial on DDPM](https://nn.labml.ai/diffusion/ddpm/index.html)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)