# DDPM Implementation
Date: Early May, 2025

# Setup
pip install -r requirements.txt

# Project Structure
- CIFAR-10
  - ddpm implementation with CIFAR10 dataset with cosine scheduling
  - cdpm implementation with cosine scheduling
- MNIST
  - ddpm implementation with unet thats downscaled from the ddpm paper with both noise schedulers 
  - cdpm implementation with default linear beta scheduling
- Naive Model MNIST
  - Naive Diffusion implementation(CDPM) with MNIST dataset with experiments to have the model to learn $x_0$ instead of $\epsilon_\theta$ and calculate loss from $\^{x_0}$ and $x_0$
  - DDPm with same backbone model

# My Naive Implementation (Clean-DPM) vs DDPM

When I first submitted my project for CS 445, I didn't have a comparison between DDPM and my implementaion(I've been calling it CDPM for now) at all because I assumed my implementation will lead to gradient explosion and fail. 

Now I have ran the model of ddpm, And it seems like having the model **predict the clean image** and computing the loss function with the $\hat{x_0}$ and $x_0$ have resulted in **slightly better generated image quality**. However, there is seems to be fundamental flaws on the diffusion model (based on what yann said and my ituition) so I'm hesitant diving deep into the field of architecture/training strategies.



You can compare the sampled images in [My mnist ddpm notebook](mnist/DDPM(MNIST).ipynb) and [My mnist cdpm notebook](mnist/CDPM(MNIST).ipynb) and you can see that the quality of the cdpm is generally better.

The model also seems to converge better with a smaller model [naive model ddpm](NaiveModelMNIST/DDPM%20(Naive%20Model).ipynb) vs. [naive model cdpm](NaiveModelMNIST/CDPM%20(Naive%20Model).ipynb)

Also the [CIRFAR10 ddpm notebook](cifar10/DDPM.ipynb) vs. [CIFAR10 cdpm notebook](cifar10/CDPM.ipynb) seems to produce better/more natural looking images(?)


~~Even though conceptually predicting for $x_0$ and $\epsilon_\theta$ are theoretically identical, trying to predict for some things while having little to no context (when T is high) as well as a small network is not optimal~~
No conclusions yet as my conclusion have changed 3 times in the past 2 days.

# Sources used
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) 
- [Understanding Diffusion Models: A Unified Perspective (Luo, 2022)](https://arxiv.org/abs/2208.11970)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [LabML.ai Tutorial on DDPM](https://nn.labml.ai/diffusion/ddpm/index.html)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)