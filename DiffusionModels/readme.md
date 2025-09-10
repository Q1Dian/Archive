# DDPM Implementation
Date: Early May, 2025

I came to aware that this is not the best format for research... I will use configs next time...

# Setup
pip install -r requirements.txt

# Project Structure

- [DDPM](ddpm.ipynb)
  - DDPM implementation with unet thats downscaled from the ddpm paper with linear beta scheduler
- [DDPM with Cosine Scheduler](ddpm-cosine-scheduler.ipynb) 
  - A paper mentioned that cosine scheduler could provide better generation quality so i tried it
- [Diffusion model that predicts for clean image rather than noise](dm-predicts-x0.ipynb)
  - Diffusion implementation for intuition building
  - (Unexptected results)
- Other Tests
  - Naive Model MNIST
    - Naive Diffusion implementation which have the model to learn $x_0$ instead of $\epsilon_\theta$ and calculate loss from $\^{x_0}$ and $x_0$
    - DDPm with same small backbone model
    - (Surprisingly, the naive model predicting $x_0$ showed signs of convergence and generated recognizable outputs while ddpm did not)
    - (I later came to aware that this might be a hyperparameter tuning issue, as step goes high, all ddpm approach is just learning how to go from one gaussian noise to another)
  - CIFAR-10
    - ddpm implementation with CIFAR10 dataset with cosine scheduling
    - diffusion (predict $x_0$) implementation with cosine scheduling
    - ddpm outperforms the version that predicts $x_0$ and I suspect the quality of clean prediction will degrade as the data complexity increases 

# Takeaway from the project
- Cool project, but I cold emailed two professors and showed no empirical evidence of improvements or findings. Probably should not have done that next time. 
- Observations and others thoughts:
  - Diffusion is a cool concept that demonstrates the unique perspective of a well designed training strategy based on ELBO and KL-Divergence
  - If you want to style transfer from one image to another or modify some properties of the image (facial accessories etc.) would you corrupt the image by eg. 300 steps and then denoise it? (how do you make sure no important signals are lost?) (there's probably a paper somewhere)


# Sources used
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) 
- [Understanding Diffusion Models: A Unified Perspective (Luo, 2022)](https://arxiv.org/abs/2208.11970)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
- [LabML.ai Tutorial on DDPM](https://nn.labml.ai/diffusion/ddpm/index.html)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)