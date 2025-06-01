# Naive Diffusion (predict $x_0$) Implementation

This is a diffusion model implementation with the idea of having the model predicting for the clean image $x_0$ instead of $\epsilon_\theta$. Only a small UNet is used, therefore the model is not able to converge to a ideal loss.

However, there are multiple things I have learend while implementing this approach:
- Predicting for $x_0$ is more likely to cause a exploding gradient problem (Maybe? just my intuition)
  
Theory of the model not performing well:
- Small model
- "While the linear noise schedule used in Ho et al. (2020) worked well for high-resolution images, it was sub-optimal for images of resolution 64×64 and 32×32. In particular, the end of the forward noising process is too noisy, and so doesn’t contribute very much to sample quality." Nichol & Dhariwal (2021) (Maybe not [My cosine schedule notebook](../mnist/DDPM(Cosine%20schedule).ipynb))