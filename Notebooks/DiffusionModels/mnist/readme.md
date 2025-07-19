# My Naive Implementation (Clean-DPM) vs DDPM

When I first submitted my project for CS 445, I didn't have a comparison between DDPM and my implementaion(ive been calling it CDPM for now) at all because I assumed my implementation will lead to gradient explosion and fail. 

Now I have ran the model of ddpm, And it seems like having the model **predict the clean image** and computing the loss function with the $\^{x_0}$ and $x_0$ have resulted in **better generated image quality**. However, there is still fundamental flaws on the diffusion model (based on what yann said and my ituition so I will not dive down into it further).

You can compare the sampled images in [My ddpm notebook](DDPM(MNIST).ipynb) and [My cdpm notebook](C-DPM%20(MNIST).ipynb) and you can clearly see that the quality of the cdpm is generally better.

~~Note: It seems it only generate better when the model is **too capable** for the data (which negates the effect of potential gradient exploding)~~
~~Even though conceptually predicting for $x_0$ and $\epsilon_\theta$ are theoretically identical, trying to predict for some things while having little to no context (when T is high) as well as a small network is not optimal~~
No conclusions yet as my conclusion have changed 3 times in the past 2 days.