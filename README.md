# 🖌 Image Colorization Project 1.0

## ℹ️ Overview

In this project, we tackle the task of coloring a black and white image in a realistic way, with the aim of being able to color historical images and videos. We framed the problem as a regression problem, so we converted each image from RGB to LAB color space, and given the L (luminance) channel, for each pixel we tried to predict the A and B channel values, and then reconstructed the image. <br>
To perform this task, we designed and implemented different Convolutional Neural Network architectures, taking into account both the problem's characheristics and the limited computational resources available to us. We implemented our models in Tensorflow and used the MSE loss for the regression problem.
The results on the MirFlickr25k dataset were quite unsatisfactory, mainly because the subjects presented in the pictures were very different, so our simple model struggled to learn any meaningful way of coloring them, and simply returned images almost fully colored with a sepia-brown color. We found that this is a common problem in the literature, which is often solved by framing the problem as a classification problem (see 'Colorful Image Colorization', Zhang et al. 2016), or by exploiting deeper, more powerful networks. Unfortunately, our limited computational resources made both approaches infeasible, so we tried to solve the problem on a simpler dataset, namely the Stanford Dogs Dataset.
Indeed, the results on this more uniform dataset were much more satisfactory, and we were able to obtain meaningful colorization of the black and white pictures of golden retrievers.


### ✍️ Authors

Alessandro Ardenghi, Leonardo Saveri, Andrea Zoccante



