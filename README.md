# Hyperspectral ImageReconstruction with GAN

Based on the Image Super-Resolution GAN https://arxiv.org/pdf/1609.04802.pdf We implemented a 
GAN to create an end-to-end optimization framework for hyperspectral image reconstruction with a custom layer that creates a convolution between the hyperspectral 
image and a group of PSFs. This structured is used to jointly optimize the DoE and the CCA.
