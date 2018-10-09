# VAE - Creation Of A Variational Auto Encoder For Image Compression And Generation

This is the final exam project I wrote for the course in Image Analysis for Applied Physics. It consists in the creation of a Variational Auto Encoder which compress images in a features vector thanks to a convolutional neural network (encoder).
The image can then be reconstructed thanks to another neural network (decoder). Both parts are trained together, so the VAE is an "adversarial model".

## The Project

The project is divided into 5 main files:

* header.py: it contains all the imports and definitions of some parameters, the dataset class and encoder and decoder functions;

* project.py: it contains the loss function, the optimizer and the code used to train the neural networks;

* loader.py & loader2.py: are used to load pre-trained neural networks to generate original images into "generated/" and "comparison/".

* visualize.ipynb: it allows to manually modify the latent vector provided to the decoder in order to visualize the effects of different features on final images.

### The Convolutional Neural Network

![Encoder structure](https://github.com/tommaso1311/VAE/blob/master/encodercor.png)

![Decoder structure](https://github.com/tommaso1311/VAE/blob/master/decodercor.png)