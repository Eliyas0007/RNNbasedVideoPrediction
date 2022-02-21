# RNN based Video Prediction

In this repo we have 3 models for simple video prediction using basic recurrent networks, back bone is convolution and recurrent network. Each of RNN block contains a Convolutional network, and dataset we use is moving-MNIST

### TYPES

- RNN + Conv2d (Almost done)
in this model, we firstly compress an image to a 2d latent vector using CNN and use RNN to predict next 10 frames

- LSTM + Conv2d
- GRU + Conv2d