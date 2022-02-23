# RNN based Video Prediction

In this repo we have 2 models for simple video prediction using basic recurrent networks, back bone is convolution and recurrent network. First models uses ConvLSTM cells and the other one uses a latent compressed by a VAE.

### TYPES

- **RNNs(Vanilla-RNN, LSTM, GRU) + Conv2d (done, but won't work)**

In this model, we firstly compress an image to a 2d latent vector using CNN and use RNN to predict next 10 frames

- **Vanilla VAE + RNNs**

In this model, we use VEA to encode 2d image into 1d latent vector, then use the latent vector to feed RNN to predic next n frames.