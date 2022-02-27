# RNN based Video Prediction (Unfinished Project)

In this repo we have 2 models for simple video prediction using recurrent networks.

### TYPES

#### RNNs(Vanilla-RNN, LSTM, GRU) + Conv2d

In this model, we firstly compress an image to a 2d latent vector using CNN and appliy 3 types of RNN to predict next 10 frames.

**Tested**

- Vanilla RNN + Conv2d (Failed): Loss is unable to converge.

**Implementing**

- LSTM + Conv2d
- GRU + Conv2d

#### Vanilla VAE + RNNs

In this model, we use VEA to encode 2d image into 1d latent vector, then use the latent vector to feed RNN to predic next n frames.
