# RNN based Video Prediction

In this repo we have 2 models for simple video prediction using basic recurrent networks, back bone is convolution and recurrent network. First type of models uses Conv2d and different RNN layers and the other one uses a latent vector compressed by a VAE and use latent vector to train an RNN then use the same VAE to decode predicted latent vector to 2d image.

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
