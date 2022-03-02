# RNN based Video Prediction (Unfinished Project)

In this repo we have 2 models for simple video prediction using recurrent networks.

## Tested Environment

- Python 3.7.11
- opencv 4.5.5.62
- torch 1.10.1
- torchvision 0.11.2

## Dataset
### Moving-Mnist Dataset
<p align="center">
    <img width="200" src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000001.gif" alt="Moving-Mnist Dataset Example">
</p>

Source: http://www.cs.toronto.edu/~nitish/unsupervised_video/

Contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame.

## Type of models

### RNNs(Vanilla-RNN, LSTM, GRU) + Conv2d

In this model, we firstly compress an image to a 2d latent matrix using CNN and appliy 3 types of RNN to predict next 10 frames.

#### Tested

- Vanilla RNN + Conv2d (Failed): Loss is unable to converge.

#### Implementing

- LSTM + Conv2d
- GRU + Conv2d

### Vanilla VAE + RNNs

In this model, we use VEA to encode 2d image into 1d latent vector, then use the latent vector to feed RNN to predic next n frames.

#### Tested

- Vanilla VAE is trained using moving mnist and it is able to generate good quality images
- VAE + LSTM is able to predict well generalized future 10 frames

#### Result

First 10 frames is used to predict next 10 frames

Ground truth is on the left and predicted sequence is on the right. Note that previous 10 frames is identical, comparison is made using last 10 frames.

<p align="center">
    <img width="200" src="https://github.com/Eliyas0007/RNNbasedVideoPrediction/blob/main/images/movingmnistprediction.gif" alt="Moving-Mnist Dataset Example"> 
</p>

