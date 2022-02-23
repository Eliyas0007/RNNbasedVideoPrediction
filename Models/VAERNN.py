import torch
import torch.nn as nn

import sys
sys.path.append(".")
from VAE import models


class VEARNN(nn.Module):

    def __init__(self,  in_channels,
                        latent_size, # also input size for rnn layer
                        hiddin_dims, # this is the depth of VAE
                        rnn_type='LSTM', 
                        hidden_size=512,
                        num_layer=1,
                        batch_first=False,
                        bidirectional=False,
                        ):

        super(VEARNN, self).__init__()


        self.VAE_layer = VanillaVAE(in_channels, latent_size, hiddin_dims)

        if rnn_type is 'LSTM':
            self.rnn_layer = nn.LSTM(latent_size, hidden_size, num_layer, batch_first, bidirectional)
        elif rnn_type is 'RNN':
            self.rnn_layer = nn.RNN()
        elif rnn_type is 'GRU':
            self.rnn_layer = nn.GRU()

        

    def forward(self, x):

        x = self.VAE_layer(x)

        return x


if __name__ == '__main__':
    
    model = VEARNN(1, 128, 3, )

    image = torch.random(10, 1, 64, 64)

    pred = model(image)

    print(pred.shape)