import cv2
import torch
import torch.nn as nn

from einops import rearrange

import sys
sys.path.append("/home/yiliyasi/Documents/Projects/RNNbasedVideoPrediction")

from PytorchVAE.models.vanilla_vae import VanillaVAE

class VEARNN(nn.Module):

    def __init__(self,  in_channels,
                        latent_size, # also input size for rnn layer
                        rnn_type='LSTM', 
                        hidden_size=512,
                        num_layer=1,
                        batch_first=False,
                        bidirectional=False,
                        ):

        super(VEARNN, self).__init__()


        self.VAE_layer = VanillaVAE(in_channels, latent_size)

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
    
    model = VEARNN(1, 128)

    image = torch.rand(1, 1, 64, 64)

    pred = model(image)[0][0].unsqueeze(0)
    # pred = rearrange(pred, 'h w c -> c h w')


    print(pred.shape)
    # print(type(pred.cpu().detach().numpy()))

    # cv2.imwrite('hehe.png', pred.cpu().detach().numpy())

