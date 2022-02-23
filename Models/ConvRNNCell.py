import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from einops import rearrange


import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset

class ConvRNNEncoderCell(nn.Module):

    def __init__(self, hidden_size=512):

        super(ConvRNNEncoderCell, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, 5),
            nn.GELU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, 5),
            nn.GELU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 1, 5),
            nn.GELU(),
            nn.BatchNorm2d(1)

        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(52 * 52 + hidden_size, hidden_size),
            nn.Tanh()
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 128)
        )

    def forward(self, x, hidden_state=None):
        x = x.to('cuda')
        x = x.unsqueeze(0)
        x = self.conv_layer(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        
        x = torch.cat([x, hidden_state], dim=1)

        hidden_state = self.hidden_layer(x)

        x = self.linear_layer(hidden_state)

        return x, hidden_state


class ConvRNNDecoderCell(nn.Module):

    def __init__(self, hidden_size=128):

        super(ConvRNNDecoderCell, self).__init__()

        self.hidden_size = hidden_size

        self.hidden_layer = nn.Sequential(
            nn.Linear(128 + hidden_size, 128),
            nn.Tanh()
        )

        self.linear_expand_layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 52 * 52)
        )

        self.conv_expand_layer = nn.Sequential(
            nn.ConvTranspose2d(1, 3, 5),
            nn.GELU(),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, 5),
            nn.GELU(),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 1, 5),
            nn.Softmax2d()
        )

    def forward(self, x, hidden_state=None):
        x = x.to('cuda')
        x = torch.cat([x, hidden_state], dim=1)
        hidden_state = self.hidden_layer(x)

        x = self.linear_expand_layer(hidden_state)
        # print(x.shape)
        x = rearrange(x, 'b (h w) -> b h w', h=52)
        x = x.unsqueeze(dim=1)

        x = self.conv_expand_layer(x)

        return x, hidden_state




if __name__ == '__main__':

    encoder_cell = ConvRNNEncoderCell(hidden_size=54 * 54)
    decoder_cell = ConvRNNDecoderCell()

    data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'

    dataset = MovingMNISTDataset(root_dir=data_path,
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5))
                                    ])
                                )

    loader =  DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)

    train, label = next(iter(loader))

    encoded_output, e_hidden_state = encoder_cell(train)

    # decoded_output, d_hidden_state = decoder_cell(encoded_output)
    
    print(encoded_output.shape)
