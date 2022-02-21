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
            nn.ReLU(),
            nn.Conv2d(3, 3, 5),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(54 * 54 + hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x, hidden_state=None):

        
        x = self.conv_layer(x)
        x = torch.flatten(x, start_dim=1)
        batch_size, latent_size = x.shape

        if hidden_state is None:
            hidden_state = self.get_initial_hiddent_state(batch_size, latent_size)

        x = torch.cat([x, hidden_state], dim=1)
        x = self.linear_layer(x)

        return x

    def get_initial_hiddent_state(self, batch_size, latent_size):
        return torch.zeros(batch_size, latent_size)


class ConvRNNDecoderCell(nn.Module):

    def __init__(self, hidden_size=128):

        super(ConvRNNDecoderCell, self).__init__()

        self.hidden_size = hidden_size

        self.linear_expand_layer = nn.Sequential(
            nn.Linear(128 + hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 54 * 54)
        )

        self.conv_expand_layer = nn.Sequential(
            nn.ConvTranspose2d(1, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 5),
            nn.ReLU(),
        )

    def forward(self, x, hidden_state=None):

        if hidden_state is None:
            batch_size, latent_size = x.shape
            hidden_state = torch.zeros(batch_size, latent_size)

        x = torch.cat([x, hidden_state], dim=1)

        x = self.linear_expand_layer(x)
        x = rearrange(x, 'b (h w) -> b h w', h=54)
        x = x.unsqueeze(dim=1)

        x = self.conv_expand_layer(x)

        return x, hidden_state




if __name__ == '__main__':

    encoder_cell = ConvRNNEncoderCell(hidden_size=54 * 54)
    decoder_cell = ConvRNNDecoderCell()



    data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'

    dataset = MovingMNISTDataset(root_dir=data_path,
                                transform=transforms.Compose([
                                    # transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
                                )

    loader =  DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)

    train, label = next(iter(loader))

    encoded_output = encoder_cell(train[0])
    decoded_output = decoder_cell(encoded_output)
    print(encoded_output.shape)
