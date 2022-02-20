import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset

class ConvRNNCell(nn.Module):

    def __init__(self):
        super(ConvRNNCell, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, 5),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5),
            nn.ReLU(),
        )

    def forward(self, x, hidden_state=None):
        x = self.conv_layer(x)
        return x

    def get_initial_hiddent_state(self, frame_size):
        return torch.zeros(frame_size, frame_size)

if __name__ == '__main__':

    model = ConvRNNCell()
    data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'

    dataset = MovingMNISTDataset(root_dir=data_path,
                                # transform=transforms.Compose([
                                #     transforms.ToTensor(),
                                #     transforms.Normalize((0.5), (0.5))
                                    # ])
                                )

    loader =  DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)

    train, label = next(iter(loader))
    print(train.shape)
    # print(type(train))
    # output = model(train[0][0].unsqueeze(dim=0).unsqueeze(dim=0))
    # print(output.shape)
