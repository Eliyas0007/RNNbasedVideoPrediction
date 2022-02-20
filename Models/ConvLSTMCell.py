import cv2
import numpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset

class ConvLSTMCell(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                            padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels + out_channels, 
                                    out_channels=4 * out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

        self.input_weight = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.output_weight = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.forget_weight = nn.Parameter(torch.Tensor(out_channels, *frame_size))
    

    def forward(self, input, hidden_state, cell_state):
        
        # previous hidden state concatonates input and feed in to conv
        conv_output = self.conv_layer(torch.cat([input, hidden_state], dim=1))
        
        # split the out put to different chunks
        input_conv, forget_conv, cell_conv, output_conv = torch.chunk(conv_output, chunks=4, dim=1)

        # intput * cell 
        input_gate = torch.sigmoid(input_conv + self.input_weight * cell_state)

        forget_gate = torch.sigmoid(forget_conv + self.forget_weight * cell_state)

        current_cell_state = forget_gate * cell_state + input_gate * nn.Tanh(cell_conv)

        output_gate = torch.sigmoid(output_conv + self.output_weight * current_cell_state)

        current_hidden_state = output_gate * nn.Tanh(output_gate)

        return current_hidden_state, current_cell_state


if __name__ == "__main__":
    
    data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
    datas = numpy.load(data_path)
    datas = rearrange(datas, 'f b w h -> b f w h')

    dataset = MovingMNISTDataset(root_dir=data_path,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                    ]))
    loader =  DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)
    train, label = next(iter(loader))
    print(train.shape, label.shape)

    # for data in datas:
    #     for frame in data:
    #     # print(frame.shape)
    #         cv2.imshow('asd', frame)
    #         cv2.waitKey(100)

    
    # print(data[0].shape)

    # convlstmcell = ConvLSTMCell()