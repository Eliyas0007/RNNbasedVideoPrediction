import imp
from turtle import forward
import torch
import torch.nn as nn
import ConvRNNCell


import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset



class ConvRNN(nn.Module):

    def __init__(self):
        super(ConvRNN, self).__init__()

    def forward(self, x, hidden_state):
        return output, hidden_state