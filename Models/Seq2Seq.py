import torch
import torch.nn as nn

import sys
sys.path.append(".")
from ConvRNNCell import ConvRNNEncoderCell, ConvRNNDecoderCell

class Seq2Seq(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
