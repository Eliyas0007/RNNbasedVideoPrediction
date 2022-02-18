import torch
import torch.nn as nn

import sys
sys.path.append(".")
from Models.ConvLSTMCell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, frame_size):

        super(ConvLSTM, self).__init__()
       
        # self.num_layers = num_layers
        self.out_channels = out_channels

        # self.dropout = nn.Dropout(dropout)
        self.ConvLSTMCell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, frame_size)

    def forward(self, input_frame):

        batch_size, seq_len, height, width = input_frame.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
                        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

        output, (hidden, cell) = self.rnn(ConvLSTMCell, (h_state, c_state))

        return hidden, cell