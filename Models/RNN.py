import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x, h_state, c_state):

        
        output, (hidden, cell) = self.rnn(x, (h_state, c_state))

        return hidden, cell