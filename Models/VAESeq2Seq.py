import torch
import torch.nn as nn
 
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, drop_out):

        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop_out)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_out, batch_first=True)

    def forward(self, x):
        
        x = self.dropout(x)

        output, (hidden, cell) = self.lstm(x)

        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, drop_out):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop_out)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_out, batch_first=True)
        

    def forward(self, x, hidden, cell):

        x = x.unsqueeze(0)
        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        return output, (hidden, cell)


class VAESeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(VAESeq2Seq, self).__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)



    def forward(self, batch_of_encoded_videos): # shape of [batch_size, squence_len, latent_szie]

        hidden, cell = self.encoder(batch_of_encoded_videos)

        batch_size, sequence_len, latent_size = batch_of_encoded_videos.shape
        predicted_latnets = torch.empty(batch_size, sequence_len, latent_size)

        for batch in range(batch_size):
            for t in range(sequence_len):
                output, (hidden, cell) = self.decoder(hidden, cell)
                predicted_latnets[batch][t] = output
                

        return predicted_latnets

