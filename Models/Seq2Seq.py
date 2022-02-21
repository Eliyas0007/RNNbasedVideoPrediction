import torch
import torch.nn as nn

import sys
sys.path.append(".")
from ConvRNNCell import ConvRNNEncoderCell, ConvRNNDecoderCell

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.encoder_init_hidden_state = torch.zeros(10, 54 * 54)

        self.decoder = decoder
        self.decoder_init_hidden_state = torch.zeros(10, 128)
        

    def forward(self, source_videos, target_videos, teacher_force_ratio=0.5):
        
        batch_size, sequence_len, channel, height, width = source_videos.shape
        encoder_hidden_state = None
        decoder_hidden_state = None
        latent_vector = None
        predicted_frames = torch.empty(batch_size, sequence_len, channel, height, width)

        for i, video in enumerate(source_videos):
            for j, frames in enumerate(video):

                if encoder_hidden_state is None:
                    encoder_hidden_state = self.encoder_init_hidden_state

                latent_vector, encoder_hidden_state = self.encoder(frames, encoder_hidden_state)

            for i in range(sequence_len):

                if decoder_hidden_state is None:
                    decoder_hidden_state = self.decoder_init_hidden_state
                image, _ = self.decoder(latent_vector, decoder_hidden_state)
                predicted_frames[i][j] = image

        return predicted_frames
