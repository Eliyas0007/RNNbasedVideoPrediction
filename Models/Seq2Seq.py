import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, batch_size):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.encoder = self.encoder.to(device)
        self.encoder_init_hidden_state = torch.zeros(1, 54 * 54).to(device)

        self.decoder = decoder
        self.decoder = self.decoder.to(device)
        self.decoder_init_hidden_state = torch.zeros(batch_size, 128).to(device)
        

    def forward(self, source_videos, target_videos=None, teacher_force_ratio=0.5):
        
        batch_size, sequence_len, channel, height, width = source_videos.shape
        encoder_hidden_state = None
        decoder_hidden_state = None
        latent_vector = None
        latent_vector_batch = torch.empty(batch_size, 128)
        predicted_frames = torch.empty(batch_size, sequence_len, channel, height, width)

        for i, video in enumerate(source_videos):
            for j, frame in enumerate(video):
                if encoder_hidden_state is None:
                    encoder_hidden_state = self.encoder_init_hidden_state

                latent_vector, encoder_hidden_state = self.encoder(frame, encoder_hidden_state)
            latent_vector_batch[i] = latent_vector

            for i in range(sequence_len):
                if decoder_hidden_state is None:
                    decoder_hidden_state = self.decoder_init_hidden_state
                image, decoder_hidden_state = self.decoder(latent_vector_batch, decoder_hidden_state)
                
                for b in range(batch_size):
                    predicted_frames[b][i] = image[b]          

        return predicted_frames
