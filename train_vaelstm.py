import sys
sys.path.append("/home/yiliyasi/Documents/Projects/RNNbasedVideoPrediction")

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import Models.AutoEncoder as AutoEncoder
import torchvision.transforms as transforms

# from tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from PytorchVAE.models.vanilla_vae import VanillaVAE
from Models.VAESeq2Seq import VAESeq2Seq, Encoder, Decoder
from DataLoader.MovingMnistDataset import MovingMNISTDataset


print('Initializing...')

# save path for model
model_save_path = './SavedModels/'

# Hyperparameters
num_epochs = 50

learning_rate = 0.001

batch_size = 64

# General settings
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f'aelstm_runs/')
step = 0

# VAE hyperparameters
model_path = 'SavedModels/vae_step6200.pth'
in_channels = 1
latent_size = 128
vae = VanillaVAE(in_channels, latent_size)
vae.load_state_dict(torch.load(model_path))
vae = vae.to(device)

# simple autoencoder
ae_model_path = 'SavedModels/autoencoder_step16000.pth'
ae_encoder = AutoEncoder.Encoder(in_channels, latent_size)
ae_decoder = AutoEncoder.Decoder(latent_size)
ae = AutoEncoder.AutoEncoder(ae_encoder, ae_decoder)
ae.load_state_dict(torch.load(ae_model_path))
ae.to(device)


# Seq2Seq model and hype parameters
input_size = latent_size
hidden_size = 1024
num_layers = 2
dropout = 0.5
encoder = Encoder(input_size, hidden_size, num_layers, dropout)
decoder = Decoder(input_size, hidden_size, num_layers, dropout)
seq2seq = VAESeq2Seq(encoder, decoder, device).to(device)

# Optimizer
optimizer = optim.Adam(seq2seq.parameters(), lr=learning_rate)

# Loss function
criterion = nn.MSELoss()

# Data loading
print('Loading Data')
data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
dataset = MovingMNISTDataset(root_dir=data_path, load_type='video',
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    # transforms.ToTensor()
                                    ])
                            )
loader =  DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print('Data Loaded, Start training..')


# if load_model:
#     model.load_state_dict(torch.load(model_path))
#     print('model state dict loaded')

for epoch in range(num_epochs):

    print(f'Epoch [current: {epoch} / total: {num_epochs}]')

    with tqdm.tqdm(loader, unit='batch') as tepoch:
        for (train, target) in tepoch:

            train = train.to(device)
            target = target.to(device)

            # Encode image to latent using trained VAE
            batch, frame_len, c, h, w = train.shape
            encoded_train = torch.empty(batch, frame_len, latent_size).to(device)
            encoded_target = torch.empty(batch, frame_len, latent_size).to(device)
            for b in range(batch):
                for f in range(frame_len):
                    '''
                    this is for variational autoencoder
                    '''
                    # train_frame_latent = vae(train[b][f].unsqueeze(0))
                    # target_frame_latent = vae(target[b][f].unsqueeze(0))
                    # # reparameterize
                    # std = torch.exp(0.5 * train_frame_latent[2])
                    # eps = torch.randn_like(train_frame_latent[3])
                    # latent_tr =  eps * std + train_frame_latent[2]

                    # std = torch.exp(0.5 * target_frame_latent[2])
                    # eps = torch.randn_like(target_frame_latent[3])
                    # latent_ta =  eps * std + target_frame_latent[2]

                    # encoded_train[b][f] = latent_tr
                    # encoded_target[b][f] = latent_ta

                    '''
                    this is for simple autoencoder
                    '''
                    encoded_train[b][f] = ae.encoder(train[b][f].unsqueeze(0))
                    encoded_target[b][f] = ae.encoder(target[b][f].unsqueeze(0))

            encoded_train = rearrange(encoded_train, 'b f l -> f b l')
            encoded_target = rearrange(encoded_target, 'b f l -> f b l')

            pred = seq2seq(encoded_train)

            optimizer.zero_grad()

            loss = criterion(pred.to(device), encoded_target)
            loss.backward()

            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=1)
            optimizer.step()
            # print(f'Training Loss : {loss.item()}')

            if step % 100 == 0:
                torch.save(seq2seq.state_dict(), model_save_path + f'aelstm_step{step}.pth')

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1
            # break