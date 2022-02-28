import sys
sys.path.append('.')

import cv2
import numpy
import torch
import random
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import DataLoader
from PytorchVAE.models.vanilla_vae import VanillaVAE
from Models.VAESeq2Seq import VAESeq2Seq, Encoder, Decoder
from DataLoader.MovingMnistDataset import MovingMNISTDataset

# Random seed
random.seed(0)
torch.manual_seed(0)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model path
model_path = 'SavedModels/step6200.pth'
vaelstm_model_path = 'SavedModels/vaelstm_step7800.pth'

# Model and its hyperparameters
in_channels = 1
latent_size = 128
vae = VanillaVAE(in_channels, latent_size)
vae.load_state_dict(torch.load(model_path))
vae.to(device)

input_size = latent_size
hidden_size = 1024
num_layers = 2
dropout = 0.5
encoder = Encoder(input_size, hidden_size, num_layers, dropout)
decoder = Decoder(input_size, hidden_size, num_layers, dropout)
seq2seq = VAESeq2Seq(encoder, decoder, device).to(device)
seq2seq.load_state_dict(torch.load(vaelstm_model_path))


# Sample data
data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
dataset = MovingMNISTDataset(root_dir=data_path, load_type='video',
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    # transforms.ToTensor()
                                    ])
                            )
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

video, target = next(iter(loader))

# print(video.shape, target[0].shape)

video = video.to(device)
target = target.to(device)
# Encode image to latent using trained VAE
batch, frame_len, c, h, w = video.shape
encoded_train = torch.empty(batch, frame_len, latent_size).to(device)
encoded_target = torch.empty(batch, frame_len, latent_size).to(device)
for b in range(batch):
    for f in range(frame_len):
        train_frame_latent = vae(video[b][f].unsqueeze(0))
        target_frame_latent = vae(target[b][f].unsqueeze(0))
        # reparameterize
        std = torch.exp(0.5 * train_frame_latent[2])
        eps = torch.randn_like(train_frame_latent[3])
        latent_tr =  eps * std + train_frame_latent[2]
        std = torch.exp(0.5 * target_frame_latent[2])
        eps = torch.randn_like(target_frame_latent[3])
        latent_ta =  eps * std + target_frame_latent[2]
        encoded_train[b][f] = latent_tr
        encoded_target[b][f] = latent_ta

encoded_train = rearrange(encoded_train, 'b f l -> f b l')
encoded_target = rearrange(encoded_target, 'b f l -> f b l')
# print(encoded_train.shape, encoded_target.shape)

pred = seq2seq(encoded_train)
pred = rearrange(pred, 'f b l -> b f l')
pred = pred.unsqueeze(0).to(device)

inference = vae.decode(pred)


for b in range(inference.shape[0]):

    target_image = target[0][b]
    infer_image = inference[b]

    # print(infer_image.shape)
    target_image = rearrange(target_image.cpu().detach().numpy(), 'c h w -> h w c')
    infer_image = rearrange(infer_image.cpu().detach().numpy(), 'c h w -> h w c')

    Hori = numpy.concatenate((target_image, infer_image), axis=1)

    cv2.imshow('target and inferece', Hori)

    cv2.waitKey(300)

