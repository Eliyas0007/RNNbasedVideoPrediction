import sys
from unittest import FunctionTestCase
sys.path.append('.')

import cv2
import numpy
import torch
import random
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import matplotlib.animation as animation
import Models.AutoEncoder as AutoEncoder
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import DataLoader
from PytorchVAE.models.vanilla_vae import VanillaVAE
from Models.VAESeq2Seq import VAESeq2Seq, Encoder, Decoder
from DataLoader.MovingMnistDataset import MovingMNISTDataset


def tensor2image(tensor):
    return rearrange(tensor.cpu().detach().numpy(), 'c h w -> h w c') * 256


def animate(index):
    ...
    y = encoded_train.squeeze(1).detach().numpy()[index]
    plt.cla()
    plt.plot(y, x)
    plt.xlim([-10, 10])
    plt.ylim([-1, 130])
    plt.tight_layout()

# Random seed
random.seed(333)
torch.manual_seed(333)

# plt.style.use('seaborn-poster')
# plt.style.use('seaborn-paper')
plt.style.use('fivethirtyeight')

# Device
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# Model path
ae_model_path = 'workingModels/autoencoder_step18000.pth'
# fine_tuned_model_path = 'SavedModels/fine_tuned_autoencoder_withvideodata__step1500.pth'
# fine_tuned_model_path = 'SavedModels/autoencoder_step18000.pth'
vae_model_path = 'SavedModels/vae_step6200.pth'
vaelstm_model_path = 'SavedModels/vaelstm_step7800.pth'
aelstm_model_path = 'SavedModels/aelstm_step18000.pth'

# variational autoencoder
in_channels = 1
latent_size = 128
vae = VanillaVAE(in_channels, latent_size)
vae.load_state_dict(torch.load(vae_model_path))
vae.to(device)

# simple autoencoder
ae_encoder = AutoEncoder.Encoder(in_channels, latent_size)
ae_decoder = AutoEncoder.Decoder(latent_size)
ae = AutoEncoder.AutoEncoder(ae_encoder, ae_decoder)
ae.load_state_dict(torch.load(ae_model_path))
ae.to(device)

# LSTM 
input_size = latent_size
hidden_size = 1024
num_layers = 2
dropout = 0.5
encoder = Encoder(input_size, hidden_size, num_layers, dropout)
decoder = Decoder(input_size, hidden_size, num_layers, dropout)
seq2seq = VAESeq2Seq(encoder, decoder, device).to(device)
seq2seq.load_state_dict(torch.load(aelstm_model_path))


# MovingMNIST
data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
dataset = MovingMNISTDataset(root_dir=data_path, load_type='video',
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    # transforms.ToTensor()
                                    ])
                            )
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
video, target = next(iter(loader))

# MNIST
mnist_dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    # transforms.Normalize((0.5), (0.5)),
                                    transforms.ToTensor(),
                                    transforms.Resize((64, 64))
                                    ]))
mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)
image, _ = next(iter(mnist_loader))

# print(video.shape, target[0].shape)

video = video.to(device)
target = target.to(device)
# Encode image to latent using trained VAE
batch, frame_len, c, h, w = video.shape
encoded_train = torch.empty(batch, frame_len, latent_size).to(device)
encoded_target = torch.empty(batch, frame_len, latent_size).to(device)
for b in range(batch):
    for f in range(frame_len):

        '''
        for VAE
        '''
        # train_frame_latent = vae(video[b][f].unsqueeze(0))
        # target_frame_latent = vae(target[b][f].unsqueeze(0))

        # # infer_image = decoded_image[0]
        # # infer_image = rearrange(infer_image.cpu().detach().numpy(), 'c h w -> h w c')

        # # cv2.imshow('Inferece', infer_image)

        # # cv2.waitKey(200)

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
        for AE
        '''
        train_frame_latent = ae.encoder(video[b][f].unsqueeze(0))
        target_frame_latent = ae.encoder(target[b][f].unsqueeze(0))

        encoded_train[b][f] = train_frame_latent
        encoded_target[b][f] = target_frame_latent

encoded_train = rearrange(encoded_train, 'b f l -> f b l')
encoded_target = rearrange(encoded_target, 'b f l -> f b l')





ani = animation.FuncAnimation(plt.gcf(), animate, interval=200)



x = numpy.array([x for x in range(128)])
y = encoded_train.squeeze(1).detach().numpy()[0]
# y = rearrange(y, 'x y -> y x')[0]
print(x.shape, y.shape)

plt.show()