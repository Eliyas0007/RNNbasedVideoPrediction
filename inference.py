import sys
sys.path.append('.')

import cv2
import torch
import random
import torchvision.datasets as datasets
import Models.AutoEncoder as AutoEncoder
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import DataLoader
from PytorchVAE.models.vanilla_vae import VanillaVAE
from Models.VAESeq2Seq import VAESeq2Seq, Encoder, Decoder
from DataLoader.MovingMnistDataset import MovingMNISTDataset

def tensor2image(tensor):
    return rearrange(tensor.cpu().detach().numpy(), 'c h w -> h w c')

# Random seed
# random.seed(333)
# torch.manual_seed(333)

# Device
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# Model path
ae_model_path = 'workingModels/autoencoder_step18000.pth'
fine_tuned_model_path = 'SavedModels/fine_tuned_autoencoder_step700.pth'
vae_model_path = 'SavedModels/vae_step6200.pth'
vaelstm_model_path = 'SavedModels/aelstm_step77000.pth'

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
ae.load_state_dict(torch.load(fine_tuned_model_path))
ae.to(device)

# LSTM 
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


mnist_dataset = datasets.MNIST(root='MNISTDATA/', train=False, download=True, transform=transforms.Compose([
                                    # transforms.Normalize((0.5), (0.5)),
                                    transforms.ToTensor(),
                                    transforms.Resize((64, 64))
                                    ]))
mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True)
image, _ = next(iter(mnist_loader))
print(video.shape)

for b in range(1):
    image = ae(video[b])

    for f in range(10):
        cv2.imshow('AE', tensor2image(image[f]))
        cv2.waitKey(300)

    for f in range(10):
        cv2.imshow('ORIGINAL', tensor2image(video[b][f]))
        cv2.waitKey(300)

# # print(video.shape, target[0].shape)

# video = video.to(device)
# target = target.to(device)
# # Encode image to latent using trained VAE
# batch, frame_len, c, h, w = video.shape
# encoded_train = torch.empty(batch, frame_len, latent_size).to(device)
# encoded_target = torch.empty(batch, frame_len, latent_size).to(device)
# for b in range(batch):
#     for f in range(frame_len):
#         # train_frame_latent = vae(video[b][f].unsqueeze(0))
#         # target_frame_latent = vae(target[b][f].unsqueeze(0))

#         train_frame_latent = ae.encoder(video[b][f].unsqueeze(0))
#         target_frame_latent = ae.encoder(target[b][f].unsqueeze(0))

#         # infer_image = decoded_image[0]
#         # infer_image = rearrange(infer_image.cpu().detach().numpy(), 'c h w -> h w c')

#         # cv2.imshow('Inferece', infer_image)

#         # cv2.waitKey(200)

#         # reparameterize
# #         std = torch.exp(0.5 * train_frame_latent[2])
# #         eps = torch.randn_like(train_frame_latent[3])
# #         latent_tr =  eps * std + train_frame_latent[2]
# #         std = torch.exp(0.5 * target_frame_latent[2])
# #         eps = torch.randn_like(target_frame_latent[3])
# #         latent_ta =  eps * std + target_frame_latent[2]
# #         encoded_train[b][f] = latent_tr
# #         encoded_target[b][f] = latent_ta

#         encoded_train[b][f] = train_frame_latent
#         encoded_target[b][f] = target_frame_latent

# encoded_train = rearrange(encoded_train, 'b f l -> f b l')
# encoded_target = rearrange(encoded_target, 'b f l -> f b l')

# # 10 means it will predict 100 frames, (r * 10)
# range_of_prediction = 100

# for r in range(range_of_prediction):

#     if r == 0:
#         # first 10 frames
#         pred = seq2seq(encoded_train)

#         pred_for_infer = rearrange(pred, 'f b l -> b f l')
#         pred_for_infer = pred_for_infer.unsqueeze(0).to(device)
#         inference = ae.decoder(pred_for_infer)
#         print(inference.shape)
#         predicted_video = torch.cat([ae(video[0]), ae(target[0]), inference])
#     else:
#         pred = seq2seq(pred)

#         pred_for_infer = rearrange(pred, 'f b l -> b f l')
#         inference = ae.decoder(pred_for_infer)
        
#         predicted_video = torch.cat([predicted_video, inference])


# original_decoded = ae.decoder(encoded_train)

# original_video = torch.cat([video.squeeze(0), target.squeeze(0)])

# for b in range(len(predicted_video)):

#     # target_image = original_video[b]
#     infer_image = predicted_video[b]

#     # print(infer_image.shape)
#     # target_image = rearrange(target_image.cpu().detach().numpy(), 'c h w -> h w c')
#     infer_image = rearrange(infer_image.cpu().detach().numpy(), 'c h w -> h w c')

#     # Hori = numpy.concatenate((target_image, infer_image), axis=1)

#     cv2.imshow('Inferece', infer_image)
#     infer_image *= 256
#     cv2.imwrite(f'inferImages/inference{b}.png', infer_image)

#     cv2.waitKey(125)
