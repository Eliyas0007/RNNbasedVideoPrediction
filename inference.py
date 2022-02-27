import sys
sys.path.append('.')

import cv2
import random
import torch
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import DataLoader
from PytorchVAE.models.vanilla_vae import VanillaVAE
from DataLoader.MovingMnistDataset import MovingMNISTDataset

# Random seed
random.seed(0)
torch.manual_seed(0)

# Model path
model_path = 'SavedModels/step6200.pth'

# Model and its hyperparameters
in_channels = 1
latent_size = 128
model = VanillaVAE(in_channels, latent_size)
model.load_state_dict(torch.load(model_path))

# Sample data
data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
dataset = MovingMNISTDataset(root_dir=data_path, load_type='image',
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    # transforms.ToTensor()
                                    ])
                            )
loader =  DataLoader(dataset=dataset, batch_size=1, shuffle=True)

image, target = next(iter(loader))

print(image.shape, target[0].shape)



inference = model(image)

infer_image = inference[0][0]

# print(infer_image.shape)
infer_image = rearrange(infer_image.cpu().detach().numpy(), 'c h w -> h w c')

cv2.imshow('inference', infer_image)
cv2.imshow('target', rearrange(target[0].cpu().detach().numpy(), 'c h w -> h w c'))
cv2.waitKey(10000)


cv2.imwrite('inference.jpg', infer_image)
cv2.imwrite('target.jpg', rearrange(target[0].cpu().detach().numpy(), 'c h w -> h w c'))


