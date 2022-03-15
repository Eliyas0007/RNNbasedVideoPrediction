import sys
sys.path.append("/home/yiliyasi/Documents/Projects/RNNbasedVideoPrediction")

import cv2
import tqdm
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# from tensorboard import SummaryWriter
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Models.AutoEncoder import Encoder, Decoder, AutoEncoder
from PytorchVAE.models.vanilla_vae import VanillaVAE
from DataLoader.MovingMnistDataset import MovingMNISTDataset


print('Initializing...')

# save path for model
model_save_path = './SavedModels/'
model_path = './workingModels/autoencoder_step18000.pth'

# Hyperparameters
num_epochs = 1
learning_rate = 0.0002
batch_size = 16

# Model Hyperparameters
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(f'aelstm_runs/')
step = 0

# VAE hyperparameters
in_channels = 1
latent_size = 128
model = VanillaVAE(in_channels, latent_size)
model = model.to(device)

# Autoencoder
encoder = Encoder(in_channels, latent_size).to(device)
decoder = Decoder(latent_size).to(device)
autoencoder = AutoEncoder(encoder, decoder).to(device)

# Optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Loss function
criterion = nn.MSELoss()

# Data loading
print('Loading Data')
data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'
dataset = MovingMNISTDataset(root_dir=data_path, load_type='image',
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    # transforms.ToTensor()
                                    ])
                            )
loader =  DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

mnist_dataset = datasets.MNIST(root='MNISTDATA/', train=True, download=True, transform=transforms.Compose([
                                    # transforms.Normalize((0.5), (0.5)),
                                    transforms.ToTensor(),
                                    transforms.Resize((64, 64))
                                    ]))
mnist_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

print('Data Loaded, Start training..')

if load_model:
    autoencoder.load_state_dict(torch.load(model_path))
    print('state dict loaded!')

# t, l = next(iter(loader))
# print(t.shape, l.shape)
# image = numpy.array(l[10].cpu().detach().numpy())
# image = rearrange(image, 'c w h -> w h c')
# cv2.imshow('asd', image)
# cv2.waitKey(3000)

for epoch in range(num_epochs): 

    print(f'Epoch [current: {epoch} / total: {num_epochs}]')

    with tqdm.tqdm(mnist_loader, unit='batch') as tepoch:
        for (train, _) in tepoch:
            
            train = train.to(device)
            # target = target.to(device)
            
            pred = autoencoder(train)

            optimizer.zero_grad()

            loss = criterion(pred, train)
            loss.backward()

            nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1)
            optimizer.step()

            if step % 100 == 0:
                print(f'Training Loss : {loss.item()}')
                torch.save(autoencoder.state_dict(), model_save_path + f'fine_tuned_autoencoder_step{step}.pth')

                image = numpy.array(pred[0].cpu().detach().numpy())
                image = rearrange(image, 'c w h -> w h c')
                cv2.imwrite(f'aelstm_runs/ae_{step}.png', image*256)

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1