from curses import flash
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# from tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset
from Models.Seq2Seq import Seq2Seq
from Models.ConvRNNCell import ConvRNNEncoderCell, ConvRNNDecoderCell

# Hyperparameters
num_epochs = 1

learning_rate = 0.001

batch_size = 16

# Model Hyperparameters
load_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# writer = SummaryWriter(f'runs/loss_plot')
step = 0

encoder_cell = ConvRNNEncoderCell(hidden_size=54 * 54)
decoder_cell = ConvRNNDecoderCell()

model = Seq2Seq(encoder_cell, decoder_cell, device).to(device)

data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'

dataset = MovingMNISTDataset(root_dir=data_path,
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5))
                                    ])
                            )

loader =  DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)

for e in range(num_epochs):
    for i, data_pair in enumerate(loader):
        train, target = data_pair
        train = train.to(device)
        target = target.to(device)
        # print(target.shape)

        predicted_frames = model(train)
        # print(predicted_frames.shape)

        break