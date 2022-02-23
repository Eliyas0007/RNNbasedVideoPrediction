import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# from tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

import sys
sys.path.append(".")
from DataLoader.MovingMnistDataset import MovingMNISTDataset
from Models.Seq2Seq import Seq2Seq
from Models.ConvRNNCell import ConvRNNEncoderCell, ConvRNNDecoderCell

# save path for model
model_save_path = './SavedModels/'

# Hyperparameters
num_epochs = 1

learning_rate = 0.0002

batch_size = 32

# Model Hyperparameters
load_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(f'runs/')
step = 0

# models
encoder_cell = ConvRNNEncoderCell(hidden_size=54 * 54)
decoder_cell = ConvRNNDecoderCell()
model = Seq2Seq(encoder_cell, decoder_cell, device, batch_size).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_path = '/home/yiliyasi/Downloads/mnist_test_seq.npy'

dataset = MovingMNISTDataset(root_dir=data_path,
                                transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5))
                                    ])
                            )
loader =  DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

criterion = nn.BCELoss()

if load_model:
    # TODO 
    pass

for epoch in range(num_epochs):

    print(f'Epoch [current: {epoch} / total: {num_epochs}]')


    with tqdm.tqdm(loader, unit='batch') as tepoch:
        for data_pair in tepoch:
            train, target = data_pair
            train = train.to(device)
            target = target.to(device)

            predicted_frames = model(train)

            optimizer.zero_grad()
            loss = criterion(predicted_frames.to(device), train)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            print(f'Training Loss : {loss.item()}')

            if step % 25 == 0:
                torch.save(model.state_dict(), model_save_path + f'step{step}.pth')

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1
