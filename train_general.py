import cv2
import tqdm
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# from tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from PytorchVAE.models.vanilla_vae import VanillaVAE
from DataLoader.MovingMnistDataset import MovingMNISTDataset
from Models.AutoEncoder import Encoder, Decoder, AutoEncoder

def train(  model,
            load_model, 
            lr_rate,
            loss_fn, 
            data_set, 
            num_epochs, 
            device, 
            save_model_path, 
            writer):

    step = 0

    if load_model:
        autoencoder.load_state_dict(torch.load(model_path))
        print('state dict loaded!')

    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    
    for epoch in range(num_epochs): 

        print(f'Epoch [current: {epoch} / total: {num_epochs}]')

        with tqdm.tqdm(data_set, unit='batch') as tepoch:
            for (train, target) in tepoch:
            
                train = train.to(device)
                target = target.to(device)

                pred = model(train)

                optimizer.zero_grad()

                loss = loss_fn(pred, train)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                if step % 1000 == 0:
                    print(f'Training Loss : {loss.item()}')
                    save_model_and_image(model, save_model_path, step, pred, 'autoencoder')

                writer.add_scalar('Training Loss', loss.item(), global_step=step)

                step += 1

def save_model_and_image(model, path, step, prediction, name_of_model):
    torch.save(model.state_dict(), path + f'{name_of_model}_{step}.pth')
    image = numpy.array(prediction[0].cpu().detach().numpy())
    image = rearrange(image, 'c w h -> w h c')
    cv2.imwrite(f'aelstm_runs/ae_{step}.png', image)



if __name__ == '__main__':

    print('Initializing...')

    # save path for model
    model_save_path = './SavedModels/'
    model_path = './SavedModels/autoencoder_step12300.pth'

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.0002
    batch_size = 16

    # Model Hyperparameters
    load_model = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'aelstm_runs/')
    step = 0

    # VAE
    in_channels = 1
    latent_size = 128
    model = VanillaVAE(in_channels, latent_size)
    model = model.to(device)

    # Autoencoder
    encoder = Encoder(in_channels, latent_size).to(device)
    decoder = Decoder(latent_size).to(device)
    autoencoder = AutoEncoder(encoder, decoder).to(device)

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
    print('Data Loaded, Start training..')


    train(  model,
            load_model=False, 
            lr_rate=learning_rate,
            loss_fn=criterion,
            data_set=dataset,
            num_epochs=num_epochs,
            device=device,
            save_model_path=model_save_path,
            writer=writer)

    

    