import torch
import numpy
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform

        assert (self.root_dir is not None or self.root_dir is not ''), "Root dir is empty"
        
        self.allvideos = numpy.load(self.root_dir)
        self.allvideos = numpy.array(rearrange(self.allvideos, 'f b w h -> b f w h'))
        self.allvideos = torch.from_numpy(self.allvideos)


    def __len__(self):
        return len(self.allvideos)

    def __getitem__(self, index):
        
        train_frames = torch.empty((10, 1, 64, 64))
        label_frames = torch.empty((10, 1, 64, 64))

        if len(self.allvideos) > 1:
            video = self.allvideos[index]

            for i, frame in enumerate(video):
                if i < (len(video) // 2):
                    frame = frame.unsqueeze(dim=0)
                    train_frames[i] = frame
                else:
                    frame = frame.unsqueeze(dim=0)
                    label_frames[i-10] = frame

        return train_frames, label_frames