import os
import torch
import numpy

from einops import rearrange
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.root_dir = root_dir

        assert (self.root_dir is not None or self.root_dir is not ''), "Root dir is empty"
        
        self.allvideos = numpy.load(self.root_dir)
        self.allvideos = rearrange(self.allvideos, 'f b w h -> b f w h')

    def __len__(self):
        return len(self.allvideos)

    def __getitem__(self, index):
        
        train_frames, label_frames = [], []
        if len(self.allvideos) > 1:
            video = self.allvideos[index]
            for i, frame in enumerate(video):
                if i < (len(video) // 2):
                    # print('haha')
                    train_frames.append(frame)
                else:
                    # print('sese')
                    label_frames.append(frame)

        train_frames = numpy.array(train_frames)
        label_frames = numpy.array(label_frames)
         
        return torch.as_tensor(train_frames), torch.as_tensor(label_frames)