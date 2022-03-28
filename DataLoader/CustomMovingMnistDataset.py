import cv2
import glob
import torch
import numpy
import random

from einops import rearrange
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

class CustomMovingMNISTDataset(Dataset):
    def __init__(self, root_dir=None, transforms=None):
        super().__init__()

        _default_root_dir = '/Users/eliyassuleyman/Documents/Work/Repos/MovingMNIST-Generator/data_vertical'
        self.root_dir = _default_root_dir
        self.tansforms = transforms

        _all_video_paths = natsorted(glob.glob(self.root_dir + '/**/'))
        self._all_image_paths = []
        for path in _all_video_paths:
            image_pathes = natsorted(glob.glob(path + '/*.png'))
            self._all_image_paths.append(image_pathes[random.randrange(0, 21)])
        

    def __len__(self):
        return len(self._all_image_paths)

    def __getitem__(self, index):
        image_path = self._all_image_paths[index]
        image = cv2.imread(image_path)   	
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = numpy.asarray(image)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        return image

if __name__ == '__main__':
    cmd = CustomMovingMNISTDataset()

    loader = DataLoader(dataset=cmd, batch_size=10, shuffle=True)
    batch = next(iter(loader))

