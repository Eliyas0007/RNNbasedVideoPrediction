import cv2
import glob
import torch
import numpy
import random
import torchvision.transforms as transforms

from einops import rearrange
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

class CustomMovingMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, load_type='image'):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.load_type = load_type

        self._all_video_paths = natsorted(glob.glob(self.root_dir + '/**/'))
        self._all_image_paths = []
        for path in self._all_video_paths:
            image_pathes = natsorted(glob.glob(path + '/*.png'))
            rand_index = random.randrange(0, 20)
            self._all_image_paths.append(image_pathes[rand_index])
        

    def __len__(self):
        if self.load_type == 'image':
            return len(self._all_image_paths)
        else:
            return len(self._all_video_paths)

    def __getitem__(self, index):
      
        
        if self.load_type == 'video':

            train_frames = torch.empty((10, 1, 64, 64))
            label_frames = torch.empty((10, 1, 64, 64))

            if len(self._all_video_paths) > 1:
                video_path = self._all_video_paths[index]

                frame_paths = natsorted(glob.glob(video_path + '/*.png'))
                for i, path in enumerate(frame_paths):
                    frame = cv2.imread(path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = numpy.asarray(frame)
                    frame = torch.from_numpy(frame)
                    frame = transforms.functional.convert_image_dtype(frame, dtype=torch.float32)
                    frame = frame.unsqueeze(0)
                    frame = self.transform(frame)
                    if i < (len(frame_paths) // 2):
                        train_frames[i] = frame
                    else:
                        label_frames[i-10] = frame


            return train_frames, label_frames
        else:
            image_path = self._all_image_paths[index]
            image = cv2.imread(image_path)   	
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = numpy.asarray(image)
            image = torch.from_numpy(image)
            image = transforms.functional.convert_image_dtype(image, dtype=torch.float32)
            image = image.unsqueeze(0)
            image = self.transform(image)

        return image

if __name__ == '__main__':
    custom_data_path = '/home/yiliyasi/Documents/Projects/Mine/MovingMNIST-Generator/data_horizontal'
    cmd = CustomMovingMNISTDataset(custom_data_path, transform=transforms.Compose([
                                    transforms.Normalize((0.5), (0.5)),
                                    ]),
                                    load_type='video')
    # print(cmd.__len__())
    # train, target = cmd.__getitem__(0)
    loader = DataLoader(dataset=cmd, batch_size=10, shuffle=True)
    train, target = next(iter(loader))
    print(train.shape, target.shape)

