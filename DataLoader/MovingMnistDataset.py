import torch
import numpy
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    
    def __init__(self, root_dir, load_type='video', transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.load_type = load_type

        if load_type != 'video' and load_type != 'image':
            assert 'loading type not supported! only [video] or [image]'

        assert (self.root_dir is not None or self.root_dir is not ''), "Root dir is empty"
        
        self.allvideos = numpy.load(self.root_dir)
        self.allvideos = numpy.array(rearrange(self.allvideos, 'f b w h -> b f w h'))
        self.allvideos = torch.from_numpy(self.allvideos)


    def __len__(self):
        video_len, frame_len, _, _ = self.allvideos.shape
        if self.load_type is 'video':
            return video_len
        else:
            return video_len * frame_len

    def __getitem__(self, index):
        
        if self.load_type is 'video':
            train_frames = torch.empty((10, 1, 64, 64))
            label_frames = torch.empty((10, 1, 64, 64))

            if len(self.allvideos) > 1:
                video = self.allvideos[index]

                for i, frame in enumerate(video):
                    # frame = self.transform(frame)
                    if i < (len(video) // 2):
                        frame = frame.unsqueeze(dim=0)
                        train_frames[i] = frame
                    else:
                        frame = frame.unsqueeze(dim=0)
                        label_frames[i-10] = frame


            return train_frames, label_frames
        else:
            all_videos = rearrange(self.allvideos, 'b f h w -> (b f) h w')
            all_videos = all_videos.unsqueeze(dim=1)
            image = transforms.functional.convert_image_dtype(all_videos[index], dtype=torch.float32)
            image = self.transform(image)
            return image, image
