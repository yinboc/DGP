import json
import os
import os.path as osp
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
from torchvision import get_image_backend


class ImageNet():

    def __init__(self, path):
        self.path = path
        self.keep_ratio = 1.0
    
    def get_subset(self, wnid):
        path = osp.join(self.path, wnid)
        return ImageNetSubset(path, wnid, keep_ratio=self.keep_ratio)

    def set_keep_ratio(self, r):
        self.keep_ratio = r


class ImageNetSubset(Dataset):

    def __init__(self, path, wnid, keep_ratio=1.0):
        self.wnid = wnid

        def pil_loader(path):
            with open(path, 'rb') as f:
                try:
                    img = Image.open(f)
                except OSError:
                    return None
                return img.convert('RGB')

        def accimage_loader(path):
            import accimage
            try:
                return accimage.Image(path)
            except IOError:
                return pil_loader(path)

        def default_loader(path):
            if get_image_backend() == 'accimage':
                return accimage_loader(path)
            else:
                return pil_loader(path)

        # get file list
        all_files = os.listdir(path)
        files = []
        for f in all_files:
            if f.endswith('.JPEG'):
                files.append(f)
        random.shuffle(files)
        files = files[:max(1, round(len(files) * keep_ratio))]

        # read images
        data = []
        for filename in files:
            image = default_loader(osp.join(path, filename))
            if image is None:
                continue
            # pytorch model-zoo pre-process
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            data.append(preprocess(image))
        if data != []:
            self.data = torch.stack(data) 
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.wnid

