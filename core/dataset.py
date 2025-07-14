import os
import json
import random
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from core.utils import (TrainZipReader, TestZipReader,
                        create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
import json

class FaceRetouchingDataset(Dataset):
    def __init__(self, path, resolution=512, data_type="train", data_percentage=1):
        self.resolution = resolution
        self.data_type = data_type
        self.imgs = glob.glob(os.path.join(path, data_type, 'source', '*.*'))
        self.imgs_r = glob.glob(os.path.join(path, data_type, 'target', '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.imgs_r = sorted(self.imgs_r, key=lambda x: os.path.basename(x))
        assert len(self.imgs) == len(self.imgs_r), "Can not match the FFHQ and FFHQR!"
        # for p, p_r in zip(self.imgs, self.imgs_r):
        #     assert os.path.basename(p) == os.path.basename(p_r), "Can not match the FFHQ and FFHQR!"

        self.length = len(self.imgs)
        
        if data_type == 'train':
            self.length = int(self.length*data_percentage)
            self.imgs = self.imgs[:self.length]
            self.imgs_r = self.imgs_r[:self.length]
        print(f"Data number: {len(self.imgs)}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img_r = Image.open(self.imgs_r[index]).convert("RGB")

        toTensor = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor()
        ])
            
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img, img_r = toTensor(img), toTensor(img_r) 
        img, img_r = normalize(img), normalize(img_r)
        
        if self.data_type == 'train':
            flip = random.random()
            if flip < 0.5: 
                img, img_r = TF.hflip(img), TF.hflip(img_r)

        return (img, img_r)


class UnpairFaceDataset(Dataset):
    def __init__(self, path, resolution=512, data_type="train", return_gt=False, data_percentage=0.5):
        self.resolution = resolution
        self.ret_gt = return_gt
        self.imgs = glob.glob(os.path.join(path, data_type, 'source', '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.imgs_r = glob.glob(os.path.join(path, data_type, 'target', '*.*'))
        self.imgs_r = sorted(self.imgs_r, key=lambda x: os.path.basename(x))
        self.length = len(self.imgs)
        self.imgs = self.imgs[int(self.length*data_percentage):]
        self.imgs_r = self.imgs_r[int(self.length*data_percentage):]
        self.length = len(self.imgs)
        random.shuffle(self.imgs)
        random.shuffle(self.imgs_r)
        self.data_type = data_type
        print("UnpairFaceDataset")
        print(f"Used {data_type} data: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img_r = Image.open(self.imgs_r[index]).convert("RGB")

        toTensor = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor()
        ])
            
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img, img_r = toTensor(img), toTensor(img_r) 
        img, img_r = normalize(img), normalize(img_r)
        
        if self.data_type == 'train':
            flip = random.random()
            if flip < 0.5: 
                img, img_r = TF.hflip(img), TF.hflip(img_r)
        if self.ret_gt:
            return img, img_r
        else:
            return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class wildDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        self.root = dataroot
        self.source_paths = sorted(make_dataset(self.root))
        self.size = 512
        self.dataset_size = len(self.source_paths)

    def __getitem__(self, index):
        img = Image.open(self.source_paths[index])
        toTensor = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = toTensor(img)
        return os.path.basename(self.source_paths[index])[:-4], img

    def __len__(self):
        return len(self.source_paths)
