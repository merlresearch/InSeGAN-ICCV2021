#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_HEIGHT = 64
INIT_CROP = 214# 600 #214
train_transforms = [
    transforms.CenterCrop(INIT_CROP),
    transforms.Resize(IMG_HEIGHT),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-20,20)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]
test_transforms = [
    transforms.CenterCrop(INIT_CROP),
    transforms.Resize(IMG_HEIGHT),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]
gt_transforms = [
    transforms.CenterCrop(INIT_CROP),
    transforms.Resize(IMG_HEIGHT,interpolation=Image.NEAREST),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.ToTensor()
]

class ObjectDataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = glob.glob('%s/*.png' % root)

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        img = self.transform(Image.open(filepath).convert('RGB'))
        return img

    def __len__(self):
        return len(self.files)

class ObjectDatasetTest(Dataset):
    def __init__(self,
                 root,
                 gtroot,
                 transforms_= None,
                 gt_transforms_ = None,
                 mode='test'):
        self.transform = transforms.Compose(transforms_)
        self.gt_transform = transforms.Compose(gt_transforms_)
        self.files = glob.glob('%s/*.png' % root)
        self.gtroot = gtroot

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        gtfile = os.path.join(self.gtroot, filepath.split('/')[-1].replace('depth', 'rgb'))

        img = self.transform(Image.open(filepath).convert('RGB'))
        gt = self.gt_transform(Image.open(gtfile).convert('RGB'))

        return (img, gt)

    def __len__(self):
        return len(self.files)
