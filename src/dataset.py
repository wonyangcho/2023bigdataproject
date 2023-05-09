import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import numbers
from torchvision import datasets, transforms
import torch.nn.functional as F


class listDataset(Dataset):
    def __init__(self, root, transform=None, train=False,
                 args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train

        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']

        gt_count = gt_count.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        return img, gt_count
