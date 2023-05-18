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

        if self.train == True:

            if self.transform is not None:
                img = self.transform(img)

            return img, gt_count
        else:

            if self.transform is not None:
                img = self.transform(img)

            device = img.device

            width, height = img.shape[2], img.shape[1]

            m = int(width / 384)
            n = int(height / 384)
            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = img[:, j * 384: 384 *
                                         (j + 1), i * 384:(i + 1) * 384].to(device).unsqueeze(0)
                    else:
                        crop_img = img[:, j * 384: 384 *
                                       (j + 1), i * 384:(i + 1) * 384].to(device).unsqueeze(0)

                        img_return = torch.cat(
                            [img_return, crop_img], 0).to(device)

            print(f"==========={img_return.shape}")

            return img_return, gt_count
