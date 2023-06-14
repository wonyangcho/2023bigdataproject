import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from augmentation import RandAugment

import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2
import random

import os
from dataset import listDataset

logger = logging.getLogger(__name__)


normal_mean = (0.485, 0.456, 0.406)
normal_std = (0.229, 0.224, 0.225)


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        rw = int(args.w * args.resize)
        rh = int(args.h * args.resize)

        self.ori = transforms.Compose([
            transforms.Resize((rw, rh)),
            # transforms.RandomHorizontalFlip(),

        ])
        self.aug = transforms.Compose([
            transforms.Resize((rw, rh)),
            RandAugment(n=n, m=m)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)

        return self.normalize(ori), self.normalize(aug)


def load_data(img_path, args, train=True):

    if train:
        if args.do_crop:
            gt_path = img_path.replace('.jpg', '.h5kp').replace(
                'images', 'gt_density_map')
        else:
            gt_path = img_path.replace('.jpg', '.h5').replace(
                'images', 'gt_density_map')
    else:
        gt_path = img_path.replace('.jpg', '.h5').replace(
            'images', 'gt_density_map')

    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)

            if train and args.do_crop:
                gt_count = np.asarray(gt_file['kpoint'])
            else:
                gt_count = np.asarray(gt_file['gt_count'])

            break  # Success!
        except OSError:
            print("load error:", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        if train and args.do_crop:
            width, height = img.size[0], img.size[1]

            m = int(width / 384)
            n = int(height / 384)

            for i in range(0, m):
                for j in range(0, n):
                    crop_img = img.crop(
                        (i * 384, j * 384, (i + 1) * 384, (j + 1) * 384))
                    crop_kpoint = gt_count[j *
                                           384: (j + 1) * 384, i * 384: (i + 1) * 384]
                    gt_count_crop = np.sum(crop_kpoint)

                    blob = {}
                    blob['img'] = crop_img
                    blob['gt_count'] = gt_count_crop
                    blob['fname'] = fname
                    data_keys[count] = blob
                    count += 1

        else:
            blob = {}
            blob['img'] = img
            blob['gt_count'] = gt_count
            blob['fname'] = fname
            data_keys[count] = blob
            count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys


def get_crowd(args):

    train_l_list = []
    val_l_list = []
    train_ul_list = []
    test_l_list = []

    train_dataset_paths = [args.train_ShanghaiA_data,
                           args.train_ShanghaiB_data, args.train_qnrf_data]

    test_dataset_paths = [args.test_dataset]

    train_labeled_nums = [30, 40, 120]
    train_unlabeled_nums = [210, 280, 721]
    val_labeled_nums = [60, 80, 240]

    if args.dataset_index != -1:
        train_dataset_paths = [train_dataset_paths[args.dataset_index]]

        print(
            f"train_dataset_paths: {train_dataset_paths} {args.dataset_index}")

        train_labeled_nums = [train_labeled_nums[args.dataset_index]]
        train_unlabeled_nums = [train_unlabeled_nums[args.dataset_index]]
        val_labeled_nums = [val_labeled_nums[args.dataset_index]]

    for i, data_path in enumerate(train_dataset_paths):

        with open(args.home+data_path, 'rb') as outfile:
            train_list = np.load(outfile).tolist()
            np.random.shuffle(train_list)

            labeled_list = train_list[:train_labeled_nums[i]]
            labeled_sum = train_labeled_nums[i]

            val_labeled_list = train_list[labeled_sum:labeled_sum +
                                          val_labeled_nums[i]]
            labeled_sum = train_labeled_nums[i]+val_labeled_nums[i]

            unlabeled_list = train_list[labeled_sum:labeled_sum +
                                        train_unlabeled_nums[i]]

            train_l_list.extend(labeled_list)
            train_ul_list.extend(unlabeled_list)
            val_l_list.extend(val_labeled_list)

    print(f"test_dataset_paths: {test_dataset_paths}")

    for i, data_path in enumerate(test_dataset_paths):

        with open(args.home+data_path, 'rb') as outfile:
            test_list = np.load(outfile).tolist()
            np.random.shuffle(train_list)

            # test_list = train_list[:test_labeled_nums[i]]

            test_l_list.extend(test_list)

    print(
        f"===== labeled: {len(train_l_list)} unlabeled:{len(train_ul_list)} test:{len(test_l_list)}")

    train_l_data = pre_data(train_l_list, args, train=True)
    train_ul_data = pre_data(train_ul_list, args, train=True)
    val_l_data = pre_data(val_labeled_list, args, train=False)
    test_data = pre_data(test_l_list, args, train=False)

    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default

    rw = int(args.w * args.resize)
    rh = int(args.h * args.resize)

    transform_labeled = transforms.Compose([
        # transforms.Resize((rw, rh)),
        # transforms.RandomHorizontalFlip(),
        # RandAugment(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])
    transform_finetune = transforms.Compose([
        # transforms.Resize((rw, rh)),
        # transforms.RandomHorizontalFlip(),
        # RandAugment(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])

    transform_val = transforms.Compose([
        # transforms.Resize((rw, rh)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])

    train_labeled_dataset = listDataset(
        train_l_data, transform_labeled, train=True, args=args)
    finetune_dataset = listDataset(
        train_l_data, transform_finetune, train=True, args=args)
    train_unlabeled_dataset = listDataset(train_ul_data, TransformMPL(
        args, mean=normal_mean, std=normal_std), train=True, args=args)
    val_labeled_dataset = listDataset(
        val_l_data, transform_val, train=False, args=args)
    test_dataset = listDataset(
        test_data, transform_val, train=False, args=args)

    return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, test_dataset, finetune_dataset


DATASET_GETTERS = {'crowd': get_crowd}
