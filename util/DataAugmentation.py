#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import scipy as sp
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import pdb
import cv2
from PIL import Image
from torchvision import transforms
class ValidateDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.imgFrame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgFrame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgFrame.ix[idx, 0])
        imgName = self.imgFrame.ix[idx, 0]
        if not os.path.isfile(img_path):
            print('>>> No Such File: {}'.format(img_path))
            exit()
        image = cv2.imread(img_path)
        image = image.astype(np.float32)
        image = (image * 2)/255 - 1  # Inew = (I - I.min) * (newmax - newmin)/(I.max - I.min)  + newmin
        ID = self.imgFrame.ix[idx, 1]
        if self.transform:
            image = self.transform(image)

        return [image, imgName, ID]


class FaceIdPoseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.imgFrame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgFrame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgFrame.ix[idx, 0])
        imgName = self.imgFrame.ix[idx, 0]
        if not os.path.isfile(img_path):
            print('>>> No Such File: {}'.format(img_path))
            exit()
        #transform=transforms.Compose([transforms.ToTensor()])
        #image=transform(image)
        #print("image2=",image2)
        #image = cv2.imread(img_path)
        #image = image.astype(np.float32)
        #image = (image * 2)/255 - 1  # Inew = (I - I.min) * (newmax - newmin)/(I.max - I.min)  + newmin
        image = Image.open(img_path)
        image=image.convert('RGB')
        #print(image.size)

        ID = self.imgFrame.ix[idx, 1]
        if self.transform:
            image_tensor = self.transform(image)

        return [image_tensor, imgName, ID]

class Resize(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        new_h, new_w = self.output_size
        pad_width = int((new_h - image.shape[1]) / 2)
        resized_image = np.lib.pad(image, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 'edge')
        return resized_image


class RandomCrop(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[:, top:top+new_h, left:left+new_w]

        return cropped_image
