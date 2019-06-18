#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.autograd import Variable
from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
from util.SaveFeature import SaveFeature
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
def Extract_Feature(Model, args):


    save_dir = '{}/{}/Feature'.format(args.output, args.snapshot)

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    if args.cuda:
        Model.cuda()
    Model.eval()


    count = 0
    # Load augmented data
    #transformed_dataset = FaceIdPoseDataset(args.test_csv_file, args.data_place,
     #                                       transform = transforms.Compose([Resize((256,256)), RandomCrop((224,224))]))
    transformed_dataset = FaceIdPoseDataset(args.test_csv_file, args.data_place,
                                            transforms.Compose([transforms.Resize(256),
                                                                transforms.CenterCrop(224),
                                                                transforms.ToTensor()
                                                                ])
                                            )
    dataloader = DataLoader(transformed_dataset, batch_size=args.Test_Batch, shuffle=False,num_workers=4)

    for i, batch_data in enumerate(dataloader):
        batch_image = torch.FloatTensor(batch_data[0].float())
        minibatch_size = len(batch_image)

        if args.cuda:
            batch_image = batch_image.cuda()
        #batch_image = Variable(batch_image, volatile=True)
        batchImageName = batch_data[1]
        module=list(Model.children())[:-1]
        feature_extractor=nn.Sequential(*module)
        for p in feature_extractor.parameters():
            p.requires_grad = False
        features = feature_extractor(batch_image)
        features = features.data.cpu().numpy()
        SaveFeature(features, batchImageName, save_dir, args)
        count += minibatch_size
        print("Finish Processing {} images...".format(count))