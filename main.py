#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import torch
from model import resnet as model
from Train import Train
from Extract_Feature import Extract_Feature
from util.Load_PretrainModel import Load_PretrainModel
import torchvision.models as models

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='ResNet')
    # learning & saving parameterss
    parser.add_argument('-train', action='store_true', default=True,
                        help='Generate pose modified image from given image')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-step-learning', action='store_true', default=False, help='enable lr step learning')
    parser.add_argument('-gamma', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('-stepsize', type=int, default=0, help='Set Step to change lr by multiply lr-decay thru every lr-step epoch [default: 35]')
    parser.add_argument('-dynamic-lr', type=bool, default=False, help='dynamic adjustment learning rate')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 1000]')
    parser.add_argument('-Train-Batch', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-Val-Batch', type=int, default=4, help='batch size for training [default: 4]')
    parser.add_argument('-Test-Batch', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-snapshot-dir', type=str, default='snapshot', help='where to save the snapshot while training')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('-start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    # data souce
    parser.add_argument('-data-place', type=str, default=None, help='prepared data path to run program')
    parser.add_argument('-output', type=str, default='Output', help='Output path for features')
    parser.add_argument('-train-csv-file', type=str, default=None, help='csv file to load image for training')
    parser.add_argument('-val-csv-file', type=str, default=None, help='csv file to load image for validation')
    parser.add_argument('-test-csv-file', type=str, default=None, help='csv file to load image for test')
    parser.add_argument('-Nd', type=int, default=500, help='initial Number of ID [default: 188]')
    parser.add_argument('-Nf', type=int, default=256, help='Dimension of feature extraction layer')
    parser.add_argument('-ChannelPerGroup', type=int, default=16, help='Number Per Group in Group Normalization, 0 will be Batch Normalization')
    parser.add_argument('-Channel', type=int, default=3, help='initial Number of Channel [default: 3 (RGB Three Channel)]')
    # option
    parser.add_argument('-Angle-Loss', action='store_true', default=False, help='Use Angle Loss')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-test', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-pretrain', default='', type=str, metavar='PATH', help='path to the pretrain model (default:none)')

    args = parser.parse_args()


    # update args and print
    if args.train:
        args.snapshot_dir = os.path.join(args.snapshot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) #拼接路徑
        os.makedirs(args.snapshot_dir)

    if args.test:
        if args.snapshot is None:
            print(">>> Sorry, please set snapshot path while extracting features")
            exit()
        else:
            print('\n>>> Loading model from [%s]...' % args.snapshot)
            checkpoint = torch.load('{}_checkpoint.pth.tar'.format(args.snapshot))
            Model = model.resnet50(args)
            Model.load_state_dict(checkpoint['Model'])
            Extract_Feature(Model, args)
    elif args.train:
        print("Parameters:")
        for attr, value in sorted(args.__dict__.items()):
            text = "\t{}={}\n".format(attr.upper(), value)
            print(text)
            with open('{}/Parameters.txt'.format(args.snapshot_dir), 'a') as f:
                f.write(text)

        if args.train_csv_file is None or args.val_csv_file is None:
            print(">>> Sorry, please set csv-file for your training/validation data")
            exit()

        if args.resume:
            if os.path.isfile(args.resume):
                print(">>> loading checkpoint '{}'".format(args.resume))
                Model = model.resnet50(args)
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                Model.load_state_dict(checkpoint['Model'])
                print(">>> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                print(">>> loaded Discriminator Pretrained Model")
                Train(Model, args)
            else:
                print(">>> no checkpoint found at '{}'".format(args.resume))
                exit()
        elif args.pretrain:
            Pretrain_Path = args.pretrain + '.pt'
            if os.path.isfile(Pretrain_Path):
                print(">>> loading Discriminator Pretrain Model '{}'".format(Pretrain_Path))
                Model = model.resnet50(args)
                print(Model)
                Pretrain_dict = torch.load(Pretrain_Path)
                Model_dict = Model.state_dict()
                MODEL = Load_PretrainModel(Model, Model_dict, Pretrain_dict)
                Train(MODEL, args)
            else:
                print(">>> no pretrain model at '{}'".format(Pretrain_Path))
                exit()
        else:
             Model = model.resnet50(args)
             # print(Model)
             #原始架構
             #Model = models.resnet50()
             Train(Model, args)


