#/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from util.log_learning import log_learning
from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
from util.exp_lr_scheduler import adjust_learning_rate
from util.Validation import Validation_Process
from util.checkpoint import save_checkpoint
import torch.backends.cudnn as CUDNN
from torch.optim import lr_scheduler
from model.resnet import AngleLoss

import numpy
#writer = SummaryWriter()

def Train(Model, args):

    writer = SummaryWriter()
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2


    if args.cuda:
        Model.cuda()

    #optimizer = optim.Adam(Model.parameters(), lr=args.lr, betas=(beta1_Adam, beta2_Adam))
    optimizer = optim.SGD(Model.parameters(), lr=args.lr)

    if args.resume:
        checkpoint = torch.load(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])

    Model.train()


    steps = 0
    #loss_criterion_Angular = AngleLoss().cuda()
    CUDNN.benchmark = True
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    if args.dynamic_lr == True:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3000, verbose=False,
                                                   threshold=0.00001, threshold_mode='rel', cooldown=2000, min_lr=0,
                                                   eps=1e-08)
    for epoch in range(args.start_epoch, args.epochs+1):
        #if epoch==3:
            #optimizer = optim.SGD(Model.parameters(), lr=args.lr)
        # Every args.lr_step, changes learning rate by multipling args.lr_decay
                    #adjust_learning_rate(optimizer, epoch, args)

        # Load augmented data
        #transformed_dataset = FaceIdPoseDataset(args.train_csv_file, args.data_place,
                                        #transform = transforms.Compose([Resize((256,256)), RandomCrop((224,224))])) #for ResNet256x256->224x224 for VGG110x110->96x96
        # transformed_dataset = FaceIdPoseDataset(args.train_csv_file, args.data_place,
        #                                             transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),transforms.ToTensor()]))  # for ResNet256x256->224x224 for VGG110x110->96x96
        transformed_dataset = FaceIdPoseDataset(args.train_csv_file, args.data_place,transforms.Compose([transforms.Resize(256),
                                                                                                         transforms.RandomCrop(224),
                                                                                                         transforms.ToTensor()
                                                                                                         ]))  # for ResNet256x256->224x224 for VGG110x110->96x96

        dataloader = DataLoader(transformed_dataset, batch_size=args.Train_Batch, shuffle=True, num_workers=8)
        if args.stepsize > 0:
            scheduler.step()

        for i, batch_data in enumerate(dataloader):
            # backward() function accumulates gradients, however we don't want to mix up gradients between minibatches
            optimizer.zero_grad()
            batch_image = torch.FloatTensor(batch_data[0].float())

            batch_id_label = batch_data[2]

            if args.cuda:
                batch_image, batch_id_label = batch_image.cuda(), batch_id_label.cuda()

            batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

            steps += 1

            Prediction = Model(batch_image)
            Loss = Model.ID_Loss(Prediction, batch_id_label)
            #Loss = loss_criterion_Angular(Prediction, batch_id_label)

            Loss.backward()
            optimizer.step()
            if args.dynamic_lr == True:
                scheduler.step(Loss)
            log_learning(epoch, steps, 'ResNet50_Model', args.lr, Loss.item(), args)
            writer.add_scalar('Train/Train_Loss', Loss, steps)
            writer.add_scalar('Train/Model_Lr', optimizer.param_groups[0]['lr'], epoch)

            # Validation_Process(Model, epoch, writer, args)
        Validation_Process(Model, epoch, writer, args)

        if epoch % args.save_freq == 0:
            if not os.path.isdir(args.snapshot_dir): os.makedirs(args.snapshot_dir)
            save_path = os.path.join(args.snapshot_dir, 'epoch{}.pt'.format(epoch))
            torch.save(Model.state_dict(), save_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'Model': Model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_dir=os.path.join(args.snapshot_dir, 'epoch{}'.format(epoch)))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()