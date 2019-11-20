import os
import sys
from itertools import ifilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torchvision import transforms

sys.path.insert(0, '../../')
from datasets import ThirdFirstOfflineDataLayer
from models import ThirdFirstMsk
from models import ContrastiveLoss
from utils import *

# Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--prefix', default='/data/mx6/data/ShareView2018/', type=str)
parser.add_argument('--root', default='../../data/', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=1e-6, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

phases = ['train', 'test']

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def main(args):
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225],),
    ])

    flw_transforms = transforms.Compose([
        transforms.ToTensor(),
        #  transforms.Normalize([0.500, 0.500],
        #                       [1.000, 1.000],),
    ])

    data_sets = {
        phase: ThirdFirstOfflineDataLayer(
            prefix = args.prefix,
            root=os.path.join(args.root, 'thirdfirst_offline_'+phase+'_list.json'),
            img_transforms = img_transforms,
            flw_transforms = flw_transforms,
        )
        for phase in phases
    }

    data_loaders = {
        phase: data.DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        for phase in phases
    }

    model = ThirdFirstMsk(with_flw=True)
    state_dict = torch.load('../../pretrained_models/onestream_spatial_temporal.pth')
    model.weights_init_from_fcn8s_shareview(state_dict)
    msk_criterion = nn.CrossEntropyLoss()
    match_criterion = ContrastiveLoss()

    if torch.cuda.is_available():
        model.cuda()
        msk_criterion.cuda()
        match_criterion.cuda()

    # Freeze segmentation parameters
    for name, param in model.named_parameters():
        if 'match' not in name:
            param.requires_grad = False
    parameters = ifilter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs+1):
        # Unfreeze segmentation parameters
        if epoch > 30:
            for name, param in model.named_parameters():
                param.requires_grad = True
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)

        msk_losses = {phase: 0.0 for phase in phases}
        match_losses = {phase: 0.0 for phase in phases}
        ious = {phase: 0.0 for phase in phases}

        for phase in phases:
            if phase == 'train':
                model.train(True)
            else:
                if epoch%5 == 0:
                    model.train(False)
                else:
                    continue

            for idx, (img1, flw1, msk, img2, flw2, target) in enumerate(data_loaders[phase]):
                img1, img2 = to_variable(img1), to_variable(img2)
                flw1, flw2 = to_variable(flw1), to_variable(flw2)
                msk = to_variable(msk)
                target = to_variable(target)

                score, feature1, feature2 = model(img1, img2, flw1, flw2)

                msk_loss = msk_criterion(score, msk)
                match_loss = match_criterion(feature1, feature2, target)
                _, output = torch.max(score, 1)
                iou = compute_iou(output, msk)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss = msk_loss + 0.5 * match_loss
                    loss.backward()
                    optimizer.step()

                msk_losses[phase] += msk_loss.cpu().data[0]*len(target.cpu().data)
                match_losses[phase] += match_loss.cpu().data[0]*len(target.cpu().data)
                ious[phase] += iou*len(target.cpu().data)

        print('epoch {:2} | (train) loss {:.4f} match: {:.4f} iou {:.4f} | '
              '(test) loss {:.4f} match: {:.4f} iou {:.4f}'.format(
                  epoch,
                  msk_losses['train']/len(data_loaders['train'].dataset),
                  match_losses['train']/len(data_loaders['train'].dataset),
                  ious['train']/len(data_loaders['train'].dataset),
                  msk_losses['test']/len(data_loaders['test'].dataset),
                  match_losses['test']/len(data_loaders['test'].dataset),
                  ious['test']/len(data_loaders['test'].dataset),
              ))

        if epoch%5 == 0:
            snapshot_path = '../../snapshots/thirdfirst_spatial_temporal_msk/'
            if not os.path.isdir(snapshot_path):
                os.makedirs(snapshot_path)
            test_iou = ious['test']/len(data_loaders['test'].dataset)
            snapshot_name = 'offline-%s-%s.pth' % (str(epoch), str(round(test_iou, 4)))
            torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))

if __name__ == '__main__':
    main(args)
