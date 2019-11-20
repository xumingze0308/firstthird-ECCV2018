import os
import json

import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

def get_pre_name(name):
    return str(int(name.split('.')[0])-1).zfill(5) + '.png'

def img_loader(prefix, dataset, name, fp):
    return Image.open(str(os.path.join(prefix, dataset, 'PNGImages', fp, name)))

def flw_loader_from_numpy(prefix, dataset, name, fp):
    flw_path = os.path.join(prefix, dataset, 'NUMPYFlows', fp)
    flw_x = np.load(str(os.path.join(flw_path, name.split('.')[0]+'_flow_x.npy'))).astype(np.float32)[..., np.newaxis]
    flw_y = np.load(str(os.path.join(flw_path, name.split('.')[0]+'_flow_y.npy'))).astype(np.float32)[..., np.newaxis]
    return np.concatenate((flw_x, flw_y), 2)

def flw_loader_from_image(prefix, dataset, name, fp):
    flw_path = os.path.join(prefix, dataset, 'PNGFlows', fp)
    flw_x = np.array(Image.open(str(os.path.join(flw_path, name.split('.')[0]+'_flow_x.png'))))[..., np.newaxis]
    flw_y = np.array(Image.open(str(os.path.join(flw_path, name.split('.')[0]+'_flow_y.png'))))[..., np.newaxis]
    return np.concatenate((flw_x, flw_y), 2)

def msk_loader(prefix, dataset, name, fp, tp):
    return Image.open(str(os.path.join(prefix, dataset, 'Annotations', fp, tp+'_'+name)))

class ThirdThirdOfflineDataLayer(data.Dataset):

    def __init__(self, prefix, root, img_transforms=None, flw_transforms=None,
                 img_loader=img_loader, flw_loader=flw_loader_from_numpy, msk_loader=msk_loader):
        self.prefix = prefix
        self.inputs = json.load(open(root, 'r'))
        self.img_loader = img_loader
        self.flw_loader = flw_loader
        self.msk_loader = msk_loader
        self.img_transforms = img_transforms
        self.flw_transforms = flw_transforms

    def __getitem__(self, index):
        dataset, name, fp1, tp1, fp2, tp2, target = self.inputs[index]

        img1 = self.img_loader(self.prefix, dataset, name, fp1)
        img2 = self.img_loader(self.prefix, dataset, name, fp2)
        flw1 = self.flw_loader(self.prefix, dataset, get_pre_name(name), fp1)
        flw2 = self.flw_loader(self.prefix, dataset, get_pre_name(name), fp2)
        pre1 = self.msk_loader(self.prefix, dataset, get_pre_name(name), fp1, tp1)
        pre2 = self.msk_loader(self.prefix, dataset, get_pre_name(name), fp2, tp2)
        msk1 = self.msk_loader(self.prefix, dataset, name, fp1, tp1)
        msk2 = self.msk_loader(self.prefix, dataset, name, fp2, tp2)

        if self.img_transforms is not None:
            img1 = self.img_transforms(img1)
            img2 = self.img_transforms(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        if self.flw_transforms is not None:
            flw1 = self.flw_transforms(flw1)
            flw2 = self.flw_transforms(flw2)
        else:
            flw1 = ToTensor()(flw1)
            flw2 = ToTensor()(flw2)

        pre1 = torch.from_numpy(np.array(pre1)[np.newaxis, ...]).float()
        pre2 = torch.from_numpy(np.array(pre2)[np.newaxis, ...]).float()

        msk1 = torch.from_numpy(np.array(msk1)).long()
        msk2 = torch.from_numpy(np.array(msk2)).long()

        target = torch.from_numpy(np.array([target])).float()

        img1 = torch.cat((img1, pre1), 0)
        img2 = torch.cat((img2, pre2), 0)
        flw1 = torch.cat((flw1*pre1, pre1), 0)
        flw2 = torch.cat((flw2*pre2, pre2), 0)

        return img1, flw1, msk1, img2, flw2, msk2, target

    def __len__(self):
        return len(self.inputs)
