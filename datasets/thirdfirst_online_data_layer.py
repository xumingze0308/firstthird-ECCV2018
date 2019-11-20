import os

import numpy as np
from PIL import Image
import torch
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

def thirdfirst_online_data_layer(prefix, dataset, name, fp, tp=None):
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225],),
    ])

    flw_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    if tp is not None:
        img = img_loader(prefix, dataset, name, fp)
        flw = flw_loader_from_numpy(prefix, dataset, get_pre_name(name), fp)
        pre = msk_loader(prefix, dataset, get_pre_name(name), fp, tp)
        msk = msk_loader(prefix, dataset, name, fp, tp)

        img = img_transforms(img)
        flw = flw_transforms(flw)
        pre = torch.from_numpy(np.array(pre)[np.newaxis, ...]).float()
        msk = torch.from_numpy(np.array(msk)).long()

        return img, flw, pre, msk
    else:
        img = img_loader(prefix, dataset, name, fp)
        flw = flw_loader_from_numpy(prefix, dataset, get_pre_name(name), fp)

        img = img_transforms(img)
        flw = flw_transforms(flw)

        return img, flw
