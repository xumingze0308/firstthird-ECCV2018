import os
import sys
import json
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

sys.path.insert(0, '../../')
from datasets import thirdthird_online_data_layer
from models import ThirdThirdMsk
from utils import *

# Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--prefix', default='/data/mx6/data/ShareView2018/', type=str)
parser.add_argument('--root', default='../../data/', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def to_variable(x):
    x = torch.unsqueeze(x, 0)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def main(args):
    ious = []
    dist_positive = []
    dist_negative = []
    true_positive = 0.0
    false_positive = 0.0

    # Loading model
    model = ThirdThirdMsk(with_img=True, with_flw=True)
    state_dict = torch.load('../../saved_models/thirdthird_spatial_temporal_msk.pth')
    model.load_state_dict(state_dict)
    model.train(False)
    pdist = nn.PairwiseDistance(p=2)
    if torch.cuda.is_available():
        model.cuda()

    seqs = json.load(open(os.path.join(args.root, 'thirdthird_online_test_list.json'), 'r'))
    for idx_seq, seq in enumerate(seqs):
        if seq:
            ious.append([])
            pre = [[None for i in range(len(seq[0][3]))], [None for i in range(len(seq[0][5]))]]
            for idx_sample, sample in enumerate(seq):
                iou = [[None for i in range(len(sample[3]))], [None for i in range(len(sample[5]))]]
                feature = [[None for i in range(len(sample[3]))], [None for i in range(len(sample[5]))]]
                # Loop each first person
                for idx_fp, (fp, tps) in enumerate(zip([sample[2], sample[4]], [sample[3], sample[5]])):
                    # Loop each third person
                    for idx_tp, tp in enumerate(tps):
                        img, flw, gt_pre, msk = \
                                thirdthird_online_data_layer(args.prefix, sample[0], sample[1], fp, tp)

                        if pre[idx_fp][idx_tp] is None:
                            pre[idx_fp][idx_tp] = gt_pre
                        img = to_variable(torch.cat((img, pre[idx_fp][idx_tp]), 0))
                        flw = to_variable(torch.cat((flw*pre[idx_fp][idx_tp], pre[idx_fp][idx_tp]), 0))
                        msk = to_variable(msk)

                        score, f = model.forward_once(img, flw)
                        _, output = torch.max(score, 1)

                        iou[idx_fp][idx_tp] = compute_iou(output, msk)
                        feature[idx_fp][idx_tp] = f.cpu().data.float()
                        pre[idx_fp][idx_tp] = output.cpu().data.float()

                # Evaluate segmentation
                ious[-1].append(np.mean(iou))

                # Evaluate matching
                for idx_tp1, tp1 in enumerate(sample[3]):
                    if tp1 in sample[5]:
                        same_person = None
                        min_dist = sys.maxint
                        for idx_tp2, tp2 in enumerate(sample[5]):
                            dist = pdist(feature[0][idx_tp1], feature[1][idx_tp2])[0,0]
                            if dist < min_dist:
                                same_person = tp2
                                min_dist = dist
                            if tp1 == tp2:
                                dist_positive.append(dist)
                            else:
                                dist_negative.append(dist)
                        if tp1 == same_person:
                            true_positive += 1
                        else:
                            false_positive += 1

                for idx_tp1, tp1 in enumerate(sample[5]):
                    if tp1 in sample[3]:
                        same_person = None
                        min_dist = sys.maxint
                        for idx_tp2, tp2 in enumerate(sample[3]):
                            dist = pdist(feature[1][idx_tp1], feature[0][idx_tp2])[0,0]
                            if dist < min_dist:
                                same_person = tp2
                                min_dist = dist
                            if tp1 == tp2:
                                dist_positive.append(dist)
                            else:
                                dist_negative.append(dist)
                        if tp1 == same_person:
                            true_positive += 1
                        else:
                            false_positive += 1

            print('finish sequence {:3}, IoU: {:.4f}, Accuracy: {:.4f}'.format(
                idx_seq, np.mean(ious[-1]), true_positive/(true_positive+false_positive)))

    # Compute IoU
    iou_sum = 0
    iou_count = 0
    iou_len_sum = []
    iou_len_count = []
    for i in range(len(ious)):
        for j in range(len(ious[i])):
            iou_sum += ious[i][j]
            iou_count += 1
            if j >= len(iou_len_sum):
                iou_len_sum.append(ious[i][j])
                iou_len_count.append(1)
            else:
                iou_len_sum[j] += ious[i][j]
                iou_len_count[j] += 1
    iou_mean = float(iou_sum) / float(iou_count)
    iou_len_mean = np.array(iou_len_sum, dtype=np.float32) / np.array(iou_len_count, dtype=np.float32)

    # Compute Accuracy
    accuracy = true_positive / (true_positive + false_positive)

    print('finish sequence all, IoU: {:.4f}, Accuracy: {:.4f}'.format(
        iou_mean,
        accuracy,
    ))

if __name__ == '__main__':
    main(args)
