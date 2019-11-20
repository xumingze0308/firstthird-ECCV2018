import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ThirdFirstMsk(nn.Module):

    def __init__(self, n_classes=2, n_features=64, with_img=True, with_flw=True):
        super(ThirdFirstMsk, self).__init__()
        # Parameters
        self.n_classes = n_classes
        self.n_features = n_features
        self.with_img = with_img
        self.with_flw = with_flw

        # Feature extractor: third-person
        self.conv1_1_img_3rd = nn.Conv2d(4, 64, 3, padding=100)
        self.relu1_1_img_3rd = nn.ReLU(inplace=True)
        self.conv1_2_img_3rd = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_img_3rd = nn.ReLU(inplace=True)
        self.pool1_img_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_img_3rd = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_img_3rd = nn.ReLU(inplace=True)
        self.conv2_2_img_3rd = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_img_3rd = nn.ReLU(inplace=True)
        self.pool2_img_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_img_3rd = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_img_3rd = nn.ReLU(inplace=True)
        self.conv3_2_img_3rd = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_img_3rd = nn.ReLU(inplace=True)
        self.conv3_3_img_3rd = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_img_3rd = nn.ReLU(inplace=True)
        self.pool3_img_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_img_3rd = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_img_3rd = nn.ReLU(inplace=True)
        self.conv4_2_img_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_img_3rd = nn.ReLU(inplace=True)
        self.conv4_3_img_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_img_3rd = nn.ReLU(inplace=True)
        self.pool4_img_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_img_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_img_3rd = nn.ReLU(inplace=True)
        self.conv5_2_img_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_img_3rd = nn.ReLU(inplace=True)
        self.conv5_3_img_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_img_3rd = nn.ReLU(inplace=True)
        self.pool5_img_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_1_flw_3rd = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1_flw_3rd = nn.ReLU(inplace=True)
        self.conv1_2_flw_3rd = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_flw_3rd = nn.ReLU(inplace=True)
        self.pool1_flw_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_flw_3rd = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_flw_3rd = nn.ReLU(inplace=True)
        self.conv2_2_flw_3rd = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_flw_3rd = nn.ReLU(inplace=True)
        self.pool2_flw_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_flw_3rd = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_flw_3rd = nn.ReLU(inplace=True)
        self.conv3_2_flw_3rd = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_flw_3rd = nn.ReLU(inplace=True)
        self.conv3_3_flw_3rd = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_flw_3rd = nn.ReLU(inplace=True)
        self.pool3_flw_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_flw_3rd = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_flw_3rd = nn.ReLU(inplace=True)
        self.conv4_2_flw_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_flw_3rd = nn.ReLU(inplace=True)
        self.conv4_3_flw_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_flw_3rd = nn.ReLU(inplace=True)
        self.pool4_flw_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_flw_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_flw_3rd = nn.ReLU(inplace=True)
        self.conv5_2_flw_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_flw_3rd = nn.ReLU(inplace=True)
        self.conv5_3_flw_3rd = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_flw_3rd = nn.ReLU(inplace=True)
        self.pool5_flw_3rd = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Feature extractor: first-person
        self.conv1_1_img_1st = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1_img_1st = nn.ReLU(inplace=True)
        self.conv1_2_img_1st = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_img_1st = nn.ReLU(inplace=True)
        self.pool1_img_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_img_1st = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_img_1st = nn.ReLU(inplace=True)
        self.conv2_2_img_1st = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_img_1st = nn.ReLU(inplace=True)
        self.pool2_img_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_img_1st = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_img_1st = nn.ReLU(inplace=True)
        self.conv3_2_img_1st = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_img_1st = nn.ReLU(inplace=True)
        self.conv3_3_img_1st = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_img_1st = nn.ReLU(inplace=True)
        self.pool3_img_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_img_1st = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_img_1st = nn.ReLU(inplace=True)
        self.conv4_2_img_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_img_1st = nn.ReLU(inplace=True)
        self.conv4_3_img_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_img_1st = nn.ReLU(inplace=True)
        self.pool4_img_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_img_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_img_1st = nn.ReLU(inplace=True)
        self.conv5_2_img_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_img_1st = nn.ReLU(inplace=True)
        self.conv5_3_img_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_img_1st = nn.ReLU(inplace=True)
        self.pool5_img_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_1_flw_1st = nn.Conv2d(2, 64, 3, padding=100)
        self.relu1_1_flw_1st = nn.ReLU(inplace=True)
        self.conv1_2_flw_1st = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_flw_1st = nn.ReLU(inplace=True)
        self.pool1_flw_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_flw_1st = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_flw_1st = nn.ReLU(inplace=True)
        self.conv2_2_flw_1st = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_flw_1st = nn.ReLU(inplace=True)
        self.pool2_flw_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_flw_1st = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_flw_1st = nn.ReLU(inplace=True)
        self.conv3_2_flw_1st = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_flw_1st = nn.ReLU(inplace=True)
        self.conv3_3_flw_1st = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_flw_1st = nn.ReLU(inplace=True)
        self.pool3_flw_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_flw_1st = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_flw_1st = nn.ReLU(inplace=True)
        self.conv4_2_flw_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_flw_1st = nn.ReLU(inplace=True)
        self.conv4_3_flw_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_flw_1st = nn.ReLU(inplace=True)
        self.pool4_flw_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_flw_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_flw_1st = nn.ReLU(inplace=True)
        self.conv5_2_flw_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_flw_1st = nn.ReLU(inplace=True)
        self.conv5_3_flw_1st = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_flw_1st = nn.ReLU(inplace=True)
        self.pool5_flw_1st = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Segmentation branch
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        self.score_fr = nn.Conv2d(4096, self.n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            self.n_classes, self.n_classes, 4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(
            self.n_classes, self.n_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            self.n_classes, self.n_classes, 16, stride=8, bias=False)

        # Matching branch
        self.softmax2d = nn.Softmax2d()
        self.resize_mask = nn.AdaptiveAvgPool2d((24, 33))
        self.conv1_match = nn.Conv2d(512, 512, 3, padding=1)
        self.relu1_match = nn.ReLU(inplace=True)
        self.pool1_match = nn.MaxPool2d(2, stride=2)
        self.conv2_match = nn.Conv2d(512, 256, 3, padding=1)
        self.relu2_match = nn.ReLU(inplace=True)
        self.pool2_match = nn.MaxPool2d(2, stride=2)
        self.fc_match = nn.Linear(256*6*8, self.n_features)

    def forward(self, img_3rd=None, img_1st=None, flw_3rd=None, flw_1st=None):
        score_3rd, f_img_flw_3rd = self.forward_3rd(img_3rd, flw_3rd)
        score_1st, f_img_flw_1st = self.forward_1st(img_1st, flw_1st)

        return score_3rd, f_img_flw_3rd, f_img_flw_1st

    def forward_3rd(self, img_3rd=None, flw_3rd=None):
        # Visual extractor
        if self.with_img:
            f_img_3rd = self.relu1_1_img_3rd(self.conv1_1_img_3rd(img_3rd))
            f_img_3rd = self.relu1_2_img_3rd(self.conv1_2_img_3rd(f_img_3rd))
            f_img_3rd = self.pool1_img_3rd(f_img_3rd)

            f_img_3rd = self.relu2_1_img_3rd(self.conv2_1_img_3rd(f_img_3rd))
            f_img_3rd = self.relu2_2_img_3rd(self.conv2_2_img_3rd(f_img_3rd))
            f_img_3rd = self.pool2_img_3rd(f_img_3rd)

            f_img_3rd = self.relu3_1_img_3rd(self.conv3_1_img_3rd(f_img_3rd))
            f_img_3rd = self.relu3_2_img_3rd(self.conv3_2_img_3rd(f_img_3rd))
            f_img_3rd = self.relu3_3_img_3rd(self.conv3_3_img_3rd(f_img_3rd))
            f_img_3rd = self.pool3_img_3rd(f_img_3rd)
            f_img_pool3_3rd = f_img_3rd

            f_img_3rd = self.relu4_1_img_3rd(self.conv4_1_img_3rd(f_img_3rd))
            f_img_3rd = self.relu4_2_img_3rd(self.conv4_2_img_3rd(f_img_3rd))
            f_img_3rd = self.relu4_3_img_3rd(self.conv4_3_img_3rd(f_img_3rd))
            f_img_3rd = self.pool4_img_3rd(f_img_3rd)
            f_img_pool4_3rd = f_img_3rd

            f_img_3rd = self.relu5_1_img_3rd(self.conv5_1_img_3rd(f_img_3rd))
            f_img_3rd = self.relu5_2_img_3rd(self.conv5_2_img_3rd(f_img_3rd))
            f_img_3rd = self.relu5_3_img_3rd(self.conv5_3_img_3rd(f_img_3rd))
            f_img_pool5_3rd = self.pool5_img_3rd(f_img_3rd)

        # Motion extractor
        if self.with_flw:
            f_flw_3rd = self.relu1_1_flw_3rd(self.conv1_1_flw_3rd(flw_3rd))
            f_flw_3rd = self.relu1_2_flw_3rd(self.conv1_2_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.pool1_flw_3rd(f_flw_3rd)

            f_flw_3rd = self.relu2_1_flw_3rd(self.conv2_1_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu2_2_flw_3rd(self.conv2_2_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.pool2_flw_3rd(f_flw_3rd)

            f_flw_3rd = self.relu3_1_flw_3rd(self.conv3_1_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu3_2_flw_3rd(self.conv3_2_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu3_3_flw_3rd(self.conv3_3_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.pool3_flw_3rd(f_flw_3rd)
            f_flw_pool3_3rd = f_flw_3rd

            f_flw_3rd = self.relu4_1_flw_3rd(self.conv4_1_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu4_2_flw_3rd(self.conv4_2_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu4_3_flw_3rd(self.conv4_3_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.pool4_flw_3rd(f_flw_3rd)
            f_flw_pool4_3rd = f_flw_3rd

            f_flw_3rd = self.relu5_1_flw_3rd(self.conv5_1_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu5_2_flw_3rd(self.conv5_2_flw_3rd(f_flw_3rd))
            f_flw_3rd = self.relu5_3_flw_3rd(self.conv5_3_flw_3rd(f_flw_3rd))
            f_flw_pool5_3rd = self.pool5_flw_3rd(f_flw_3rd)

        if self.with_img and self.with_flw:
            f_pool5_3rd = (f_img_pool5_3rd + f_flw_pool5_3rd) / 2
            f_pool4_3rd = (f_img_pool4_3rd + f_flw_pool4_3rd) / 2
            f_pool3_3rd = (f_img_pool3_3rd + f_flw_pool3_3rd) / 2
        elif self.with_img:
            f_pool5_3rd = f_img_pool5_3rd
            f_pool4_3rd = f_img_pool4_3rd
            f_pool3_3rd = f_img_pool3_3rd
        elif self.with_flw:
            f_pool5_3rd = f_flw_pool5_3rd
            f_pool4_3rd = f_flw_pool4_3rd
            f_pool3_3rd = f_flw_pool3_3rd
        else:
            raise(RuntimeError('Found both image and optical flow inputs are None.'))

        # Segmentation branch
        f_pool5_3rd = self.drop6(self.relu6(self.fc6(f_pool5_3rd)))
        f_pool5_3rd = self.drop7(self.relu7(self.fc7(f_pool5_3rd)))

        score_fr = self.score_fr(f_pool5_3rd)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(f_pool4_3rd)
        score_pool4c = score_pool4[
            :,:,5:5+upscore2.size(2),5:5+upscore2.size(3)]
        fuse_pool4 = upscore2 + score_pool4c
        upscore4 = self.upscore4(fuse_pool4)

        score_pool3 = self.score_pool3(f_pool3_3rd)
        score_pool3c = score_pool3[
            :,:,9:9+upscore4.size(2),9:9+upscore4.size(3)]
        fuse_pool3 = upscore4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)

        score = upscore8[
            :,:,31:31+img_3rd.size(2),31:31+img_3rd.size(3)].contiguous()

        # Matching branch
        mask = self.softmax2d(self.resize_mask(score))
        if self.with_img and self.with_flw:
            f_img_3rd = f_img_3rd * mask[:,[0]]
            f_flw_3rd = f_flw_3rd * mask[:,[1]]
            f_img_flw_3rd = f_img_3rd + f_flw_3rd
        elif self.with_img:
            f_img_3rd = f_img_3rd * mask[:,[0]]
            f_img_flw_3rd = f_img_3rd
        elif self.with_flw:
            f_flw_3rd = f_flw_3rd * mask[:,[1]]
            f_img_flw_3rd = f_flw_3rd
        else:
            raise(RuntimeError('Found both image and optical flow inputs are None.'))

        f_img_flw_3rd = self.pool1_match(self.relu1_match(self.conv1_match(f_img_flw_3rd)))
        f_img_flw_3rd = self.pool2_match(self.relu2_match(self.conv2_match(f_img_flw_3rd)))
        f_img_flw_3rd = f_img_flw_3rd.view(-1, 256*6*8)
        f_img_flw_3rd = self.fc_match(f_img_flw_3rd)

        return score, f_img_flw_3rd

    def forward_1st(self, img_1st=None, flw_1st=None):
        # Visual extractor
        if self.with_img:
            f_img_1st = self.relu1_1_img_1st(self.conv1_1_img_1st(img_1st))
            f_img_1st = self.relu1_2_img_1st(self.conv1_2_img_1st(f_img_1st))
            f_img_1st = self.pool1_img_1st(f_img_1st)

            f_img_1st = self.relu2_1_img_1st(self.conv2_1_img_1st(f_img_1st))
            f_img_1st = self.relu2_2_img_1st(self.conv2_2_img_1st(f_img_1st))
            f_img_1st = self.pool2_img_1st(f_img_1st)

            f_img_1st = self.relu3_1_img_1st(self.conv3_1_img_1st(f_img_1st))
            f_img_1st = self.relu3_2_img_1st(self.conv3_2_img_1st(f_img_1st))
            f_img_1st = self.relu3_3_img_1st(self.conv3_3_img_1st(f_img_1st))
            f_img_1st = self.pool3_img_1st(f_img_1st)

            f_img_1st = self.relu4_1_img_1st(self.conv4_1_img_1st(f_img_1st))
            f_img_1st = self.relu4_2_img_1st(self.conv4_2_img_1st(f_img_1st))
            f_img_1st = self.relu4_3_img_1st(self.conv4_3_img_1st(f_img_1st))
            f_img_1st = self.pool4_img_1st(f_img_1st)

            f_img_1st = self.relu5_1_img_1st(self.conv5_1_img_1st(f_img_1st))
            f_img_1st = self.relu5_2_img_1st(self.conv5_2_img_1st(f_img_1st))
            f_img_1st = self.relu5_3_img_1st(self.conv5_3_img_1st(f_img_1st))

        # Motion extractor
        if self.with_flw:
            f_flw_1st = self.relu1_1_flw_1st(self.conv1_1_flw_1st(flw_1st))
            f_flw_1st = self.relu1_2_flw_1st(self.conv1_2_flw_1st(f_flw_1st))
            f_flw_1st = self.pool1_flw_1st(f_flw_1st)

            f_flw_1st = self.relu2_1_flw_1st(self.conv2_1_flw_1st(f_flw_1st))
            f_flw_1st = self.relu2_2_flw_1st(self.conv2_2_flw_1st(f_flw_1st))
            f_flw_1st = self.pool2_flw_1st(f_flw_1st)

            f_flw_1st = self.relu3_1_flw_1st(self.conv3_1_flw_1st(f_flw_1st))
            f_flw_1st = self.relu3_2_flw_1st(self.conv3_2_flw_1st(f_flw_1st))
            f_flw_1st = self.relu3_3_flw_1st(self.conv3_3_flw_1st(f_flw_1st))
            f_flw_1st = self.pool3_flw_1st(f_flw_1st)

            f_flw_1st = self.relu4_1_flw_1st(self.conv4_1_flw_1st(f_flw_1st))
            f_flw_1st = self.relu4_2_flw_1st(self.conv4_2_flw_1st(f_flw_1st))
            f_flw_1st = self.relu4_3_flw_1st(self.conv4_3_flw_1st(f_flw_1st))
            f_flw_1st = self.pool4_flw_1st(f_flw_1st)

            f_flw_1st = self.relu5_1_flw_1st(self.conv5_1_flw_1st(f_flw_1st))
            f_flw_1st = self.relu5_2_flw_1st(self.conv5_2_flw_1st(f_flw_1st))
            f_flw_1st = self.relu5_3_flw_1st(self.conv5_3_flw_1st(f_flw_1st))

        # Matching branch
        if self.with_img and self.with_flw:
            f_img_flw_1st = f_img_1st + f_flw_1st
        elif self.with_img:
            f_img_flw_1st = f_img_1st
        elif self.with_flw:
            f_img_flw_1st = f_flw_1st
        else:
            raise(RuntimeError('Found both image and optical flow inputs are None.'))

        f_img_flw_1st = self.pool1_match(self.relu1_match(self.conv1_match(f_img_flw_1st)))
        f_img_flw_1st = self.pool2_match(self.relu2_match(self.conv2_match(f_img_flw_1st)))
        f_img_flw_1st = f_img_flw_1st.view(-1, 256*6*8)
        f_img_flw_1st = self.fc_match(f_img_flw_1st)

        return None, f_img_flw_1st

    def weights_init_from_fcn8s_shareview(self, fcn8s):
        self.conv1_1_img_3rd.weight.data = fcn8s['conv1_1_img.weight']
        self.conv1_1_flw_3rd.weight.data = fcn8s['conv1_1_flw.weight']
        self.conv1_1_img_3rd.bias.data = fcn8s['conv1_1_img.bias']
        self.conv1_1_flw_3rd.bias.data = fcn8s['conv1_1_flw.bias']

        init.xavier_normal(self.conv1_1_img_1st.weight.data)
        init.xavier_normal(self.conv1_1_flw_1st.weight.data)

        layers = [
            'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3',
        ]
        for layer in layers:
            l_img_3rd = getattr(self, layer+'_img_3rd')
            l_flw_3rd = getattr(self, layer+'_flw_3rd')
            l_img_1st = getattr(self, layer+'_img_1st')
            l_flw_1st = getattr(self, layer+'_flw_1st')
            l_img_3rd.weight.data = fcn8s[layer+'_img.weight']
            l_img_1st.weight.data = fcn8s[layer+'_img.weight']
            l_flw_3rd.weight.data = fcn8s[layer+'_flw.weight']
            l_flw_1st.weight.data = fcn8s[layer+'_flw.weight']
            l_img_3rd.bias.data = fcn8s[layer+'_img.bias']
            l_img_1st.bias.data = fcn8s[layer+'_img.bias']
            l_flw_3rd.bias.data = fcn8s[layer+'_flw.bias']
            l_flw_1st.bias.data = fcn8s[layer+'_flw.bias']

        layers = [
            'fc6', 'fc7',
            'upscore2', 'upscore4', 'upscore8',
            'score_fr', 'score_pool4', 'score_pool3',
        ]
        for layer in layers:
            l = getattr(self, layer)
            l.weight.data = fcn8s[layer+'.weight']
            if l.bias is not None:
                l.bias.data = fcn8s[layer+'.bias']

        layers = [
            'conv1_match',
            'conv2_match',
            'fc_match',
        ]
        for layer in layers:
            l = getattr(self, layer)
            init.xavier_normal(l.weight.data)
