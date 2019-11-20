import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ThirdThirdMsk(nn.Module):

    def __init__(self, n_classes=2, n_features=64, with_img=True, with_flw=True):
        super(ThirdThirdMsk, self).__init__()
        # Parameters
        self.n_classes = n_classes
        self.n_features = n_features
        self.with_img = with_img
        self.with_flw = with_flw

        # Feature extractor
        self.conv1_1_img = nn.Conv2d(4, 64, 3, padding=100)
        self.relu1_1_img = nn.ReLU(inplace=True)
        self.conv1_2_img = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_img = nn.ReLU(inplace=True)
        self.pool1_img = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_img = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_img = nn.ReLU(inplace=True)
        self.conv2_2_img = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_img = nn.ReLU(inplace=True)
        self.pool2_img = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_img = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_img = nn.ReLU(inplace=True)
        self.conv3_2_img = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_img = nn.ReLU(inplace=True)
        self.conv3_3_img = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_img = nn.ReLU(inplace=True)
        self.pool3_img = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_img = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_img = nn.ReLU(inplace=True)
        self.conv4_2_img = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_img = nn.ReLU(inplace=True)
        self.conv4_3_img = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_img = nn.ReLU(inplace=True)
        self.pool4_img = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_img = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_img = nn.ReLU(inplace=True)
        self.conv5_2_img = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_img = nn.ReLU(inplace=True)
        self.conv5_3_img = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_img = nn.ReLU(inplace=True)
        self.pool5_img = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_1_flw = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1_flw = nn.ReLU(inplace=True)
        self.conv1_2_flw = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_flw = nn.ReLU(inplace=True)
        self.pool1_flw = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1_flw = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_flw = nn.ReLU(inplace=True)
        self.conv2_2_flw = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_flw = nn.ReLU(inplace=True)
        self.pool2_flw = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1_flw = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_flw = nn.ReLU(inplace=True)
        self.conv3_2_flw = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_flw = nn.ReLU(inplace=True)
        self.conv3_3_flw = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_flw = nn.ReLU(inplace=True)
        self.pool3_flw = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1_flw = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_flw = nn.ReLU(inplace=True)
        self.conv4_2_flw = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_flw = nn.ReLU(inplace=True)
        self.conv4_3_flw = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_flw = nn.ReLU(inplace=True)
        self.pool4_flw = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1_flw = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_flw = nn.ReLU(inplace=True)
        self.conv5_2_flw = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_flw = nn.ReLU(inplace=True)
        self.conv5_3_flw = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_flw = nn.ReLU(inplace=True)
        self.pool5_flw = nn.MaxPool2d(2, stride=2, ceil_mode=True)

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
        self.resize_mask = nn.AdaptiveMaxPool2d((24, 33))
        self.conv1_match = nn.Conv2d(512, 512, 3, padding=1)
        self.relu1_match = nn.ReLU(inplace=True)
        self.pool1_match = nn.MaxPool2d(2, stride=2)
        self.conv2_match = nn.Conv2d(512, 256, 3, padding=1)
        self.relu2_match = nn.ReLU(inplace=True)
        self.pool2_match = nn.MaxPool2d(2, stride=2)
        self.fc_match = nn.Linear(256*6*8, self.n_features)

    def forward_once(self, img=None, flw=None):
        # Visual extractor
        if self.with_img:
            f_img = self.relu1_1_img(self.conv1_1_img(img))
            f_img = self.relu1_2_img(self.conv1_2_img(f_img))
            f_img = self.pool1_img(f_img)

            f_img = self.relu2_1_img(self.conv2_1_img(f_img))
            f_img = self.relu2_2_img(self.conv2_2_img(f_img))
            f_img = self.pool2_img(f_img)

            f_img = self.relu3_1_img(self.conv3_1_img(f_img))
            f_img = self.relu3_2_img(self.conv3_2_img(f_img))
            f_img = self.relu3_3_img(self.conv3_3_img(f_img))
            f_img = self.pool3_img(f_img)
            f_img_pool3 = f_img

            f_img = self.relu4_1_img(self.conv4_1_img(f_img))
            f_img = self.relu4_2_img(self.conv4_2_img(f_img))
            f_img = self.relu4_3_img(self.conv4_3_img(f_img))
            f_img = self.pool4_img(f_img)
            f_img_pool4 = f_img

            f_img = self.relu5_1_img(self.conv5_1_img(f_img))
            f_img = self.relu5_2_img(self.conv5_2_img(f_img))
            f_img = self.relu5_3_img(self.conv5_3_img(f_img))
            f_img_pool5 = self.pool5_img(f_img)

        # Motion extractor
        if self.with_flw:
            f_flw = self.relu1_1_flw(self.conv1_1_flw(flw))
            f_flw = self.relu1_2_flw(self.conv1_2_flw(f_flw))
            f_flw = self.pool1_flw(f_flw)

            f_flw = self.relu2_1_flw(self.conv2_1_flw(f_flw))
            f_flw = self.relu2_2_flw(self.conv2_2_flw(f_flw))
            f_flw = self.pool2_flw(f_flw)

            f_flw = self.relu3_1_flw(self.conv3_1_flw(f_flw))
            f_flw = self.relu3_2_flw(self.conv3_2_flw(f_flw))
            f_flw = self.relu3_3_flw(self.conv3_3_flw(f_flw))
            f_flw = self.pool3_flw(f_flw)
            f_flw_pool3 = f_flw

            f_flw = self.relu4_1_flw(self.conv4_1_flw(f_flw))
            f_flw = self.relu4_2_flw(self.conv4_2_flw(f_flw))
            f_flw = self.relu4_3_flw(self.conv4_3_flw(f_flw))
            f_flw = self.pool4_flw(f_flw)
            f_flw_pool4 = f_flw

            f_flw = self.relu5_1_flw(self.conv5_1_flw(f_flw))
            f_flw = self.relu5_2_flw(self.conv5_2_flw(f_flw))
            f_flw = self.relu5_3_flw(self.conv5_3_flw(f_flw))
            f_flw_pool5 = self.pool5_flw(f_flw)

        if self.with_img and self.with_flw:
            f_pool5 = (f_img_pool5 + f_flw_pool5) / 2
            f_pool4 = (f_img_pool4 + f_flw_pool4) / 2
            f_pool3 = (f_img_pool3 + f_flw_pool3) / 2
        elif self.with_img:
            f_pool5 = f_img_pool5
            f_pool4 = f_img_pool4
            f_pool3 = f_img_pool3
        elif self.with_flw:
            f_pool5 = f_flw_pool5
            f_pool4 = f_flw_pool4
            f_pool3 = f_flw_pool3
        else:
            raise(RuntimeError('Found both image and optical flow inputs are None.'))

        # Segmentation branch
        f_pool5 = self.drop6(self.relu6(self.fc6(f_pool5)))
        f_pool5 = self.drop7(self.relu7(self.fc7(f_pool5)))

        score_fr = self.score_fr(f_pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(f_pool4)
        score_pool4c = score_pool4[
            :,:,5:5+upscore2.size(2),5:5+upscore2.size(3)]
        fuse_pool4 = upscore2 + score_pool4c
        upscore4 = self.upscore4(fuse_pool4)

        score_pool3 = self.score_pool3(f_pool3)
        score_pool3c = score_pool3[
            :,:,9:9+upscore4.size(2),9:9+upscore4.size(3)]
        fuse_pool3 = upscore4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)

        score = upscore8[
            :,:,31:31+img.size(2),31:31+img.size(3)].contiguous()

        # Matching branch
        mask = self.softmax2d(self.resize_mask(score))
        if self.with_img and self.with_flw:
            f_img = f_img * mask[:,[1]]
            f_flw = f_flw * mask[:,[1]]
            f_img_flw = (f_img + f_flw) / 2
        elif self.with_img:
            f_img = f_img * mask[:,[1]]
            f_img_flw = f_img
        elif self.with_flw:
            f_flw = f_flw * mask[:,[1]]
            f_img_flw = f_flw
        else:
            raise(RuntimeError('Found both image and optical flow inputs are None.'))

        f_img_flw = self.pool1_match(self.relu1_match(self.conv1_match(f_img_flw)))
        f_img_flw = self.pool2_match(self.relu2_match(self.conv2_match(f_img_flw)))
        f_img_flw = f_img_flw.view(-1, 256*6*8)
        f_img_flw = self.fc_match(f_img_flw)

        return score, f_img_flw

    def forward(self, img1=None, img2=None, flw1=None, flw2=None):
        score1, f_img_flw_1 = self.forward_once(img1, flw1)
        score2, f_img_flw_2 = self.forward_once(img2, flw2)

        return score1, score2, f_img_flw_1, f_img_flw_2

    def weights_init_from_fcn8s_voc(self, fcn8s):
        init.xavier_normal(self.conv1_1_img.weight.data)
        init.xavier_normal(self.conv1_1_flw.weight.data)

        layers = [
            'conv1_2', 'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3',
        ]
        for layer in layers:
            l_img = getattr(self, layer+'_img')
            l_flw = getattr(self, layer+'_flw')
            l_img.weight.data = l_flw.weight.data = fcn8s[layer+'.weight']
            l_img.bias.data = l_flw.bias.data = fcn8s[layer+'.bias']

        layers = ['fc6', 'fc7']
        for layer in layers:
            l = getattr(self, layer)
            l.weight.data = fcn8s[layer+'.weight']
            l.bias.data = fcn8s[layer+'.bias']

        layers = ['upscore2', 'upscore4', 'upscore8']
        for layer in layers:
            l = getattr(self, layer)
            l.weight.data = fcn8s[layer+'.weight']

        layers = ['score_fr', 'score_pool4', 'score_pool3']
        for layer in layers:
            l = getattr(self, layer)
            init.xavier_normal(l.weight.data)

        layers = [
            'conv1_match',
            'conv2_match',
            'fc_match',
        ]
        for layer in layers:
            l = getattr(self, layer)
            init.xavier_normal(l.weight.data)

    def weights_init_from_fcn8s_shareview(self, fcn8s):
        self.load_state_dict(fcn8s, strict=False)

        layers = [
            'conv1_match',
            'conv2_match',
            'fc_match',
        ]
        for layer in layers:
            l = getattr(self, layer)
            init.xavier_normal(l.weight.data)
