"""
@author:  muzishen
@contact: shenfei140721@126.com
"""

import copy
import torch
import random

import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from opt import opt
from torch.autograd import Variable
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
       # init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])




class PRM(nn.Module):
    def __init__(self,concatrandom=False):
        super(PRM, self).__init__()
        self.concatrandom = concatrandom
        self.conv_part = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False))
        self.conv_rest = nn.Sequential(nn.Conv2d(in_channels=2048 * 3, out_channels=1024, kernel_size=1, bias=False))
        self.addconv = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, bias=False))
        self.conv_part.apply(weights_init_kaiming)
        self.conv_rest.apply(weights_init_kaiming)
        self.addconv.apply(weights_init_kaiming)
        self.tanh = nn.Tanh()

    def relation_score(self, part, rest):
        part_score = self.conv_part(part)
        rest_score = self.conv_rest(rest)
        add_score = part_score + rest_score
        full_score = self.addconv(add_score)
        tanh_full_score = self.tanh(full_score)
        part_relation = part.mul(tanh_full_score )
        return part_relation

    def forward(self, x):
        p0 = x[:, :, 0:1, :]
        p1 = x[:, :, 1:2, :]
        p2 = x[:, :, 2:3, :]
        p3 = x[:, :, 3:4, :]

        if self.concatrandom == True and opt.mode == 'train':
            randr0 = random.randint(0,5)
            randr1 = random.randint(0,5)
            randr2 = random.randint(0,5)
            randr3 = random.randint(0,5)
          #  global r0, r1, r2, r3
            if randr0 == 0 :
                r0 = torch.cat((p1, p2, p3), dim=1)
            elif randr0 == 1:
                r0 = torch.cat((p1, p3, p2), dim=1)
            elif randr0 == 2:
                r0 = torch.cat((p2, p1, p3), dim=1)
            elif randr0 == 3:
                r0 = torch.cat((p2, p3, p1), dim=1)
            elif randr0 == 4:
                r0 = torch.cat((p3, p1, p2), dim=1)
            elif randr0 == 5:
                r0 = torch.cat((p3, p2, p1), dim=1)

            if randr1 == 0:
                r1 = torch.cat((p0, p2, p3), dim=1)
            elif randr1 == 1:
                r1 = torch.cat((p0, p3, p2), dim=1)
            elif randr1 == 2:
                r1 = torch.cat((p2, p0, p3), dim=1)
            elif randr1 == 3:
                r1 = torch.cat((p2, p3, p0), dim=1)
            elif randr1 == 4:
                r1 = torch.cat((p3, p0, p2), dim=1)
            elif randr1 == 5:
                r1 = torch.cat((p3, p2, p0), dim=1)

            if randr2 == 0:
                r2 = torch.cat((p0, p1, p3), dim=1)
            elif randr2 == 1:
                r2 = torch.cat((p0, p3, p1), dim=1)
            elif randr2 == 2:
                r2 = torch.cat((p1, p0, p3), dim=1)
            elif randr2 == 3:
                r2 = torch.cat((p1, p3, p0), dim=1)
            elif randr2 == 4:
                r2 = torch.cat((p3, p0, p1), dim=1)
            elif randr2 == 5:
                r2 = torch.cat((p3, p1, p0), dim=1)

            if randr3 == 0:
                r3 = torch.cat((p0, p1, p2), dim=1)
            elif randr3 == 1:
                r3 = torch.cat((p0, p2, p1), dim=1)
            elif randr3 == 2:
                r3 = torch.cat((p1, p0, p2), dim=1)
            elif randr3 == 3:
                r3 = torch.cat((p1, p2, p0), dim=1)
            elif randr3 == 4:
                r3 = torch.cat((p2, p0, p1), dim=1)
            elif randr3 == 5:
                r3 = torch.cat((p2, p1, p0), dim=1)

        else :
            r0 = torch.cat((p1, p2, p3), dim=1)
            r1 = torch.cat((p2, p3, p0), dim=1)
            r2 = torch.cat((p3, p0, p1), dim=1)
            r3 = torch.cat((p0, p1, p2), dim=1)

        p0_relation = self.relation_score(p0, r0)
        p1_relation = self.relation_score(p1, r1)
        p2_relation = self.relation_score(p2, r2)
        p3_relation = self.relation_score(p3, r3)

        return p0_relation, p1_relation, p2_relation, p3_relation

class RPN(nn.Module):

    def __init__(self, num_classes):
        super(RPN, self).__init__()
        feats = 256
        self.model = ResNet()
        self.model.load_param('/home/shenfei/.cache/torch/checkpoints/resnet50-19c8e357.pth')

        self.global_maxpool = nn.MaxPool2d(kernel_size=(opt.h // 16, opt.w // 16))  # GMP

        self.local_maxpool = nn.MaxPool2d(kernel_size=(opt.h//64, opt.w//16))
        self.Part_relation = PRM()

        self.reduction_g = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_0 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_1 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=feats, kernel_size=1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())


        self.reduction_g.apply(weights_init_kaiming)
        self.reduction_0.apply(weights_init_kaiming)
        self.reduction_1.apply(weights_init_kaiming)
        self.reduction_2.apply(weights_init_kaiming)
        self.reduction_3.apply(weights_init_kaiming)


        self.fc_id_2048_0 = nn.Linear(feats, num_classes)


        self.fc_id_256_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2 = nn.Linear(feats, num_classes)
        self.fc_id_256_3 = nn.Linear(feats, num_classes)
        self.fc_id_256_4 = nn.Linear(feats, num_classes)


        self.fc_id_2048_0.apply(weights_init_classifier)


        self.fc_id_256_1.apply(weights_init_classifier)
        self.fc_id_256_2.apply(weights_init_classifier)
        self.fc_id_256_3.apply(weights_init_classifier)
        self.fc_id_256_4.apply(weights_init_classifier)


    def forward(self, x):

        feature = self.model(x)
       # print(feature.shape)
        global_maxpool = self.global_maxpool(feature)  # global feature
        local_f = self.local_maxpool(feature)    # Local feature
        local_p0_relation, local_p1_relation, local_p2_relation, local_p3_relation = self.Part_relation(local_f)


        global_g_tri = self.reduction_g(global_maxpool).squeeze(dim=3).squeeze(dim=2)
        local_p0_f = self.reduction_0(local_p0_relation).squeeze(dim=3).squeeze(dim=2)
        local_p1_f = self.reduction_1(local_p1_relation).squeeze(dim=3).squeeze(dim=2)
        local_p2_f = self.reduction_2(local_p2_relation).squeeze(dim=3).squeeze(dim=2)
        local_p3_f = self.reduction_3(local_p3_relation).squeeze(dim=3).squeeze(dim=2)

        global_fc_f = self.fc_id_2048_0(global_g_tri)
        part0_f = self.fc_id_256_1(local_p0_f)
        part1_f = self.fc_id_256_2(local_p1_f)
        part2_f = self.fc_id_256_3(local_p2_f)
        part3_f = self.fc_id_256_4(local_p3_f)


        predict = torch.cat([ global_g_tri,  local_p0_f, local_p1_f, local_p2_f, local_p3_f], dim=1)
        return predict,  global_g_tri,  global_fc_f, part0_f, part1_f, part2_f, part3_f

if __name__ == '__main__':
  #  debug model structure
    net = RPN(num_classes=576)
    #print(net)
    PRM = PRM()
    print(sum(param.numel() for param in net.parameters()))
    print(sum(param.numel() for param in PRM.parameters()))
    input = Variable(torch.FloatTensor(128, 3, 256, 256))
    x = RPN.forward(net, input)

