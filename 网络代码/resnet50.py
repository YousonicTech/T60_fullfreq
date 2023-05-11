# -*- coding: utf-8 -*-
"""
@file      :  resnet50.py
@Time      :  2022/8/17 17:43
@Software  :  PyCharm
@summary   :  Resnet50
@Author    :  Bajian Xiang
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import os


class ResNet50(nn.Module):
    def __init__(self, ln_out=1):
        super(ResNet50, self).__init__()
        # input = torch.Size([batch, 3, 224, 224])
        self.resnet_out = 1000
        self.fc1_out = 512
        self.fc2_out = 256
        self.fc3_out = 128
        self.fc4_out = 64
        self.ln_out = ln_out

        # After Resnet = torch.Size([batch, 1000])
        self.model_weight_path = '/mnt/sda/xbj/0910_t60_resnet50/model_pt/resnet50-19c8e357.pth'
        self.resnet = resnet50(pretrained=False)


        # 第一个conv改成单通道
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 加载预训练模型并且把不需要的层去掉

        resnet50_weights = torch.load(self.model_weight_path)
        resnet50_weights['conv1.weight'] = resnet50_weights['conv1.weight'].sum(1, keepdim=True)
        self.resnet.load_state_dict(resnet50_weights)

        # After fc = torch.Size([batch, 1])
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.resnet_out, out_features=self.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc1_out),
            nn.Dropout(0.3),

            nn.Linear(in_features=self.fc1_out, out_features=self.fc2_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc2_out),
            nn.Dropout(0.2),

            nn.Linear(in_features=self.fc2_out, out_features=self.fc3_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc3_out),
            nn.Dropout(0.1),

            nn.Linear(in_features=self.fc3_out, out_features=self.fc4_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc4_out),
            nn.Dropout(0.1),

            nn.Linear(in_features=self.fc4_out, out_features=self.ln_out),
        )


    def forward(self, x):
        y = self.resnet(x)
        z = self.fc(y)
        return z

if __name__ == "__main__":

    x = torch.randn(4, 3, 224, 224)
    resnet50 = ResNet50()
    y = resnet50(x)
    print(y.shape)



