'''
# 输入80*80*3

极小模型 缩小版lenet5
保证和线性模型差不多体量

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LeNet5']

class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=1, padding=2)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.AdaptiveAvgPool2d(output_size=(10, 10)) # kernel 压缩成1X1，channels作为features

        self.fc1 = nn.Linear(8 * 20 * 20, 200)
        self.fc2 = nn.Linear(200, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # print('分类头 输入', x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # 注: dim-0 是Batch Size, 对单个样本展开
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x