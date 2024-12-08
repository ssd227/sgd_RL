'''
# 输入80*80*3

极小模型 lenet

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LeNet5']

class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积层 1: 输入通道 1（灰度图），输出通道 20，卷积核大小 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=2)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2: 输入通道 20，输出通道 50，卷积核大小 5x5
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2)  
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(10, 10)) # kernel 压缩成1X1，channels作为features

        # 全连接层 1
        self.fc1 = nn.Linear(50 * 10 * 10, 500)  # 50个 7x7 channel
        # 全连接层 2 (输出层)
        self.fc2 = nn.Linear(500, num_classes)
        
        

    def forward(self, x):
        # conv 1 + 激活 + pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # conv2 2 + 激活 + pooling
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1, end_dim=-1) # 注: dim-0 是Batch Size, 对单个样本展开
        
        # 全连接层 1 + 激活
        x = F.relu(self.fc1(x))
        # 全连接层 2 + 激活
        x = self.fc2(x)
        return x