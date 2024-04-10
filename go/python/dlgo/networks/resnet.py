import torch
import torch.nn as nn
import torch.nn.functional as F

'''
19*19 本身数量就不是很大，不使用pooling降低维度

'''

import torch
import torch.nn as nn

# 定义基本的卷积块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        # 输入与输出channel不同时，通过1*1的卷积核匹配到正确的
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x) # 残差的部分
        out = self.relu(out)

        return out

class ResNet34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=19*19):
        super(ResNet34, self).__init__()
        
        self.encoder_planes = 1
        self.in_planes = 32
        
        self.conv1 = nn.Conv2d(self.encoder_planes, self.in_planes, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)
        # print("log1", out.shape)

        out = self.layer1(out)
        # print("log2", out.shape)
        out = self.layer2(out)
        # print("log3", out.shape)
        out = self.layer3(out)
        # print("log4", out.shape)
        out = self.layer4(out)
        # print("log5", out.shape)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# 创建 ResNet34 模型实例
def resnet34():
    return ResNet34(BasicBlock, [3, 4, 6, 3])


class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes, encoder_planes):
        super(ResNet18, self).__init__()
        
        self.encoder_planes = encoder_planes
        self.in_planes = 32
        
        self.conv1 = nn.Conv2d(self.encoder_planes, self.in_planes, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)
        # print("log1", out.shape)

        out = self.layer1(out)
        # print("log2", out.shape)
        out = self.layer2(out)
        # print("log3", out.shape)
        out = self.layer3(out)
        # print("log4", out.shape)
        out = self.layer4(out)
        # print("log5", out.shape)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# 创建 ResNet34 模型实例
def resnet18(encoder_planes, num_classes):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, encoder_planes=encoder_planes)


