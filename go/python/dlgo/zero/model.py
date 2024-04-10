"""
AlphaGo Zero net
    1 input
    2 output
"""

import torch.nn as nn

__all__ = ['agznet',]


def VisionNet(input_channel_num):
    return nn.Sequential(*[
        nn.Conv2d(input_channel_num, 64, kernel_size=3, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 64, kernel_size=3, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 64, kernel_size=3, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 64, kernel_size=3, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),

    ])


def PolicyNet(moves_num):
    return nn.Sequential(*[
        nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(out_features=moves_num, bias=True),
        # nn.Softmax(), // 在CEloss内实现了
    ])


def ValueNet():
    return nn.Sequential(*[
        nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(1),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(out_features=256, bias=True),
        nn.Linear(256,1),
        nn.Tanh(),
    ])


class AGZNet(nn.Module):
    def __init__(self, input_channel_num, moves_num):
        super(AGZNet, self).__init__()

        # 视觉处理层
        self.vision_net = VisionNet(input_channel_num)
        self.policy_net = PolicyNet(moves_num)
        self.value_net = ValueNet()

    def forward(self, x):
        vision_out = self.vision_net(x)
        actions_out = self.policy_net(vision_out)
        values_out = self.value_net(vision_out)
        return actions_out, values_out

def agznet(input_channel_num, moves_num):
    return AGZNet(input_channel_num, moves_num)
