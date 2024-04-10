"""
 一个输入，两个输出
"""

import torch.nn as nn
from .small import cnn_small_cv


__all__ = [
    'acnet_small',
]


class ACNet(nn.Module):
    def __init__(self, cv_model, output_channel_num, board_size):
        super(ACNet, self).__init__()

        # 视觉处理层
        self.cv_model = cv_model

        # 线性分类层(双输出层)
        linear_input_num = output_channel_num * (board_size*board_size)
        
        self.linear1 = nn.Linear(linear_input_num, 512)
        self.linear21 = nn.Linear(512, board_size*board_size) # for actions
        self.linear22 = nn.Linear(512, 1) # for values
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        cv_faltten_out = self.cv_model(x)
        out = self.relu(self.linear1(cv_faltten_out))
        
        actions_out = self.linear21(out)
        values_out = self.tanh(self.linear22(out)) 
        return actions_out, values_out

    
def acnet_small(input_channel_num, board_size, output_channel_num=32):
    cv_model = cnn_small_cv(input_channel_num, 
                            output_channel_num)

    return ACNet(cv_model= cv_model,
                output_channel_num=output_channel_num,
                board_size= board_size)
