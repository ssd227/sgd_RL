import torch.nn as nn

# 保证原始大小
def cnn_small_cv(input_channel_num, output_channel_num):
    return nn.Sequential(*[
        nn.Conv2d(input_channel_num, 48, kernel_size=7, padding=3, stride=1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        
        nn.Conv2d(48, 32, kernel_size=5, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        
        nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        
        nn.Conv2d(32, output_channel_num, kernel_size=5, padding=2, stride=1, bias=False),
        nn.BatchNorm2d(output_channel_num),
        nn.ReLU(),

        nn.Flatten(start_dim=1),
    ])

def cnn_small_linear(output_channel_num, board_size):
    return nn.Sequential(*[
        nn.Linear(output_channel_num*board_size*board_size, 512),
        nn.ReLU(),
        nn.Linear(512, board_size*board_size),  
    ])

# 拆分成视觉模块和分类模块
def cnn_small(input_channel_num, board_size, output_channel_num=32):
    return nn.Sequential(*[
        cnn_small_cv(input_channel_num, output_channel_num),
        cnn_small_linear(output_channel_num, board_size)
    ])

# class CnnSmall(nn.Module):
#     def __init__(self, encoder_planes=1):
#         super().__init__()
#         self.encoder_planes = encoder_planes
        
#         self.conv1 = nn.Conv2d(self.encoder_planes, 48, kernel_size=7, stride=1, padding=2, bias=False) #[Cin, Cout, Kernel]
#         self.conv2 = nn.Conv2d(48, 32, kernel_size=5, stride=1, padding=2, bias=False) # 自动增加0 padding 保证输入输出维度
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False) 
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False) 
        
#         self.fc1 = nn.LazyLinear(out_features=19*19*2, bias=True)
#         self.fc2 = nn.Linear(19*19*2, 19*19, bias=True)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         # print("log1", x.shape)
#         x = F.relu(self.conv2(x))
#         # print("log2", x.shape)
#         x = F.relu(self.conv3(x))
#         # print("log3", x.shape)
#         x = F.relu(self.conv4(x))
#         # print("log4", x.shape)
        
#         x = torch.flatten(x, 1)
#         # print("log5-flatten", x.shape)
#         x = F.relu(self.fc1(x))
#         # print("log6-fc1", x.shape)
#         x = self.fc2(x)
#         # print("log7-fc3", x.shape)
#         return x