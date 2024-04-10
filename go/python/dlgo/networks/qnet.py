import torch
import torch.nn as nn
import torch.nn.functional as F

from .small import cnn_small_cv


__all__ = [
    'qnet_small',
]

class Qnet(nn.Module):
    def __init__(self, cv_model, output_channel_num, board_size, embedding_dim):
        super(Qnet, self).__init__()
        
        # input1 视觉处理层
        self.cv_model = cv_model
        # input2  move emb
        self.emb = nn.Embedding(num_embeddings=board_size*board_size,
                                embedding_dim=embedding_dim)
        # 线性分类层
        linear_input_num = output_channel_num*board_size*board_size + \
                            embedding_dim # one hot     
        self.linear1 = nn.Linear(linear_input_num, 512)
        self.linear2 = nn.Linear(512, 1) # 输出为单个logit
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, action):
        # input 1
        cv_faltten_out = self.cv_model(x)
        # input 2
        action_emb = self.emb(action)

        out = torch.cat((cv_faltten_out, action_emb), dim=1)
        out = self.relu(self.linear1(out))
        out = self.tanh(self.linear2(out)) 
        return out
    
    
def qnet_small(input_channel_num, board_size, embedding_dim, output_channel_num=32):
    cv_model = cnn_small_cv(input_channel_num, 
                            output_channel_num)

    return Qnet(cv_model= cv_model,
                output_channel_num=output_channel_num,
                board_size= board_size,
                embedding_dim= embedding_dim)
