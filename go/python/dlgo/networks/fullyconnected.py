import torch
from torch import nn

def DNN(input_shape, hidden_size):
    model = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=input_shape, out_features=hidden_size),
            nn.ReLU(),
            )

    return model
