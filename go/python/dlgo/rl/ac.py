""" actor critic method """

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from dlgo import encoders
from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


board_transform = v2.Compose([torch.from_numpy, v2.ToDtype(torch.float32, scale=True)])        

 # 通过experience构造dataset (numpy to pytorch)
def get_dataset(experience, num_moves, device=None):
    B = len(experience)
    
    # numpy data
    states = experience.states          # [B, Cin, H, W] np.ndarray np.float64
    actions = experience.actions        # [B,] np.int64
    advanatges = experience.advantages  # [B,] np.float64
    value_target = experience.rewards   # [B,] np.int64
    
    # pytorch data
    th_states = torch.from_numpy(states).to(torch.float32) # [B, Cin, H, W]
    
    th_actions = torch.tensor(actions, dtype=torch.long) # [B]
    th_one_hot_actions = F.one_hot(th_actions, num_moves) # [B, num_class]
    th_advantages = torch.tensor(advanatges, dtype=torch.float32).reshape(-1, 1) # [B, 1]
    th_policy_target = th_one_hot_actions * th_advantages # [B, num_class]
    
    th_value_target = torch.tensor(value_target, dtype=torch.float32).reshape(-1, 1) # [B, 1]    
    
    if device:
        th_states = th_states.to(device)
        th_policy_target = th_policy_target.to(device)
        th_value_target = th_value_target.to(device)

    return TensorDataset(th_states, th_policy_target, th_value_target)


class CrossEntropyLoss(nn.Module):
    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        y = logits - torch.max(logits, dim=1, keepdim=True)[0] # 防止exp(x)数值溢出  [B, 1]

        lse = torch.log(torch.sum(torch.exp(y), dim=1, keepdim=True)) # [B,1]
        return -1 * torch.sum(labels * (y-lse)) / batch_size


class ACAgent(Agent):
    def __init__(self, model, encoder, device):
        self.model = model.to(device)
        self.encoder = encoder
        self.collector = None
        self.last_state_value = 0
        self.device = device

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        X = board_transform(board_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions, values = self.model(X)
        move_probs = torch.softmax(actions, dim=1) # [1, num_positions]
        estimated_value = values.item() # 2d tensor -> float

        eps = 1e-6
        move_probs = torch.clamp(move_probs, eps, 1-eps)
        move_probs = move_probs / torch.sum(move_probs)
        move_probs = move_probs.cpu().numpy().flatten()

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(game_state.board,
                                            point,
                                            game_state.next_player)
            if move_is_valid and (not fills_own_eye):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,  # numpy.ndarray
                        action=point_idx, # numpy.int64
                        estimated_value=estimated_value
                    )
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

        
    def train(self, experience, lr=0.1, batch_size=128):
        # 训练数据准备 
        num_moves = self.encoder.num_points()
        train_ds = get_dataset(experience, num_moves=num_moves, device=self.device)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        ce_loss = CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        loss_weights=[1.0, 0.5]

        # model fit
        self.train_epoch(
            model=self.model,
            dataloader=train_dl,
            optimizer=optimizer,
            loss_fns=[ce_loss, mse_loss],
            loss_weights = loss_weights
            )
    
    @staticmethod
    def train_epoch(model, dataloader, optimizer, loss_fns, loss_weights):
        iter = 0
        for xs, policy_target, value_target in dataloader:
            iter+=1
            print("iter:", iter, "batch_size", xs.shape[0])
            
            y = model(xs)
            
            targets = [policy_target, value_target]
            loss = 0
            for i in range(len(loss_fns)):
                loss += loss_weights[i] * loss_fns[i](y[i], targets[i])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # persist its encoder and model to disk
    def serialize(self, save_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'encoder': {
                'name':self.encoder.name(),
                'board_width':self.encoder.board_width,
                'board_height':self.encoder.board_height,
            }
        }
        torch.save(checkpoint, save_path)

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(model, device, encoder=None, save_path=None,):
    assert (encoder is None) ^ (save_path is None) # 不能既接受现有encoder, 又从checkpoint中还原旧的encoder
    # 加载默认模型和encoder
    if save_path is None:
        return ACAgent(model, encoder, device=device)
    
    else: # 从check_point中还原数据
        checkpoint = torch.load(save_path)
        # 模型参数加载
        model.load_state_dict(checkpoint['model_state_dict'])
        # 重新实例化encoder
        encoder_name =checkpoint['encoder']['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        board_width = checkpoint['encoder']['board_width']
        board_height = checkpoint['encoder']['board_height']
        encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
        # 实例化agent类
        return ACAgent(model, encoder, device=device)
