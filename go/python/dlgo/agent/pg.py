"""Policy gradient learning."""

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard


__all__ = [
    'PolicyAgent',
    'load_policy_agent',
]


go_transform = v2.Compose([torch.from_numpy, v2.ToDtype(torch.float32, scale=True)])


 # 通过experience构造dataset (numpy to pytorch)
def get_dataset(experience, num_moves, device=None):
    B = len(experience)
    
    states = experience.states # [B, Cin, H, W] np.ndarray
    actions = experience.actions # [B, 1] np.int64
    rewards = experience.rewards # [B, 1] np.int64
    
    th_states = torch.from_numpy(states).to(torch.float32) # [B, Cin, H, W]
    th_actions = torch.tensor(actions, dtype=torch.long) # [B,]
    th_one_hot_actions = F.one_hot(th_actions, num_moves) # [B, num_class]
    th_rewards = torch.tensor(rewards, dtype=torch.long).reshape(B, -1) # [B, 1]
    
    xs = th_states
    ys = th_one_hot_actions * th_rewards # auto-broadcast [B, num_class]
    if device:
        xs = xs.to(device)
        ys = ys.to(device)
    return TensorDataset(xs, ys)

class CrossEntropyLoss(nn.Module):
    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        y = logits - torch.max(logits, dim=1, keepdim=True)[0] # 防止exp(x)数值溢出  [B, 1]

        lse = torch.log(torch.sum(torch.exp(y), dim=1, keepdim=True)) # [B,1]
        return -1 * torch.sum(labels * (y-lse)) / batch_size

class PolicyAgent(Agent):
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder, device):
        self.model = model.to(device)
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
        self.device = device

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        X = go_transform(board_tensor).unsqueeze(0).to(self.device)

        # 随温度升高，增加agent的随机性
        if np.random.random() < self.temperature:
            # Explore random moves.
            move_probs = torch.ones(num_moves) / num_moves
        else:
            # Follow our current policy.
            with torch.no_grad():
                move_probs = torch.softmax(self.model(X), dim=1)

        # todo 实际部署时就不用这个策略了吧？直接max选move
        # Prevent move probs from getting stuck at 0 or 1. 
        eps = 1e-5
        move_probs = torch.clamp(move_probs, eps, 1-eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / torch.sum(move_probs) # norm(一范式)
        move_probs = move_probs.cpu().numpy().flatten() # to numpy 1d
        
        # Turn the probabilities into a ranked list of moves.
        # 随机无放回采样（todo 效率可能不太行）
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx) # 反解出point
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,  # numpy.ndarray
                        action=point_idx # numpy.int64
                    )
                return goboard.Move.play(point)
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    # lr, clipnorm, batch_size  控制每一步更新模型变动的幅度
    def train(self, experience, lr, clipnorm=1.0, batch_size=512):
        # 训练数据准备 
        num_moves = self.encoder.board_width * self.encoder.board_height
        train_ds = get_dataset(experience, num_moves, device=self.device)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        ce_loss = CrossEntropyLoss()

        # fit
        self.train_epoch(
            model=self.model,
            dataloader=train_dl,
            loss_fn=ce_loss,
            optimizer=optimizer,
            clipnorm=clipnorm)
    
    @staticmethod
    def train_epoch(model, dataloader, optimizer, loss_fn, clipnorm):
        iter = 0
        for x, labels in dataloader:
            iter+=1
            print("iter:", iter, "batch_size", x.shape[0])
            
            y = model(x)
            loss = loss_fn(y,labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)  # 裁剪梯度，阈值为1.0
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

def load_policy_agent(model, device, encoder=None, save_path=None, ):
    assert (encoder is None) ^ (save_path is None) # 不能既接受现有encoder, 又从checkpoint中还原旧的encoder
    # 加载默认模型和encoder
    if save_path is None:
        return PolicyAgent(model, encoder, device=device)
    
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
        return PolicyAgent(model, encoder, device=device)