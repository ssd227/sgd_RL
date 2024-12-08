from .basic import Agent
from ..transform import img_transform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from ..data import PongDataset


class PolicyAgent(Agent):
    """An agent that uses a deep policy network to select moves."""

    def __init__(self, model, device, collector, action_size):
        super().__init__()
        self.model = model.to(device)
        self.collector = collector
        self.temperature = 0.0
        self.device = device

        self.action_size = action_size
        self.state_transform = img_transform

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    # 与环境交互
    def select_action(self, state):
        action_probs = None
        # 随温度升高，增加agent的随机性
        if np.random.random() < self.temperature:
            action_probs = torch.ones(self.action_size, device=self.device) / self.action_size  # Explore random moves.
        else:
            # inference
            self.model.eval()
            with torch.no_grad():
                state_tensor = state.to(self.device)
                action_logits = self.model(state_tensor)
                action_probs = nn.Softmax(dim=-1)(action_logits)

        if False:
            # Prevent move probs from getting stuck at 0 or 1. 
            eps = 1e-5
            action_probs = torch.clamp(action_probs, eps, 1 - eps)
            # Re-normalize to get another probability distribution.
            action_probs = action_probs / torch.sum(action_probs)  # norm(一范式)

        probs = action_probs.squeeze().cpu().detach().numpy()  # 策略网络output
        action = np.random.choice(len(probs), p=probs)  # np提供的概率抽样, 随机抽一次

        return action

    # 模型前向过程 (lr, clipnorm, batch_size 控制每步模型更新的幅度)
    def training(self, batch_size, clipnorm=1.0):
        # 训练数据准备
        # TODO 数据量过多，需不需要加工成dataloader的形式。随机梯度更新
        states_tensor = torch.cat(self.collector.states, dim=0)
        actions_tensor = torch.tensor(self.collector.actions, dtype=torch.int64)
        advantages_tensor = torch.tensor(self.collector.advantages, dtype=torch.float32)

        ds = PongDataset(states=states_tensor, actions=actions_tensor, advantages=advantages_tensor)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

        # TODO 训练参数，打穿到外部传入
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        ce_loss_fun = nn.CrossEntropyLoss()

        # training only one epoch
        for xStates, xActions, xAdvantages in dl:
            print(f'training batch data, B[{xStates.shape[0]}]')
            xStates = xStates.to(self.device)
            xActions = xActions.to(self.device)
            xAdvantages = xAdvantages.to(self.device)

            action_logits = self.model(xStates)
            # todo num_class 和 action保持一致，需要修改
            targets = nn.functional.one_hot(xActions, num_classes=6) * xAdvantages.unsqueeze(
                dim=-1)  # [B, C]*[B,1] = [B, C]
            loss = ce_loss_fun(action_logits, targets)

            # 更新网络
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipnorm)  # 裁剪梯度，阈值为1.0
            optimizer.step()

    # persist model to disk
    def persist(self, save_path):
        torch.save(self.model.state_dict(), save_path)


def load_policy_agent(model, device, collector, action_size, model_path=None):
    if model_path:
        model.load_state_dict(torch.load(model_path))  # 加载模型的参数

    return PolicyAgent(model=model,
                       device=device,
                       collector=collector,
                       action_size=action_size)
