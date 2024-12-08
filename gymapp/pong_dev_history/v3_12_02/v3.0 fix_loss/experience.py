import os
import numpy as np
import torch
from torch import nn

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]

class ExperienceCollector(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.episode_count = 0
        self.data_pair_count = 0
        
        # 多episode数据汇总
        self.states = []
        self.actions = []
        self.rewards = [] # 累计rewards
        self.advantages = []

        # episode统计量
        self._episode_states = []
        self._episode_actions = []
        self._episode_rewards = [] # 当前env反馈的Time t rewards

    def begin_episode(self):
        self._episode_states = []
        self._episode_actions = []
        self._episode_rewards = []

    def record_decision(self, state, action, reward):
        self._episode_states.append(state)
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)

    def complete_episode(self):
        # 本轮episode收集到的数据量
        N = len(self._episode_actions)
        self.episode_count += 1
        self.data_pair_count += N
        
        self.states += self._episode_states # [1]  List(tensor)
        self.actions += self._episode_actions # [2] List(Int)
        
        # 计算每步的折扣回报
        discounted_rewards = []
        cumulative = 0
        for reward in self._episode_rewards[::-1]:
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)

        self.rewards += discounted_rewards # [3] List(Float)

        # 计算advantage，此处用修正rewards替代 (reward标准化，均值和期望)   抑制一半的动作，推动一半的正向动作
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) 
        self.advantages += advantages.tolist() # [4] List(Float)
        
    def clean(self):
        # 每迭代M次，清理掉所有旧数据
        self.episode_count = 0
        self.data_pair_count = 0
        
        # 多episode数据汇总
        self.states = []
        self.actions = []
        self.rewards = [] # 累计rewards
        self.advantages = []
        

class ExperienceBuffer(object):
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    # 序列化
    def serialize(self, pth):
        experience = {
            'states' : self.states,
            'actions' : self.actions,
            'rewards' : self.rewards,
            'advantages' : self.advantages,
        }
        torch.save(experience, pth)
        
    def __len__(self):
        return len(self.states)

# 整合多个collector结果
def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages)

# 反序列化
def load_experience(pth):
    experience = torch.load(pth)    
    return ExperienceBuffer(
        states=np.array(experience['states']),
        actions=np.array(experience['actions']),
        rewards=np.array(experience['rewards']),
        advantages=np.array(experience['advantages']))
