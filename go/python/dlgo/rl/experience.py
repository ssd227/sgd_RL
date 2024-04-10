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
    def __init__(self):
        #数据汇总(不清空的变量) 只要本轮agent还在博弈, 不进入训练状态
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []

        # 每一步move需要记录的值
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)] # episode结束后，添加每个reward

        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


# (states, actions, rewards, advantages), 用作pg模型训练的四元组
class ExperienceBuffer(object):
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    # 序列化 save to disk
    def serialize(self, pth):
        experience = {
            'states' : self.states,
            'actions' : self.actions,
            'rewards' : self.rewards,
            'advantages' : self.advantages,
        }
        torch.save(experience, pth)

    # 在这里直接构造dataset和dataloader呢
    # def to_dataset():
    # def to_dataLoader():
        
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

# 反序列化, load from disk
def load_experience(pth):
    experience = torch.load(pth)    
    return ExperienceBuffer(
        states=np.array(experience['states']),
        actions=np.array(experience['actions']),
        rewards=np.array(experience['rewards']),
        advantages=np.array(experience['advantages']))
