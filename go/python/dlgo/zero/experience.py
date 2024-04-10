import os
import numpy as np
import torch
from torch import nn

__all__ = [
    'ZeroExperienceCollector',
    'ZeroExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []

# (states, actions, rewards, advantages), 用作pg模型训练的四元组
class ZeroExperienceBuffer(object):
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    # 序列化 save to disk
    def serialize(self, pth):
        experience = {
            'states' : self.states,
            'visit_counts' : self.visit_counts,
            'rewards' : self.rewards,
        }
        torch.save(experience, pth)
        
    def __len__(self):
        return len(self.states)

# 整合多个collector结果
def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ZeroExperienceBuffer(
        combined_states,
        combined_visit_counts,
        combined_rewards)


# 反序列化, load from disk
def load_experience(pth):
    experience = torch.load(pth)    
    return ZeroExperienceBuffer(
        states=np.array(experience['states']),
        visit_counts=np.array(experience['visit_counts']),
        rewards=np.array(experience['rewards']))
