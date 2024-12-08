'''
线程池模式，数据采集可提速3倍

'''

import gymnasium as gym
import numpy as np
import torch

from collections import deque
from time import time
from torch import nn
from rl.transform import img_transform
from concurrent.futures import ThreadPoolExecutor, as_completed


class Worker:
    def __init__(self, env_name, model):
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.model = model
        self.total_reward = 0

    def run_episode(self):
        raw_state, info = self.env.reset(seed=42)
        state = img_transform(raw_state)

        done = False
        episode_data = []

        while not done:
            # 模型预测动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_logits = self.model(state_tensor)
            action_probs = nn.Softmax(dim=-1)(action_logits)
            probs = action_probs.squeeze().cpu().detach().numpy()
            action = np.random.choice(len(probs), p=probs)  # 概率抽样

            # 与环境交互
            next_state, reward, done, truncated, info = self.env.step(action)
            state = img_transform(next_state)
            self.total_reward += reward

            # 收集每一步数据
            episode_data.append((state, action, reward, self.total_reward))

        return episode_data


def collect_data_parallel(env_name, model, num_workers, num_episodes):
    result_queue = deque()
    workers = [Worker(env_name, model) for _ in range(num_workers)]

    for i in range(3):
        # 创建线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker.run_episode): worker for worker in workers for _ in range(num_episodes)}

            for future in as_completed(futures):
                episode_data = future.result()
                result_queue.extend(episode_data)

    return list(result_queue)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # logit
        return x


# 主程序
if __name__ == "__main__":
    # Model
    input_size = 80 * 80  # 预处理后的图像大小
    hidden_size = 128
    action_size = 6  # 动作空间 (停、向上、向下)
    policy_net = PolicyNetwork(input_size, hidden_size, action_size)

    # 数据收集
    start_time = time()
    num_workers = 3
    num_episodes_per_worker = 3
    data = collect_data_parallel("Pong-v4", policy_net, num_workers, num_episodes_per_worker)
    print(f"Collected {len(data)} steps in {time() - start_time:.2f} seconds.")
