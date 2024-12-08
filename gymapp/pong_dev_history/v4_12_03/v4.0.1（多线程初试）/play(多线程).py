import gymnasium as gym
import numpy as np
import threading
import torch
from collections import deque
from time import time

from torch import nn
from rl.transform import img_transform


class Worker(threading.Thread):
    def __init__(self, env_name, model, result_queue):
        threading.Thread.__init__(self)
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.model = model
        self.result_queue = result_queue

    def run(self):
        raw_state, info = self.env.reset(seed=42)
        state = img_transform(raw_state)
        
        done = False
        total_reward = 0
        
        while not done:
            # 模型预测动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_logits = self.model(state_tensor)
            action_probs = nn.Softmax(dim=-1)(action_logits)
            probs = action_probs.squeeze().cpu().detach().numpy()  # 策略网络output
            action = np.random.choice(len(probs), p=probs)  # np提供的概率抽样, 随机抽一次

            # 与环境交互
            next_state, reward, done, truncated, info = self.env.step(action)
            state = img_transform(next_state)  # update state
            total_reward += reward

            # 将每个worker收集的数据放入队列
            self.result_queue.append((state, action, reward, total_reward))
        
        # print("worker-x Done!")


def collect_data_parallel(env_name, model, num_workers):
    result_queue = deque()
    workers = []

    # 启动多个worker线程
    for _ in range(num_workers):
        worker = Worker(env_name, model, result_queue)
        workers.append(worker)
    
    start_time = time()
    
    for worker in workers:
        worker.start()

    # 等待所有worker线程完成
    for worker in workers:
        worker.join()
    
    use_time = time() - start_time
    steps = len(result_queue)
    print(f"Collected {steps} steps in {use_time:.2f} seconds. avg {use_time/steps * 1000}")

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
    # model
    input_size = 80 * 80  # 预处理后的图像大小
    hidden_size = 128
    action_size = 6  # 动作空间 (停、向上、向下)
    policy_net = PolicyNetwork(input_size, hidden_size, action_size)
    
    # 并发
    data = collect_data_parallel("Pong-v4", policy_net, num_workers=1)
