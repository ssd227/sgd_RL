import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchvision import transforms as T
import torchvision.transforms.functional as F


# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 创建 Pong 环境
env = gym.make("Pong-v4", render_mode="rgb_array")
env.seed(SEED)
state_shape = (1, 80, 80)  # 状态为单通道的 80x80 图像
n_actions = env.action_space.n

# 超参数
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 64  # 批量大小
LR = 1e-4  # 学习率
MEMORY_SIZE = 100000  # 经验池大小
MIN_MEMORY = 10000  # 经验池中最小样本数以开始训练
EPSILON_START = 1.0  # 初始 ε
EPSILON_END = 0.1  # 最小 ε
EPSILON_DECAY = 1e-6  # ε 衰减
TARGET_UPDATE = 1000  # 更新目标网络的频率

# 经验池
memory = deque(maxlen=MEMORY_SIZE)


# 原始图像 (210, 160, 3)
img_transform = T.Compose([
    T.ToPILImage(),  # 将输入的 NumPy 数组或 Tensor 转为 PIL 图像
    T.Lambda(lambda img: F.crop(img, top=34, left=0, height=160, width=160)),  # 裁剪行 [34:194]
    T.Grayscale(num_output_channels=1),  # 转为灰度图像
    T.ToTensor(),
    T.Lambda(lambda x: (x > 0.35).float()),  # 将非零像素值变为 1 (二值化)
    T.Resize((80, 80), interpolation=T.InterpolationMode.NEAREST),  # 缩放图像到 80x80, 插值防模糊
    # nn.Flatten(start_dim=1)
])


# Q 网络
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # Conv2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # Conv3
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),  # Fully Connected Layer
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Output layer
        )

    def forward(self, x):
        return self.network(x)

# 初始化网络
policy_net = DQN(state_shape, n_actions).to("cuda")
target_net = DQN(state_shape, n_actions).to("cuda")
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# 计算 ε
def get_epsilon(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * step)

# 选择动作
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)  # 随机动作
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
        with torch.no_grad():
            q_values = policy_net(state_tensor) # 注意：policy select action
        return q_values.argmax().item()

# 存储经验
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 训练
def train():
    if len(memory) < MIN_MEMORY:
        return

    batch = random.sample(memory, BATCH_SIZE) # todo 并没有按照重要性采样，重要性和td target成正比，学习率和td_target 成反比
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to("cuda")
    actions = torch.tensor(actions, dtype=torch.long).to("cuda")
    rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to("cuda")
    dones = torch.tensor(dones, dtype=torch.float32).to("cuda")

    # 当前 Q 值
    q_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # 目标 Q 值 (Double DQN)
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1) # 注意：使用ploicy net 选择action
        next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    num_episodes = 1000
    global_step = 0
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED)
        state = img_transform(state).numpy()  # 初始状态预处理
        done = False
        total_reward = 0

        while not done:
            epsilon = get_epsilon(global_step)
            action = select_action(state, epsilon)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = img_transform(next_state).numpy()
            store_experience(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_step += 1

            train()

            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()



if __name__ == "__main__":
    main()

