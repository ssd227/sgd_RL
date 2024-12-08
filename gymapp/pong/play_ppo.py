'''
ChatGpt 给的ppo训练框架 还挺简洁

'''

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 1. 初始化网络和环境
# 策略网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_fc = nn.Linear(state_dim, 64)
        
        self.actor = nn.Linear(64, action_dim) # 动作预测 prob logits
        self.critic = nn.Linear(64, 1) # value预测 V(S)
    
    def forward(self, x):
        x = torch.relu(self.shared_fc(x))
        
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

# 2. 收集数据
def collect_trajectory(env, model, max_steps=200, gamma=0.99):
    states, actions, rewards, dones, log_probs = [], [], [], [], []
    state = env.reset()[0]
    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = model(state_tensor)
        
        # todo 又学到了新用法
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        next_state, reward, done, _, _ = env.step(action.item())
        
        # 存储数据
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(dist.log_prob(action))
        
        state = next_state
        if done:
            break
    
    return states, actions, rewards, dones, log_probs

# 3. 优势函数与目标值计算
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    gae = 0
    advantages = []
    next_value = 0
    for step in reversed(range(len(rewards))):
        td_error = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = td_error + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    return advantages

# 4. PPO 损失计算
def ppo_update(model, optimizer, old_states, old_actions, old_log_probs, returns, advantages, clip_eps=0.2):
    old_states = torch.tensor(old_states, dtype=torch.float32)
    old_actions = torch.tensor(old_actions, dtype=torch.long)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # 多轮更新
    for _ in range(10):  # 默认10轮梯度更新
        logits, values = model(old_states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy().mean()

        # 策略裁剪
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失
        value_loss = (returns - values).pow(2).mean() # 同步学习value
        # 总损失
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    # 5. 主循环
    env = gym.make("CartPole-v1")
    # env = gym.make("Pong-v4", render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(1000):
        states, actions, rewards, dones, log_probs = collect_trajectory(env, model)
        logits, values = model(torch.tensor(states, dtype=torch.float32))
        values = values.squeeze().detach().numpy()

        # 计算优势函数
        advantages = compute_gae(rewards, values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # 更新策略
        ppo_update(model, optimizer, states, actions, log_probs, returns, advantages)

        # 记录和输出奖励
        print(f"Episode {episode}: Total Reward: {sum(rewards)}")


if __name__ == "__main__":
    main()


