

import gymnasium as gym
import torch

from rl.experience import ExperienceCollector
from rl.utils import show_rewards
from rl.model.linear import PolicyNetwork
from rl.agent.pg import PolicyAgent


#############################################
def train_policy_gradient(env, agent, episodes):
    episode_rewards = []  # 统计每轮游戏reward，画折线图

    for episode in range(episodes):
        total_reward = 0  # 只为了记录每个episode的总reward，日志打印出来
        agent.collector.begin_episode()  # Episode 开始
        raw_state, info = env.reset(seed=42)  # env初始 观测图片
        state = agent.state_transform(raw_state)  # 游戏状态图片预处理 (已经是torch的tensor）

        # 收集一局的样本，就训练policy_net
        while True:
            action = agent.select_action(state)
            # 环境交互         
            next_state, reward, terminated, truncated, info = env.step(action)

            # 收集数据
            agent.collector.record_decision(state=state, action=action, reward=reward)  # 大部分帧的reward都是0
            total_reward += reward  # 加了很多0

            state = agent.state_transform(next_state)  # update state

            if terminated or truncated:  # 一轮对局结束，跳出循环
                agent.collector.complete_episode()  # 结束 一轮对局（episode）数据收集
                break
                # observation, info = env.reset()

        # 收集X轮数据，迭代一次模型参数
        M = 10
        if (episode + 1) % M == 0:
            print('train model, episode:', episode)

            # 数据模型自己保留，不用传来传去
            agent.training(batch_size=256, clipnorm=1.0)

            # 模型迭代结束，清除旧数据
            agent.collector.clean()

        # ---------------- Log ------------------------
        # 记录奖励
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # 打印游戏各轮reward的变化情况，可以观察模型是否变好。
        if (episode + 1) % 50 == 0:
            # 展示每轮episode的rewards变化情况
            show_rewards(episode, episode_rewards)
            # 保存模型参数
            agent.persist('pong_pg_policy_v3.pth')

    return episode_rewards


def main():
    # env = gym.make("Pong-v4", render_mode='human')
    env = gym.make("Pong-v4", render_mode='rgb_array')  # 不用可视化，快不少呀

    # 训练设置
    episodes = 1000  # 和环境交互1000轮

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # model
    input_size = 80 * 80  # 预处理后的图像大小
    hidden_size = 128
    action_size = 6  # 动作空间 (停、向上、向下)
    policy_net = PolicyNetwork(input_size, hidden_size, action_size)

    # data
    collector = ExperienceCollector(gamma=0.99)  # gamma for discount reward

    # agent
    agent = PolicyAgent(model=policy_net,
                        device=device,
                        collector=collector,
                        action_size=action_size)

    # 训练策略网络
    train_policy_gradient(env, agent, episodes=episodes)


if __name__ == "__main__":
    main()
