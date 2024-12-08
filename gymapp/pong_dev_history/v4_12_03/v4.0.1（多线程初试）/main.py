import gymnasium as gym
import torch

from rl.experience import ExperienceCollector
from rl.utils import show_rewards
from rl.model.linear import PolicyNetwork
from rl.agent.pg import PolicyAgent

'''
框架修改目标，更快速的收敛，模型性能增长快，稳步向好
      
doing 模块化代码 -- 模型训练、模型评估
    * MLP换成leNet（改数据三通道，差值信号）

    * 使用线程池子，3x3 轮收集9分数据，比单线程收集10轮数据要快3倍
    * model cpu版本公用的问题，实际上可以每个线程池子单独开一个model，load static dict
    
todo 
   * 机器够的话，同时训练多个模型，找效果最好的topN，参数作为下一轮的迭代
      * 多个agent相互筛选，进化选参数，胜者有资格迭代选参数。是不是比普通的pg更靠谱一些

   pg的参数空间搜索，参数用t-sen低纬度可视化，看看每一步更新的跳跃性。画一条游走曲线
   对比TRPO 和 PPO的参数搜索曲线，把所有参数点统一可视化。 随机参数固定，只换优化策略

实验：在pong这个简单的问题上，一个有效的模型，大概需要多少轮迭代的数据
'''


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
