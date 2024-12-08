import gymnasium as gym
import torch

from rl.experience import ExperienceCollector
from rl.utils import show_rewards
from rl.model.cnn import LeNet5
from rl.agent.pg import PolicyAgent

'''
框架修改目标，更快速的收敛，模型性能增长快，稳步向好

MLP换成leNet（改数据三通道，差值信号）
    收集diff信号
        纯diff学不出来，比当前图像要差太多
        
    改造成三通道信号
        先用linear做
            效果挺好啊

        然后改造成cnn模型
            使用resnet-18,推理更慢了
                效果第一次尝试失败，50轮后参数就瘫掉了-21跑不出来
            
            使用resnet-tiny，减少参数量试试（虽然也没快多少）
                尝试失败，和cnn模型类似，50轮后参数就瘫掉 -21

            使用lenet, 极小的cnn
                同样的失败，30轮后没有随机性， -21
            
            lenet缩小版本
                同样的问题，初期和linear模型类似，后面迅速死掉，估计是输出都是一个值。
                
                3个fix操作：
                    限制softmax，使得抽样更平均
                        不太行，和缩小版本一致
                    
                    增加随机性（温度T控制）
                        只是把问题发生的时间给延迟了，中期可以跳出优化中的糟糕黑洞
                    
                    clipnorm（类似tpro的思路了）
                        有点作用，但是学习效率感觉一般般
                        最终在200轮还是劣化了，也是治标不治本
        
        (猜测) cnn的问题怕不是relu都死了，导致softmax后的action只有一个值

todo           
    应该参考参考dqn的feature设置
    
    拿现有的resnet做特征提取，只训练分类头，可能会好点
        但是特征构造就不好做了
            只给单帧图像
            或者前后两帧分开处理，在叠加features，最后用分类头去分类

doing 线程池并发收集数据
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
        
        # update state
        last_state = None
        state = agent.state_transform(raw_state)  # tensor [1, 80, 80]
        diff_state = state # 开局的ugly处理
        ch3_state = torch.stack([state, torch.zeros_like(state), diff_state], dim=1) # [1,C,M,N]  tensor [1, 3, 80, 80]
        
        # 收集1局样本数据
        while True:
            action = agent.select_action(ch3_state)
            # 环境交互         
            next_state, reward, terminated, truncated, info = env.step(action)

            # 收集数据
            agent.collector.record_decision(state=ch3_state, action=action, reward=reward)  # 大部分帧的reward都是0
            total_reward += reward  # 加了很多0

            # update state
            last_state = state
            state = agent.state_transform(next_state)  
            diff_state = state - last_state
            ch3_state = torch.stack([state, last_state, diff_state], dim=1)
            
            if terminated or truncated:  # 对局结束, 跳出循环
                agent.collector.complete_episode()  # 结束1轮对局（episode）的数据收集
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
            # agent.set_temperature(max(0, 0.5-episode*1e-3))

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
    # input [80, 80, 3]
    action_size = 6  # 动作空间 (停、向上、向下)
    policy_net = LeNet5(num_classes=action_size) # 6个输出
    # data
    collector = ExperienceCollector(gamma=0.99)  # gamma for discount reward

    # agent
    agent = PolicyAgent(model=policy_net,
                        device=device,
                        collector=collector,
                        action_size=action_size)

    # agent.set_temperature(0.5)

    # 训练策略网络
    train_policy_gradient(env, agent, episodes=episodes)


if __name__ == "__main__":
    main()
