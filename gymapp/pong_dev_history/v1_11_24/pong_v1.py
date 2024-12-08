import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

# env = gym.make("Pong-v0", render_mode = 'human')
# observation, info = env.reset(seed=42)
# for i in range(1000):

#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)
#    print(f'i:{i}\n observation:{observation.shape}\n reward:{reward}\n terminated:{terminated}\n truncated:{truncated}\n info:{info}\n')
   
#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()


'''
收集一轮数据，训练一轮数据，效率还是太弱了。
肉眼可见有点效果

能不能，训练N个episode，然后训练，还能减小方差问题。
另外adam的学习率3e-4,影响大不大

todo
   先搞个V1版本，能用
   
   loss的定义需要重新推理
   NN的输出，配合loss，使用最方便的形式（是不是一定）
   
   MLP换成leNet
   
   测试有效果后，然后搞V2版本(模块化代码)--抽象数据收集、模型训练、模型评估
      收集数据剥离 experience.py
      NN训练、inference剥离
   
      代码改为agent的形式，方便后期PG、Q、AC等框架的规范化
      模型只是一个具体问题上，可以插拔的部分

todo 
   pg的参数空间搜索，参数用t-sen低纬度可视化，看看每一步更新的跳跃性。画一条游走曲线
   对比TRPO 和 PPO的参数搜索曲线，把所有参数点统一可视化。 随机参数固定，只换优化策略

实验在pong这个简单的模型上，一个有效的模型，大概需要多少轮迭代的数据
'''

 
# 原始图像 (210, 160, 3)
transform = T.Compose([
   T.ToPILImage(),   # 将输入的 NumPy 数组或 Tensor 转为 PIL 图像
   T.Lambda(lambda img: F.crop(img, top=34, left=0, height=160, width=160)),  # 裁剪行 [34:194]
   T.Grayscale(num_output_channels=1), # 转为灰度图像
   T.ToTensor(),
   T.Lambda(lambda x: (x > 0.35).float()),   # 将非零像素值变为 1 (二值化)
   T.Resize((80, 80), interpolation=T.InterpolationMode.NEAREST), # 缩放图像到 80x80, 插值防模糊
   nn.Flatten(start_dim=1)
])

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train_policy_gradient(env, policy_net, optimizer, device, episodes, gamma):
   episode_rewards = []
   for episode in range(episodes):
      state, info = env.reset(seed=42)
      state = transform(state) # 游戏状态图片预处理
      
      states, actions, rewards = [], [], []
      total_reward = 0 # 只为了记录每个episode的总reward，日志打印出来

      # 收集一局的样本，就训练policy_net
      while True:
         policy_net.eval()  # 切换到评估模式
         with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = policy_net(state_tensor)
            probs = action_probs.squeeze().cpu().detach().numpy()  # 策略网络output
            action = np.random.choice(len(probs), p=probs) # np提供的概率抽样
         
         states.append(state)
         actions.append(action)
         
         next_state, reward, terminated, truncated, info = env.step(action)
         # env.step(action + 1)  # 偏移动作到 {0, 2, 3}
         
         rewards.append(reward) # 大部分对局每轮的reward都是0
         total_reward += reward

         state = transform(next_state) # update state
         
         if terminated or truncated: # 一轮对局结束，跳出循环。 记为一个episode
            break
            # observation, info = env.reset()

      # 计算每步的折扣回报
      discounted_rewards = []
      cumulative = 0
      for reward in rewards[::-1]:
         cumulative = reward + gamma * cumulative
         discounted_rewards.insert(0, cumulative)
      
      discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
      discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # reward标准化，均值和期望。抑制一半的动作，推动一半的正向动作


      # 计算损失
      states_tensor = torch.cat(states, dim=0).to(device)
      print(states_tensor.shape)
      # states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
      actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
      
      action_probs = policy_net(states_tensor) # softmax后的概率
      action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1))) # todo 这行是干嘛的？

      loss = -(action_log_probs.squeeze() * discounted_rewards).sum() # 手动定义loss
      # todo 手动定义一个交叉熵函数，把dr的scale乘上去

      # 更新网络
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # ---------------- Log ------------------------
      episode_rewards.append(total_reward)  # 记录奖励
      print(f"Episode {episode + 1}, Total Reward: {total_reward}")

      # 打印游戏各轮reward的变化情况，可以观察模型是否变好。
      if (episode + 1) % 50 == 0:
         plt.plot(episode_rewards)
         plt.xlabel("Episode")
         plt.ylabel("Total Reward")
         # plt.show()
         plt.savefig('rewards'+str(episode)+'.png', dpi=300, bbox_inches="tight")
         plt.clf()

   return episode_rewards

def main():
   # env = gym.make("Pong-v4", render_mode='human')
   env = gym.make("Pong-v4", render_mode='rgb_array') # 不用可视化，快不少呀
   
   
   input_size = 80 * 80    # 预处理后的图像大小
   hidden_size = 128
   action_size = 6         # 动作空间 (停、向上、向下)
    
   # 定义模型、损失函数和优化器
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   policy_net = PolicyNetwork(input_size, hidden_size, action_size).to(device)
   optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
   print(device)

   # 训练策略网络
   rewards = train_policy_gradient(env, policy_net, optimizer, device=device, episodes=1000, gamma=0.99)

   # 保存模型
   torch.save(policy_net.state_dict(), "pong_pg_policy.pth")

def test():
   # 创建环境
   env = gym.make("Pong-v4", render_mode="rgb_array")
   state, info = env.reset(seed=42)
   
   for i in range(80):
      action = env.action_space.sample()  # this is where you would insert your policy
      print(f'action:{action}')
      # 实际上的动作有0、1、2、3、4、5， 不止单纯的上下，是不是还有加速操作
      # 假如只使用三种，静止、上、下，还得对PGNet的输出动作做映射
      
      state, reward, terminated, truncated, info = env.step(action)
      print(f'i:{i}\n observation:{state.shape}\n reward:{reward}\n terminated:{terminated}\n truncated:{truncated}\n info:{info}\n')
   

      if i%30==0:
         print(state.shape)
         processed_state = transform(state) # 处理状态
         print("Processed state shape:", processed_state.shape)  # 应该是 torch.Size([6400])
         print("Example processed state:", processed_state[:10]) # 输出部分数据以检查
         
         # 显示原始状态
         plt.figure(figsize=(12, 6))
         plt.subplot(1, 2, 1)
         plt.title("Original State")
         plt.imshow(state)
         plt.axis("off")

         # 显示处理后的状态
         plt.subplot(1, 2, 2)
         plt.title("Processed State")
         plt.imshow(processed_state.reshape(80,80), cmap="gray")
         plt.axis("off")

         plt.show()


if __name__ == "__main__":
   # test()
   main()