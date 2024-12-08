import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

from experience import ExperienceCollector


'''
框架修改目标，更快速的收敛，模型性能增长快，稳步向好

训练记录
   使用M=10轮一迭代，效果反而变差了
      1、使用sgd就能改善吗？ 
         这pg对参数和训练也太敏感了吧，稍微变一点就训不出来了
         还是说吧M=10调小（理论上M=1时，就和v1一致了。但是效率更差）
         
            第一次尝试
               训练效果不显著
               batch size 32， dl多线程=4
               cuda错位跳出
            第二次尝试（训练一次感觉至少要半个小时）
               增大batch szie 256，dl改为单线程
               
               runtime 又又又炸了
               
Episode 214, Total Reward: -21.0
Episode 215, Total Reward: -15.0
Episode 216, Total Reward: -20.0
Episode 217, Total Reward: -19.0
Episode 218, Total Reward: -20.0
Episode 219, Total Reward: -17.0
train model, episode: 219
  File "C:\ProgramData\miniconda3\envs\gym\Lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "C:\ProgramData\miniconda3\envs\gym\Lib\site-packages\torch\multiprocessing\reductions.py", line 321, in reduce_tensor
    ) = storage._share_cuda_()
        ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ProgramData\miniconda3\envs\gym\Lib\site-packages\torch\storage.py", line 1145, in _share_cuda_
    return self._untyped_storage._share_cuda_(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: mapping of buffer object failed
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
      
      
      2、 advantage，应该先标准化，还是10轮后标准化
      
      3、还是说数据量增多了后，adam的学习率3e-4就不合适了


收集10轮数据，迭代一轮模型，还是好慢。（这种简单的训练架构不满足需求，至少要把机器占满吧，这收集数据的效率，要等到后年马月）
   并行多开env，独立收集10轮数据，然后统一训练一次
   
   多个agent相互筛选，进化选参数，胜者有资格迭代选参数。是不是比普通的pg更靠谱一些


doing 模块化代码 -- 模型训练、模型评估

   loss的定义需要重新推理
      NN的输出，配合loss，使用最方便的形式
   
   代码改为agent的形式，方便后期PG、Q、AC等框架的规范化
      NN训练、inference剥离
      MLP换成leNet

todo 
   pg的参数空间搜索，参数用t-sen低纬度可视化，看看每一步更新的跳跃性。画一条游走曲线
   对比TRPO 和 PPO的参数搜索曲线，把所有参数点统一可视化。 随机参数固定，只换优化策略

实验：在pong这个简单的问题上，一个有效的模型，大概需要多少轮迭代的数据
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

class PongDataset(Dataset):
    def __init__(self, states, actions, advantages):
        self.states = states  
        self.actions = actions
        self.advantages = advantages

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        advantage = self.advantages[idx]
        return state, action, advantage

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
   collector = ExperienceCollector(gamma=gamma)
   
   for episode in range(episodes):
      collector.begin_episode() # Episode 开始
      
      state, info = env.reset(seed=42) # env初始 观测图片
      state = transform(state) # 游戏状态图片预处理 (已经是torch的tensor）
      
      states, actions, rewards = [], [], []
      total_reward = 0 # 只为了记录每个episode的总reward，日志打印出来

      # 收集一局的样本，就训练policy_net
      while True:
         # inference
         policy_net.eval()  # 切换到评估模式
         with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = policy_net(state_tensor)
            probs = action_probs.squeeze().cpu().detach().numpy()  # 策略网络output
            action = np.random.choice(len(probs), p=probs) # np提供的概率抽样
         
         # 环境交互         
         next_state, reward, terminated, truncated, info = env.step(action)

         # 收集数据
         collector.record_decision(state=state, action=action, reward=reward) # 大部分帧的reward都是0

         total_reward += reward # 加了很多0
         state = transform(next_state) # update state
         
         if terminated or truncated: # 一轮对局结束，跳出循环
            collector.complete_episode() # 结束 一轮对局（episode）数据收集
            break
            # observation, info = env.reset()

      # X 轮数据收集，迭代一次模型参数
      M = 10
      if (episode + 1) % M == 0: 
         print('train model, episode:', episode)
         
         # 模型前向过程
         
         # 数据量过多，需不需要加工成dataloader的形式。随机梯度更新
         states_tensor = torch.cat(collector.states, dim=0).to(device)
         actions_tensor = torch.tensor(collector.actions, dtype=torch.int64).to(device)
         advantages_tensor = torch.tensor(collector.advantages, dtype=torch.float32).to(device)
         
         ds = PongDataset(states=states_tensor, actions=actions_tensor, advantages= advantages_tensor) 
         batch_size = 256  # 批量大小
         dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)
         
         for states_batch, actions_batch, advantages_batch in dl:
            action_probs = policy_net(states_batch) # softmax后的概率
            action_log_probs = torch.log(action_probs.gather(1, actions_batch.unsqueeze(1))) # todo 这行是干嘛的？

            loss = -(action_log_probs.squeeze() * advantages_batch).sum() # 定义nn形式的loss

            # 更新网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
         # 模型迭代结束，清除旧数据
         collector.clean()

      # ---------------- Log ------------------------
      # 记录奖励
      episode_rewards.append(total_reward)
      print(f"Episode {episode + 1}, Total Reward: {total_reward}")

      # 打印游戏各轮reward的变化情况，可以观察模型是否变好。
      if (episode + 1) % 50 == 0:
         plt.plot(episode_rewards)
         plt.xlabel("Episode")
         plt.ylabel("Total Reward")
         # plt.show()
         plt.savefig('rewards'+str(episode)+'.png', dpi=300, bbox_inches="tight")
         plt.clf()
         
         # 保存模型
         torch.save(policy_net.state_dict(), 'pong_pg_policy_v2.pth')

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


if __name__ == "__main__":
   main()