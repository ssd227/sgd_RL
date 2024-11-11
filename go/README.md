# DRL-Go Game（pytorch）

## 简介
    DRL入门项目，从零实现围棋交互环境和DRL算法。
    * 基于棋类游戏实践DRL基础算法
    * 类AlphaGo、AlphaGoZero的方式训练围棋AI-Agent
    * 使用React+Flask实现棋局交互UI

    主要参考书：《Deep learning and the game of go》 Max Pumperla & Kevin Ferguson

---
## 目录
    ./apps
        /dlgo （实验和应用）
        /tic_tac_toe
    ./python
        /dlgo （围棋算法源码）
        /tictactoe

    1、python为源码根目录。代码按照棋类分成两个库。库内包含训练AI-bot所需的代码，相互间独立。
    2、apps为实验目录。notebook展示各个类、函数的具体使用场景。实验中的心得体会也整理在noteboke里。


---
## 开发进度

## 1-DRL
- Tree Search
  - [X] MinMax
    - [X] Depth Pruning （深度剪枝）
    - [X] Alpha-Beta search （宽度剪枝）
  - [X] 蒙特卡洛搜索树-MCTS
- Q-learning
  - [X] Q(s,a)
  - [ ] DQN
- [X] Policy Gradient
- [X] Actor-Critic
  - [ ] PPO 优化
- [X] Alpha Go
- [X] Alpha Go Zero
---
## 2-游戏环境的逻辑与优化
* [X] 围棋
* [X] tic and toc
---
## 3-Encoder
* 棋局编码，构造可训练数据
---
## 4-数据
* 训练数据，用dataset封装
* 模型存储、加载效率
* 棋谱导入
---
## 5-模型对战模拟
* [ ] elo机制深入 TODO
---
## 6-前端框架和数据传输
    * React+Flask
      * 实际感受：这套框架不太适合写棋类UI。简单问题复杂化，效率不高。
    * 参考五子棋项目，围棋上没有实现

---
## 7-加速 TODO
    * CNN 模型在效率和准确度上的权衡，参考cs231N相关课件
    * GPU上的单个模型提供task pool，并发棋局模拟，提高数据搜集的速度
    * python的棋盘模拟实在是太慢了，换更快的语言 -> 更大的算力 -> 显得更智能

---
## 问题