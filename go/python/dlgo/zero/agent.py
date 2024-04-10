import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import TensorDataset, DataLoader
import dlgo
from dlgo import GameState
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye


__all__ = ['ZeroAgent', 'load_agent']

board_transform = v2.Compose([torch.from_numpy, v2.ToDtype(torch.float32, scale=True)])        

# 通过experience构造dataset (numpy to pytorch)
def get_dataset(experience, device=None):
    # numpy data
    states = experience.states              # [B, Cin, H, W] np.ndarray np.float64
    visit_counts = experience.visit_counts  # [B, moves_num]
    value_target = experience.rewards       # [B,] np.int64

    # pytorch data
    th_states = torch.from_numpy(states).to(torch.float32) # [B, Cin, H, W]
    th_visit_counts = torch.tensor(visit_counts, dtype=torch.float32) # [B, moves_num]
    th_policy_target = th_visit_counts / torch.sum(th_visit_counts, dim=1, keepdim=True) # [B, moves_num]
    th_value_target = torch.tensor(value_target, dtype=torch.float32).reshape(-1, 1) # [B, 1]    
    
    if device:
        th_states = th_states.to(device)
        th_policy_target = th_policy_target.to(device)
        th_value_target = th_value_target.to(device)

    return TensorDataset(th_states, th_policy_target, th_value_target)


class CrossEntropyLoss(nn.Module):
    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        y = logits - torch.max(logits, dim=1, keepdim=True)[0] # 防止exp(x)数值溢出  [B, 1]

        lse = torch.log(torch.sum(torch.exp(y), dim=1, keepdim=True)) # [B,1]
        return -1 * torch.sum(labels * (y-lse)) / batch_size

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        
    def expected_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class ZeroTreeNode:
    def __init__(self, state:GameState, value, priors, parent, last_move):
        assert len(priors) > 0
        
        self.state = state
        self.value = value
        
        # In the root of the tree, parent and last_move will be None.
        self.parent = parent
        self.last_move = last_move
        
        self.total_visit_count = 1
        self.branches = {} # 根据先验priors构造branches {move : prob}
        for move, p in priors.items():
            if state.is_valid_move(move): #TODO 没有valid_move? pass不算validmove
                self.branches[move] = Branch(p)
        
        if len(self.branches)==0:
            print('[sgd-log] no valid move')
            for cmove in priors.keys():
                if cmove.is_pass:
                    print('\tgetcha pass', state.is_over(), state.is_valid_move(cmove), cmove)

        self.children = {}

    # Returns a list of all possible moves from this node
    def moves(self):
        return self.branches.keys()

    # Allows adding new nodes into the tree
    def add_child(self, move, child_node):
        self.children[move] = child_node
        
    # Checks whether there’s a child node for a particular move
    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value
    
    ################## helper functions ###############
    def expected_value(self, move):
        assert move in self.branches
        return self.branches[move].expected_value()

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
        

class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move, c, device):
        self.model = model.to(device)
        self.encoder = encoder
        self.collector = None
        self.num_rounds = rounds_per_move
        self.c = c # 调节 UCT score 两部分比例的超参数
        self.device = device

    def select_move(self, game_state):
        root = self.create_node(game_state)

        '''
        一些细节bug
            比如在第n轮搜索时，遇到的叶子节点为end节点，需要直接continue循环
        '''
        for roundi in range(self.num_rounds):
            # print("[log][in tree search]select move, round-{}/{}".format(roundi+1, self.num_rounds))
            
            # Step1 正向搜索
            node = root
            next_move = self.select_branch(node) # 找当前score最大的move
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node) # stop at leaf node, then append new node

            # Step2 扩展新节点
            new_state = node.state.apply_move(next_move) # gameboard执行move，进入下一个棋盘状态
            if new_state.is_over():
                # 新棋盘状态为结束状态，中止当前探索（无效搜索），进入下一轮搜索
                continue
            child_node = self.create_node(new_state, move=next_move, parent=node)

            # Step3 反向更新
            move = next_move
            value = -1 * child_node.value   # 对手局面value * -1
            while node is not None:         
                node.record_visit(move, value) # upward update value
                
                move = node.last_move
                value = -1 * value
                node = node.parent
                
        # 收集训练数据（只考虑root节点的数据）
        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx))
                for idx in range(self.encoder.num_moves())
                ])
            self.collector.record_decision(root_state_tensor, visit_counts) # 拟合state:move的visit counts

        # 选择访问最多的move作为当前决策 
        # (原始版本，不限制move in eye) 初期会10次探索，会学不到pass。TODO 待尝试验证
        # return max(root.moves(), key=root.visit_count)
        
        # 选择访问最多的move作为当前决策 (加入眼，限制版本，非必要)
        best_move = max(root.moves(), key=root.visit_count) # 选择访问最多的move作为当前决策
        if best_move.point and is_point_an_eye(game_state.board,
                                            best_move.point,
                                            game_state.next_player):
            return dlgo.Move.pass_turn()
        else:
            return best_move

    def set_collector(self, collector):
        self.collector = collector

    # 探索现有展开状态（到达叶子节点）
    def select_branch(self, node):
        total_n = node.total_visit_count
        def score_branch(move):
            q = node.expected_value(move) # 观测Q
            p = node.prior(move) # 先验P
            n = node.visit_count(move)
            return q + self.c * p*np.sqrt(total_n)/(n + 1) # UCT score
        return max(node.moves(), key=score_branch)

    # 基于叶子节点，选择move->到达新状态game_state->agz net预估先验概率prior和state value
    def create_node(self, game_state, move=None, parent=None):
        board_tensor = self.encoder.encode(game_state)
        X = board_transform(board_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            priors, values = self.model(X)
        
        priors = torch.softmax(priors, dim=1).cpu().numpy().flatten() # [1, num_positions]
           
        # Add Dirichlet noise to the !!root node!!.
        if parent is None:
            noise = np.random.dirichlet(0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise  # 迪利克雷分布增加探索随机性
        
        move_priors = {self.encoder.decode_move_index(idx): p       
                            for idx, p in enumerate(priors)}
        
        value = values.item()
        assert len(move_priors)>0
        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
        
    def train(self, experience, lr, batch_size):
        # 训练数据准备 
        train_ds = get_dataset(experience, device=self.device)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        ce_loss = CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        loss_weights=[1.0, 0.5]

        # model fit
        self.train_epoch(
            model=self.model,
            dataloader=train_dl,
            optimizer=optimizer,
            loss_fns=[ce_loss, mse_loss],
            loss_weights = loss_weights
            )
    
    @staticmethod
    def train_epoch(model, dataloader, optimizer, loss_fns, loss_weights):
        iter = 0
        for xs, policy_target, value_target in dataloader:
            iter+=1
            print("iter:", iter, "batch_size", xs.shape[0])
            
            y = model(xs)
            
            targets = [policy_target, value_target]
            loss = 0
            for i in range(len(loss_fns)):
                loss += loss_weights[i] * loss_fns[i](y[i], targets[i])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # persist its encoder and model to disk
    def serialize(self, save_path):
        checkpoint = {'model_state_dict': self.model.state_dict(),}
        torch.save(checkpoint, save_path)


def load_agent(model, encoder, save_path, rounds_per_move, c, device):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return ZeroAgent(model, encoder, rounds_per_move, c, device=device)