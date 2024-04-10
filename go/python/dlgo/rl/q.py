import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2

from dlgo import encoders
from dlgo import goboard
from dlgo.agent import Agent
from dlgo.agent.helpers import is_point_an_eye

__all__ = [
    'QAgent',
    'load_q_agent',
]

board_transform = v2.Compose([torch.from_numpy, v2.ToDtype(torch.float32, scale=True)])


 # 通过experience构造dataset (numpy to pytorch)
def get_dataset(experience, device=None):
    B = len(experience)
    
    states = experience.states # [B, Cin, H, W] np.ndarray
    actions = experience.actions # [B, 1] np.int64
    rewards = experience.rewards # [B, 1] np.int64
    
    th_states = torch.from_numpy(states).to(torch.float32) # [B, Cin, H, W]
    th_actions = torch.tensor(actions, dtype=torch.long) # [B]
    th_rewards = torch.tensor(rewards, dtype=torch.float32).reshape(B, -1) # [B, 1]
    # print("dataset shape", th_states.shape, th_actions.shape, th_one_hot_actions.shape, th_rewards.shape)
    
    if device:
        th_states = th_states.to(device)
        th_actions = th_actions.to(device)
        th_rewards = th_rewards.to(device)

    return TensorDataset(th_states, th_actions, th_rewards)


class QAgent(Agent):
    def __init__(self, model, encoder, device):
        self.model = model.to(device)
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0
        self.device = device
        self.last_move_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        
        # 每个batch包含所有legal_moves
        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor) # 重复了但是需要（todo 可以优化，但是暂时没必要）
        if not moves:
            return goboard.Move.pass_turn()

        # board input
        board_tensors = np.array(board_tensors)
        input1 = board_transform(board_tensors).to(self.device)
        # print("[log] input1 shape", input1.shape)
        
        # move input (不使用one_hot, 直接输入值，模型中通过emb缩小编码dim)
        input2 = torch.tensor(moves, dtype=torch.long).to(self.device)

        # batch data of legal moves，模型计算value值
        with torch.no_grad():
            values = self.model(input1, input2)
            values = values.reshape(len(moves)) # 变回1d数组

        ranked_moves = self.rank_moves_eps_greedy(values)

        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if not is_point_an_eye(game_state.board,
                                   point,
                                   game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=moves[move_idx],
                    )
                self.last_move_value = float(values[move_idx])
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature: # 探索eplision
            values = torch.randn_like(values)
        ranked_moves =  torch.argsort(values)   # worst to best
        return ranked_moves.tolist()[::-1]      # best to worst

    
    def train(self, experience, lr=0.1, batch_size=128):
        # 训练数据准备 
        num_moves = self.encoder.board_width * self.encoder.board_height
        train_ds = get_dataset(experience, device=self.device)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        ce_loss = torch.nn.MSELoss()

        # model fit
        self.train_epoch(
            model=self.model,
            dataloader=train_dl,
            loss_fn=ce_loss,
            optimizer=optimizer)
    
    @staticmethod
    def train_epoch(model, dataloader, optimizer, loss_fn):
        iter = 0
        for xs, actions, rewards in dataloader:
            iter+=1
            print("iter:", iter, "batch_size", xs.shape[0])
            
            y = model(xs, actions)
            loss = loss_fn(y, rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def serialize(self, save_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'encoder': {
                'name':self.encoder.name(),
                'board_width':self.encoder.board_width,
                'board_height':self.encoder.board_height,
            }
        }
        torch.save(checkpoint, save_path)

    def diagnostics(self):
        return {'value': self.last_move_value}


def load_q_agent(model, device, encoder=None, save_path=None, ):
    assert (encoder is None) ^ (save_path is None) # 不能既接受现有encoder, 又从checkpoint中还原旧的encoder
    # 加载默认模型和encoder
    if save_path is None:
        return QAgent(model, encoder, device=device)
    
    else: # 从check_point中还原数据
        checkpoint = torch.load(save_path)
        # 模型参数加载
        model.load_state_dict(checkpoint['model_state_dict'])
        # 重新实例化encoder
        encoder_name =checkpoint['encoder']['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        board_width = checkpoint['encoder']['board_width']
        board_height = checkpoint['encoder']['board_height']
        encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
        # 实例化agent类
        return QAgent(model, encoder, device=device)