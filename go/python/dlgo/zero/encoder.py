import numpy as np

from dlgo.goboard_fast import Move
from dlgo.gotypes import Player, Point


class ZeroEncoder:
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. opponent stones with 1, 2, 3, 4+ liberties
        # 8. 1 if we get komi
        # 9. 1 if opponent gets komi
        # 10. move would be illegal due to ko
        self.num_planes_ = 11

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player # 本轮落子的己方
        if game_state.next_player == Player.white:
            board_tensor[8] = 1 # 白子得komi补偿
        else:
            board_tensor[9] = 1
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None: # 空位置可以选择落子
                    if game_state.does_move_violate_ko(next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1 # check for ko
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color != next_player:
                        liberty_plane += 4 
                    board_tensor[liberty_plane][r][c] = 1 # 当前位置棋子关联string的气, 分敌我双方

        return board_tensor

    def encode_move(self, move):
        if move.is_play:
            return (self.board_size * (move.point.row-1) + (move.point.col-1)) # 类2d矩阵的1d内存寻址
        elif move.is_pass:
            return self.board_size * self.board_size # move:pass 放在最后一位
        raise ValueError('Cannot encode resign move')

    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        return self.board_size * self.board_size + 1
    
    def num_planes(self):
        return self.num_planes_

    def shape(self):
        return self.num_planes_, self.board_size, self.board_size
