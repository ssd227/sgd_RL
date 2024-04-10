import copy
from .gotypes import Player

class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign # 三种action类型（异或互斥）
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign
    
    # 通过类的method 直接构造对应的action
    @classmethod
    def play(cls, point):
        return Move(point=point)
    
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    
class GoString():
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)
        
    def remove_liberty(self, point):
        self.liberties.remove(point)
    
    def add_liberty(self, point):
        self.liberties.add(point)
        
    def merged_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones)
    
    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties
    
class Board():
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {} # 每个位置map所属的string group

    def place_stone(self, player, point):
        assert self.is_on_grid(point) # 超出棋盘
        assert self._grid.get(point) is None #放置位置为空
        
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = [] # 落子周围的气
        
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor): # 超出棋盘
                continue
            # 遍历neighbor 根据是否是string group，添加到双方阵营
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor) # 周边是空，自带一个气
            elif neighbor_string.color == player: # 同阵营的string
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else: # 不同阵营的string
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
                    
        new_string = GoString(player, [point], liberties) # 当前落子形成的新string group
        
        # merger new string
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        # 重新绑定每个point的 string group 映射
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string # 字典映射（可以改成union-find提速）
        
        # 消掉对手的气 by current point
        for other_color_string in adjacent_opposite_color:
            other_color_string.remove_liberty(point)
        # 吃子
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string) # 删掉整个string group
    
    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string: # not self
                    neighbor_string.add_liberty(point) # todo 由于重复可以提速
            self._grid[point] = None
    
    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols
    
    def get(self, point): # 当前当前point的颜色， None代表空
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point): #返回当前point关联的string
        string = self._grid.get(point)
        if string is None:
            return None
        return string
    

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous # 记录上一个状态，构成state的link_list
        self.last_move = move
    
    # 保证move is legal
    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)
    
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
    
    def is_over(self):
        if self.last_move is None: #开局 last_move == None
            return False
        if self.last_move.is_resign: # 一方弃权，游戏结束
            return True
        
        second_last_move = self.previous_state.last_move
        if second_last_move is None: # 保证上上手不为空
            return False
        return self.last_move.is_pass and second_last_move.is_pass # 双方都pass，游戏结束
    
    @property
    def situation(self):
        return (self.next_player, self.board)
    
    ###########################################################################################
    ######################################  valid check #######################################
    
    def is_valid_move(self, move):
        if self.is_over(): # 游戏结束不允许再落子
            return False
        if move.is_pass or move.is_resign: # legal action
            return True
        return (
            self.board.get(move.point) is None and  # 当前位置空
            not self.is_move_self_capture(self.next_player, move) and # 没有自抓
            not self.does_move_violate_ko(self.next_player, move)) # 没有ko
    
    ###########################  两个下完后才能通过状态确认是否legal的检查  ######################
    # 只要所在位置不为空， move is legal
       
    # 保证move is legal
    def is_move_self_capture(self, player, move): # todo 为什么player是函数参数
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point) # 下完当前子没气了。[但是这一步如果下不出来呢。程序直接assert失败]
        return new_string.num_liberties == 0
    
    # 保证move is legal
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        
        # 要比较所有历史state
        past_state = self.previous_state
        while past_state is not None:
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False