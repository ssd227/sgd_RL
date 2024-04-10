'''
五子棋的设计难点：
    快速的判定达成5子的条件(需要设计一下数据结构)
    
'''

import copy
from .gobangtypes import Player, Point


__all__ = [
    'Board',
    'Gamestate',
    'Move',
]

class IllegalMoveError(Exception):
    pass

class Move:
    def __init__(self, point=None, is_resign=False):
        assert (point is not None) ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_resign = is_resign
    
    @classmethod
    def play(cls, point):
        """A move that places a stone on the board."""
        return Move(point=point)
    
    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    
    def __str__(self) -> str:
        if self.is_resign:
            return 'resign'
        return '(r %d, c %d)' % (self.point.row, self.point.col)


# 从棋盘上每个位置都可以获取4条line
# 棋子不用深拷贝，但是Line需要。否则新棋面下落子会影响line的连接关系
class Line():
    ''' todo Line 不记录集合, 只记录两端 和 长度
        merge时长度相加
    '''
    def __init__(self, color, stones, line_type) -> None:
        self.type = line_type # 横、竖、左斜、右斜(分别表示为1-2-3-4)
        self.color = color
        self.stones = frozenset(stones) # 不可变集合
    
    def append(self): # todo 单点的效率优化
        # 左下、左中、左上、中上
        pass
    
    def merge_with(self, other_line):
        assert self.type == other_line.type
        assert self.color == other_line.color

        combined_stones = self.stones | other_line.stones
        return Line(self.color, combined_stones, self.type)

    @property
    def len(self):
        return len(self.stones)
    
    # def __eq__(self, other): # 添加了可能存在问题[line不可hash] todo 没搞懂python底层原理
    #     return isinstance(other, Line) and \
    #         self.type == other.type and \
    #         self.color == other.color and \
    #         self.stones == other.stones

    def __deepcopy__(self, memodict={}): # 提速一秒[0.1s->0.03s][speed up]
        return Line(line_type = self.type, color=self.color, stones=self.stones)
    
    
class AdvancePoint(): # 更高级的点
    def __init__(self, color:Player, point:Point, lines_data, board_handle) -> None:
        self.color = color
        self.point = point
        self.lines_data = lines_data # map{1-4:}
        
        # 只用来更新self.line_num_map() # 每次删除旧的line，构造新的line（todo ！设计的有点丑啊，逻辑混在在advancePoint和board中，还夹杂一些逻辑在line中）
        self.board_handle = board_handle  # 通过集合删除是不是也很快

    def merge(self, other_adv_point, line_type):
        cur_line = self.lines_data[line_type]
        other_line = other_adv_point.lines_data[line_type]
        new_line = cur_line.merge_with(other_line)
        
        # 更新全局line的统计信息，方便快速找到n子相连的情况（好多对象操作啊）
        self.board_handle.len_line_map[cur_line.len] -= {cur_line}
        self.board_handle.len_line_map[other_line.len] -= {other_line}
        
        self.board_handle.len_line_map.setdefault(new_line.len, frozenset())
        self.board_handle.len_line_map[new_line.len] |= {new_line}
        
        return new_line

    def update_lines(self, line, line_type):
        self.lines_data[line_type] = line
    
    def __deepcopy__(self, memodict={}):
        copied = AdvancePoint(self.color, self.point, self.lines_data, self.board_handle)
        # 字典的值不可变, 浅层拷贝只拷贝指针
        copied.lines_data = copy.copy(self.lines_data)  
        return copied   


class Board:
    def __init__(self, num_rows, num_cols) -> None:
        self.num_rows = num_rows # [1, num_rows]
        self.num_cols = num_cols # [1, num_cols]
        self.position_num = self.num_cols * self.num_rows # 总棋盘数 
        self._grid = {} # 棋盘各位置
        self.len_line_map  = {} # n连棋子-快速查询 key: line.len, value: set() 方便快速的删除具体的line
    
    def place_stone(self, player, point):
        '''
        每个位置记录横、竖、左斜、右斜
        落子后，更新上述四个状态
        '''
        assert self.is_on_grid(point) # 不超出棋盘
        if self._grid.get(point) is not None: # 落子处非空，log(不合理位置)
             print('Illegal play on %s' % str(point))
        assert self._grid.get(point) is None
    
        # 落子后先单独组成4各单元素的line
        lines = {typeid: Line(color=player, stones=[point], line_type=typeid)
                 for typeid in range(1,5)}
        
        self.len_line_map.setdefault(1, frozenset())
        self.len_line_map[1] |= frozenset(lines.values()) # 长度为1的line
        
        # 构造新的advance_point
        cur_adv_point = AdvancePoint(color=player, point=point, lines_data=lines, board_handle=self)
        self._grid[point] = cur_adv_point # 位置坐标 与 高级点 绑定
        
        # 找到对应节点进行line merge, 然后更新对应节点的lines
        def same_color(target_point, comp_color):
            color = self.get(target_point)
            if color is not None:
                return color == comp_color
            return False
        
        row_adj = [poi for poi in point.row_neighbor() if same_color(poi, player)]
        col_adj = [poi for poi in point.col_neighbor() if same_color(poi, player)]
        main_diag_adj= [poi for poi in point.main_diag_neighbor() if same_color(poi, player)]
        secondary_diag_adj = [poi for poi in point.secondary_diag_neighbor() if same_color(poi, player)]
        
        # 更新四周邻居advance_point对应的Line子结构
        def update_adj(adjs, line_type):
            for poi in adjs: # 相邻位置
                other_adv_point = self.get_advance_point(poi)
                new_line = cur_adv_point.merge(other_adv_point, line_type)
                for point in new_line.stones:
                    self._grid[point].update_lines(new_line, line_type)
                    
        update_adj(row_adj, 1)
        update_adj(col_adj, 2)
        update_adj(main_diag_adj, 3)
        update_adj(secondary_diag_adj, 4)
          

    def is_on_grid(self, point):
        return 1<= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols
    
    # 当前位置棋子颜色
    def get(self, point): #当前棋子状态（黑/白）
        advance_point = self._grid.get(point)
        if advance_point is None:
            return None
        return advance_point.color
    
    def get_advance_point(self, point):
        advance_point = self._grid.get(point)
        if advance_point is None:
            return None
        return advance_point
    
    # 注意小bug, 可以存在4-1-4 9子的情况
    def have_five(self): # key只要大于5即可
        keys = [k for k in self.len_line_map.keys() if k >= 5]
        # print('[Log] have five:', keys) 
        if len(keys) == 0:
            return False, None
        key = keys[0]
        return  True, key # 存在一个大于5的Line
    
    def five_color(self):
        ok, kid = self.have_five()
        if ok:
            lines_set = self.len_line_map[kid]
            line = next(iter(lines_set)) # get one line from set
            return line.color
        return None
    
    def __deepcopy__(self, memodict={}):
        copied = Board(self.num_rows, self.num_cols)
        copied._grid = copy.deepcopy(self._grid) # AdvancePoint下面那层集合可变
        copied.len_line_map = copy.deepcopy(self.len_line_map)
        return copied

class GameState:
    def __init__(self, board, next_player, previous, last_move) -> None:
        self.board = board # 当前棋面
        self.next_player = next_player # 本轮player
        self.previous = previous # 根据这个状态可以撤销move,链式返回
        self.last_move = last_move # 上一步棋
        
        self.stone_count = 0 # 有效落子数, check 棋盘已满.(todo 但是也可以根据是否存在legal_move来添加一个pass move来处理)

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        if move.is_resign:
            next_board = self.board
            
        # 状态更新
        new_game_state = GameState(next_board, self.next_player.other, self, move)
        if move.is_play:
            new_game_state.stone_count = self.stone_count +1
        if move.is_resign:
            new_game_state.stone_count = self.stone_count
            
        return new_game_state
    
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        if self.last_move is None:
            return False
        # 存在五子的line 或者 没有可以落子的位置
        if (self.board.have_five()[0] or # 5+子连线
            self.last_move.is_resign or # 弃权
            self.stone_count == self.board.position_num): # 总落子数=棋盘数
            return True
        return False
    
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_resign:
            return True
        return self.board.get(move.point) is None # 空位置落子
        
    def legal_moves(self):
        moves = []
        for row in range(1, self.board.num_rows+1):
            for col in range(1,self.board.num_cols+1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.resign())
        return moves
    
    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        return self.board.five_color() # 直接查询5子Line color