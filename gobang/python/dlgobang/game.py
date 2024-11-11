'''
五子棋的设计难点：
    快速的判定达成5子的条件(需要设计一下数据结构)


棋盘 grid 每个存在的点，映射到底层的一个line set 上
    存在字典 key， 说明棋盘位置被占用
    棋子颜色通过底层line访问
    每个棋子映射到四条line， 这些line在每个状态下都和周围line做merge

棋盘对象拷贝时，需要对棋盘上的所有line 做深度拷贝
    但是point、move不需要（默认初始化后就不会进行修改值）
    直接返回自身对象


todo
    * 再思考思考deep copy提速的问题，仍然有优化空间
        目前的实现对于tree search 算法还是太慢了
            board size = 5
            search depth = 3
'''

__all__ = [
    'Board',
    'Gamestate',
    'Move',
]

# 定义全局变量
global_line_id = 0
Debug = False

from enum import Enum
import copy
import uuid
from dlgobang.gametypes import Point, Player
from dlgobang.utils import print_board

class Move:
    '''
        · 对point的封装
        · +弃权move
        todo 本类可能没有必要
    '''
    def __init__(self, point=None, is_resign=False):
        assert (point is not None) ^ is_resign # 断言二选一
        self.point = point
        self.is_play = self.point is not None
        self.is_resign = is_resign
    
    @classmethod
    def play(cls, point):
        return Move(point=point) # places a stone on the board.
    
    @classmethod
    def resign(cls):
        return Move(is_resign=True) # 弃权
    
    # log
    def __str__(self) -> str:
        if self.is_resign:
            return 'resign'
        return '(r %d, c %d)' % (self.point.x, self.point.y)

class Direction(Enum):
    HORIZONTAL = 1  # 水平
    VERTICAL = 2    # 竖直
    DIAGONAL_LEFT = 3  # 左斜线（从左上到右下）
    DIAGONAL_RIGHT = 4  # 右斜线（从右上到左下）
    
class Line():
    ''' 
        加速判定五子连线的数据结构， grid的每个point对应4个方向的line
        
        todo 不记录棋子集合, 只记录两端 和 长度。merge时长度相加
            类似union-find的优化思路
    '''
    def __init__(self, stones:set, color:Player, direction:Direction, id=None) -> None:
        self.stones = stones # set 需要考虑deep copy
        self.color = color # player（黑色、白色）
        self.dir = direction
        if id:
            self.id = id
        else:
            self.id = self.newid()

    def newid(self):
        global global_line_id  # 声明使用全局变量
        global_line_id += 1  # 递增ID
        return global_line_id  # 分配当前ID
    
    def merge_with(self, other):
        assert self.dir==other.dir and self.color==other.color
        stones_union = self.stones | other.stones # set union
        return Line(stones_union, self.color, self.dir) # add new obj only，don't touch old obj

    def __len__(self):
        return len(self.stones)
    
    def __deepcopy__(self, memodict={}): # 提速[0.1s->0.04s] 依旧慢了0.01s
        return Line(stones=self.stones, color=self.color, direction= self.dir, id=self.id)
    
    def log(self) -> str:
        return '-'.join([point.log() for point in self.stones])

class LineManager():
    def __init__(self, color:Player, point:Point, global_board, plines=None) -> None:
        self.color = color
        self.point = point # 几乎无用的变量 todo 待优化
        self.global_board = global_board
        
        if plines:
            self.plines = plines
        else:
            # 落子后组成4各单元素的line，后与全局line融合后跟新指针 (每个lm对应一个point，存储四个line指针)
            self.plines = {dirtype: Line(stones= set([point]), color=color, direction=dirtype)
                    for dirtype in [Direction.HORIZONTAL,
                                    Direction.VERTICAL,
                                    Direction.DIAGONAL_LEFT,
                                    Direction.DIAGONAL_RIGHT]}
            # record line to global board.lines map 
            for line in self.plines.values():
                self.global_board.lines[line.id] = line
    
    def __deepcopy__(self, memodict={}): # 提速[0.04s->0.03s] 持平之前
        return LineManager(self.color, self.point, self.global_board, plines=copy.deepcopy(self.plines, memodict)) 
    
    def merge(self, other, dirtype:Direction):
        '''
            同色&邻居 的两个line manager可以调用merge函数
        '''
        assert self.color == other.color
        l1 = self.plines[dirtype]
        l2 = other.plines[dirtype]
        
        new_line = l1.merge_with(l2)

        if Debug:
            # 用新obj newline替换所有牵扯到的line manager
            print('l1', l1.id, [point.pos for point in l1.stones])
            print('l2', l2.id, [point.pos for point in l2.stones])
            print('newline', new_line.id, [point.pos for point in new_line.stones])

        for point in new_line.stones:
            lm = self.global_board.grid[point.pos]
            lm.plines[dirtype] = new_line # 指针指向新对象
        
        # 删掉全局旧obj
        del self.global_board.lines[l1.id]
        del self.global_board.lines[l2.id]
        if Debug:
            print('[log] lines del', l1.id, l2.id)
        self.global_board.lines[new_line.id] = new_line
        
        # set winner if needed
        if len(new_line) >= 5:
            self.global_board.winner = new_line.color
            
    def log(self):
        s ='  lines:'
        s += ' '.join([line.log() for line in self.plines.values()])
        return s
    
class Board:
    def __init__(self, num_rows, num_cols) -> None:
        self.uid = uuid.uuid4()  # 随机生成 UID
        
        self.num_rows = num_rows # 棋盘范围 [1, num_rows]
        self.num_cols = num_cols # 棋盘范围 [1, num_cols]
        self.num_pos = num_cols * num_rows # 棋子总数 
        self.grid = dict() # key记录落子位置 {(rid, cid) : line Manager} board共享，只映射指针
        self.lines = dict() # {Line id : Line obj}
        
        self.winner = None
    
    def __deepcopy__(self, memodict={}):
        # 创建一个新的 Board 实例并注册到 memodict，以便缓存使用
        new_board = Board(self.num_rows, self.num_cols)
        memodict[id(self)] = new_board
        
        # 深拷贝 self.grid，但保持全局一致性
        new_board.grid = {}
        for key, lm in self.grid.items():
            # 检查 memodict 中是否已经深拷贝过这个 LineManager
            if id(lm) in memodict:
                new_board.grid[key] = memodict[id(lm)]
            else:
                # 深拷贝 LineManager，并将新的 global_board 设置为新 board
                new_lm = copy.deepcopy(lm, memodict)
                new_lm.global_board = new_board
                new_board.grid[key] = new_lm
                memodict[id(lm)] = new_lm
        
        # 深拷贝 lines
        new_board.lines = copy.deepcopy(self.lines, memodict)
        
        return new_board
    
    
    
    def full(self):
        return len(self.grid) >= self.num_pos # 棋盘占满
    
    def on_grid(self, point):
        return 1<= point.x <= self.num_rows and \
            1 <= point.y <= self.num_cols

    # 当前位置棋子颜色
    def get(self, point:Point):
        return self.get_pos(point.pos)
    
    def get_pos(self, pos):
        line_manager = self.grid.get(pos)
        if line_manager:
            return line_manager.color
        return None
        
    def place_stone(self, player:Player, point:Point):
        '''
            Note: 调用fun-place_stone()前确认point的正确性
        '''
        assert self.on_grid(point) # 不超出棋盘
        assert self.grid.get(point.pos) is None , 'Illegal play on %s' % str(point) # 空位置可落子

        # 落子
        cur_lm = LineManager(player, point, self)
        self.grid[point.pos] = cur_lm
        if Debug:
            print('[log] put grid', point.pos, self.uid)
            print('[log] 当前棋盘棋子', self.grid.keys())
            for pos, lm in self.grid.items():
                print('\t{}:{}'.format(pos, lm.log()))
                
            print('[log] 当前棋盘lid', ' | '.join([str(lid)+':'+line.log() for lid, line in self.lines.items()]))

        ## 8个候选点，相同颜色的邻居节点的 line manager
        directions = {
            Direction.HORIZONTAL: (1, 0),
            Direction.VERTICAL: (0, 1),  
            Direction.DIAGONAL_LEFT: (1, 1),  
            Direction.DIAGONAL_RIGHT: (1, -1),  
        }

        for dirtype, dir in directions.items():
            x,y = point.pos
            dx, dy = dir
            x1, y1 = x + dx, y + dy
            x2, y2 = x - dx, y - dy

            lm1 = self.grid.get((x1,y1)) # line manager 1
            lm2 = self.grid.get((x2,y2)) # line manager 2
            
            if lm1 and lm1.color == player: # 非空 & 同色
                if Debug:
                    print('lm1 call merge, grid get', (x1, y1))
                cur_lm.merge(lm1, dirtype) # 更新全局line状态
            if lm2 and lm2.color == player:
                if Debug:
                    print('lm2 call merge, grid get', (x2, y2))
                    print('[log] 底层board指向的对象不同', lm2.global_board.uid, self.uid)
                cur_lm.merge(lm2, dirtype)
    
    def have_five(self):
        return  self.winner is not None
    
    def five_color(self):
        return self.winner

class GameState:
    def __init__(self, board, next_player, prestate, last_move) -> None:
        self.uid = uuid.uuid4()  # 随机生成 UID
        
        self.board:Board = board # 当前棋面（Lines）
        self.next_player:Player = next_player # 本轮player
        self.prestate:GameState = prestate # 撤销move, 返回之前游戏状态
        self.last_move:Move = last_move # 上一步棋

    def apply_move(self, move):
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        if move.is_resign:
            next_board = self.board
        # 状态更新
        return GameState(next_board, self.next_player.next, self, move)
    
    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.BLACK, None, None)

    def is_over(self):
        if self.last_move is None:
            return False
        if self.board.have_five() or self.last_move.is_resign or self.board.full():
            return True
        return False
    
    def is_valid_move(self, move):
        # 棋局结束
        if self.is_over():
            return False
        # 认输
        if move.is_resign:
            return True
        # 空位置落子
        return self.board.get(move.point) is None
        
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
        
        if Debug:
            print('----call winner log ------')
            print_board(self.board)

        return self.board.winner
    
    def get_board_array(self):
        # 返回NXN的二维矩阵
        N = self.board.num_rows
        board = [[0]*N for _ in range(N)]
        for pos, lm in self.board.grid.items():
            x,y = pos
            color = lm.color
            value = 1 if color == Player.BLACK else 2
            board[x-1][y-1] = value
        
        return board