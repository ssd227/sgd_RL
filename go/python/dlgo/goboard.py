import copy
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from dlgo import zobrist


__all__ = [
    'Board',
    'GameState',
    'Move',
]


class IllegalMoveError(Exception):
    pass


# tag::fast_go_strings[]
class GoString:
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)  # <1>

    def without_liberty(self, point): # 调用后，建立新string对象 <2>
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point): # 调用后，建立新string对象
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)
# <1> `stones` and `liberties` are now immutable `frozenset` instances
# <2> The `without_liberty` methods replaces the previous `remove_liberty` method...
# <3> ... and `with_liberty` replaces `add_liberty`.
# end::fast_go_strings[]

    def merged_with(self, string):
        """Return a new string containing all stones in both strings."""
        assert string.color == self.color
        combined_stones = self.stones | string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | string.liberties) - combined_stones)

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties

    def __deepcopy__(self, memodict={}): # 提速一秒[2s->1s][speed up]
        return GoString(self.color, self.stones, copy.deepcopy(self.liberties))


class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD

    def place_stone(self, player, point):
        assert self.is_on_grid(point) # 超出棋盘
        if self._grid.get(point) is not None: # 往非空处落子，打印不合理位置
            print('Illegal play on %s' % str(point))
        assert self._grid.get(point) is None
        # 0. Examine the adjacent points.
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
        
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor): # 超出棋盘
                continue
            # 遍历neighbor 根据是否是string group，添加到双方阵营
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor) # 周边是空，自带一个气
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
                    
# tag::apply_zobrist[]
        new_string = GoString(player, [point], liberties) # 当前落子形成的新string group <1>
        # merger new string
        for same_color_string in adjacent_same_color:  # <2>
            new_string = new_string.merged_with(same_color_string)
        # 重新绑定每个point的 string group 映射
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string # 字典映射（todo union-find提速）

        self._hash ^= zobrist.HASH_CODE[point, player] # 添子的hash-xor操作 <3>
        # 消掉 point占据对手的气
        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point) # 产生新的string <4>
            if replacement.num_liberties: # 重新绑定 {stone：string}
                self._replace_string(other_color_string.without_liberty(point))
            else: # 气为0，吃子
                self._remove_string(other_color_string) # <5>
# <1> Until this line `place_stone` remains the same.
# <2> You merge any adjacent strings of the same color.
# <3> Next, you apply the hash code for this point and player
# <4> Then you reduce liberties of any adjacent strings of the opposite color.
# <5> If any opposite color strings now have zero liberties, remove them.
# end::apply_zobrist[]


# tag::unapply_zobrist[]
    def _replace_string(self, new_string): # （todo 优化）<1>
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():  # <2>
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string: # not self
                    self._replace_string(neighbor_string.with_liberty(point)) # 邻接对手棋子string 加气，并重新绑定 （todo 优化）
            self._grid[point] = None # 删掉当前的子

            self._hash ^= zobrist.HASH_CODE[point, string.color]  # <3>
# <1> This new helper method updates our Go board grid.
# <2> Removing a string can create liberties for other strings.
# <3> With Zobrist hashing, you need to unapply the hash for this move.
# end::unapply_zobrist[]

    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point):
        """Return the content of a point on the board.

        Returns None if the point is empty, or a Player if there is a
        stone on that point.
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):
        """Return the entire string of stones at a point.

        Returns None if the point is empty, or a GoString if there is
        a stone on that point.
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._hash() == other._hash()

    def __deepcopy__(self, memodict={}): # 提速一秒[1s->0.14s][speed up]
        copied = Board(self.num_rows, self.num_cols)
        # Can do a shallow copy b/c the dictionary maps tuples
        # (immutable) to GoStrings (also immutable)
        copied._grid = copy.copy(self._grid)
        copied._hash = self._hash
        return copied

# tag::return_zobrist[]
    def zobrist_hash(self):
        return self._hash
# end::return_zobrist[]


class Move:
    """Any action a player can play on a turn.
    Exactly one of is_play, is_pass, is_resign will be set.
    """
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign # 三种action类型（异或互斥）
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        """A move that places a stone on the board."""
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)

    def __str__(self):
        if self.is_pass:
            return 'pass'
        if self.is_resign:
            return 'resign'
        return '(r %d, c %d)' % (self.point.row, self.point.col)


# tag::init_state_zobrist[]
class GameState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_states = frozenset() # 空状态
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())}) # 谁下的 and 棋盘hash
        self.last_move = move
# end::init_state_zobrist[]

    def apply_move(self, move):
        """Return the new GameState after applying the move."""
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

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0

    @property
    def situation(self):
        return (self.next_player, self.board)

# tag::ko_zobrist[]
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash()) # 检查tuple是否在历史集合里
        return next_situation in self.previous_states
# end::ko_zobrist[]

    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_valid_move(self, move):
        if self.is_over(): # 游戏结束不允许落子
            return False
        if move.is_pass or move.is_resign: # legal action
            return True
        return (
            self.board.get(move.point) is None and  # 当前位置空
            not self.is_move_self_capture(self.next_player, move) and # 没有自抓
            not self.does_move_violate_ko(self.next_player, move)) # 没有ko

    def legal_moves(self): # 枚举所有合法move，不算在random的提速内
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        # These two moves are always legal.
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self) # 计算游戏局面
        return game_result.winner
