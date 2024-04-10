import enum
from collections import namedtuple

class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Point(namedtuple('Point', 'row col')):
    # 直接邻节点
    def up(self):
        return Point(self.row-1, self.col)
    def down(self):
        return Point(self.row+1, self.col)
    def left(self):
        return Point(self.row, self.col-1)
    def right(self):
        return Point(self.row, self.col+1)
    
    # 对角邻节点
    def left_up(self):
        return Point(self.row-1, self.col-1)
    def left_down(self):
        return Point(self.row+1, self.col-1)
    def right_up(self):
        return Point(self.row-1, self.col+1)
    def right_down(self):
        return Point(self.row+1, self.col+1)   
        
    # 横、竖、左斜、右斜, 各两个节点
    def row_neighbor(self):
        return (self.left(), self.right())
    def col_neighbor(self):
        return (self.up(), self.down())
    def main_diag_neighbor(self):
        return (self.left_up(), self.right_down())
    def secondary_diag_neighbor(self):
        return (self.left_down, self.right_up())
    
    def __deepcopy__(self, memodict={}):
        # These are very immutable.
        return self