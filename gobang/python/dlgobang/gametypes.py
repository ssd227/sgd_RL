

__all__ = ['Point', 'Player',]

from enum import Enum

class Player(Enum):
    BLACK = 0
    WHITE = 1

    @property
    def next(self):
        return Player.BLACK if self == Player.WHITE else Player.WHITE

class Point:
    def __init__(self, row, col) -> None:
        self.x = row
        self.y = col
    
    @property
    def pos(self):
        return self.x, self.y
    
    # 直接邻节点
    def up(self):
        return Point(self.x-1, self.y)
    def down(self):
        return Point(self.x+1, self.y)
    def left(self):
        return Point(self.x, self.y-1)
    def right(self):
        return Point(self.x, self.y+1)
    # 对角邻节点
    def left_up(self):
        return Point(self.x-1, self.y-1)
    def left_down(self):
        return Point(self.x+1, self.y-1)
    def right_up(self):
        return Point(self.x-1, self.y+1)
    def right_down(self):
        return Point(self.x+1, self.y+1)   
        
    # 横、竖、左斜、右斜, 各两个邻居节点
    def row_neighbor(self):
        return self.left(), self.right()
    def col_neighbor(self):
        return self.up(), self.down()
    def main_diag_neighbor(self):
        return self.left_up(), self.right_down()
    def secondary_diag_neighbor(self):
        return self.left_down, self.right_up()
    
    def log(self):
        return '({},{})'.format(self.x, self.y)