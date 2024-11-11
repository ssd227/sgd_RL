from dlgobang.gametypes import Point, Player
import numpy as np

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    Player.BLACK: ' x ',
    Player.WHITE: ' o ',
}

def print_move(player, move):
    if move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.y - 1], move.point.x)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))
    
def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)

def coords_from_point(point):
    return '%s%d' % (
        COLS[point.y - 1],
        point.row
    )

# NOTE: MoveAge is only used in chapter 13, and doesn't make it to the main text.
# This feature will only be implemented in goboard_fast.py so as not to confuse
# readers in early chapters.
class MoveAge():
    def __init__(self, board):
        self.move_ages = - np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.x - 1, point.y - 1] = -1

    def add(self, point):
        self.move_ages[point.x - 1, point.y - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1