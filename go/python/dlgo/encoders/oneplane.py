import numpy as np
from dlgo.encoders.base import Encoder
from dlgo.goboard import Point

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_height = board_size
        self.num_planes = 1

    # We can reference this encoder by the name "oneplane".
    def name(self):
        return 'oneplane'

    # To encode, we fill a matrix with 1 if the point contains one of the current player's stones, -1 if the point contains the opponent's stones and 0 if the point is empty.
    def encode(self, game_state):
        board_matrix = np.zeros(self.shape())
        next_player = game_state.next_player
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    continue
                if go_string.color == next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1
        return board_matrix

    # Turn a board point into an integer index.
    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    # Turn an integer index into a board point.
    def decode_point_index(self, index):  
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

def create(board_size):
    return OnePlaneEncoder(board_size)