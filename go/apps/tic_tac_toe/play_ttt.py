import sys
sys.path.append('/playground/sgd_deep_learning/sgd_rl/go/python')

from tictactoe import minimax
import tictactoe as ttt
import time

COL_NAMES = 'ABC'


def print_board(board):
    print('   A   B   C')
    for row in (1, 2, 3):
        pieces = []
        for col in (1, 2, 3):
            piece = board.get(ttt.Point(row, col))
            if piece == ttt.Player.x:
                pieces.append('X')
            elif piece == ttt.Player.o:
                pieces.append('O')
            else:
                pieces.append(' ')
        print('%d  %s' % (row, ' | '.join(pieces)))


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])
    return ttt.Point(row, COL_NAMES.index(col_name) + 1)

def print_move(player, move):
    move_str = '%s%d' % (COL_NAMES[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))

def main():
    game = ttt.GameState.new_game()

    human_player = ttt.Player.x
    # bot_player = ttt.Player.o

    bot = minimax.MinimaxAgent()
    print_board(game.board)
    
    while not game.is_over():
        if game.next_player == human_player:
            human_move = input('输入[A1-C3]:')
            point = point_from_coords(human_move.strip())
            move = ttt.Move(point)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move) # 应用
        
        print(chr(27) + "[2J")  # <2> clean the screen
        print_board(game.board)
        print_move(game.next_player.other, move)
        time.sleep(0.1)
        
    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print('Winner: ' + str(winner))


if __name__ == '__main__':
    main()
