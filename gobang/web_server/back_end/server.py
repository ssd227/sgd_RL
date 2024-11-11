'''

勉强能用

todo 每个client 维护一个单独的game，隔离游戏房间

'''

import os
import sys
os.chdir('../../')  # cd到主目录
sys.path.append('./python')


from flask import Flask, jsonify, request
from flask_cors import CORS

from dlgobang.game import GameState, Player, Move, Point
from dlgobang.agent import RandomBot, MCTSAgent


app = Flask(__name__)
CORS(app)  # 允许跨域访问

game:GameState = None
# whitebot = RandomBot()
whitebot = MCTSAgent(5000, temperature=1.4)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/newgame/<boardSize>', methods=['GET'])
def new_game(boardSize):
    global game
    
    # 新游戏
    game = GameState.new_game(int(boardSize))
    
    game_state = {
        "board": game.get_board_array(),  # NxN的二维棋盘, 0表示空位, 1表示黑，2表示白
        "current_turn": 1 if game.next_player== Player.BLACK else 2, # 1表示玩家Black, 2表示玩家White
        "is_over": game.is_over(),
        "winner": None,
        }
    
    return jsonify(game_state)

# # 获取当前游戏状态
# @app.route('/get_game_state', methods=['GET'])
# def get_game_state():
#     return jsonify(game_state)

# 提交玩家的落子请求
@app.route('/make_move', methods=['POST'])
def make_move():
    global game
    global whitebot
    
    data = request.json
    x = data['x']
    y = data['y']
    # player = data['player']

    player_move = Move(Point(x+1,y+1))
    if game.is_valid_move(player_move):
        # 暂默认玩家为黑色先行棋子
        game = game.apply_move(player_move)
        
        # 白棋turn
        bot_move = whitebot.select_move(game)
        game = game.apply_move(bot_move)
        
        # 整理回执信息
        winner = None
        if game.winner():
            winner = 1 if game.winner()== Player.BLACK else 2

        game_state = {
            "board": game.get_board_array(),  # NxN的二维棋盘, 0表示空位, 1表示黑，2表示白
            "current_turn": 1 if game.next_player== Player.BLACK else 2, # 1表示玩家Black, 2表示玩家White
            "is_over": game.is_over() ,
            "winner": winner,
        }
    
        return jsonify({"status": "success", "game_state": game_state})
    else:
        msg = "Invalid move"
        if game.is_over():
            winner_str = 'Black' if game.winner()== Player.BLACK else 'White'
            msg = 'game is over, winner is {}'.format(winner_str)
        return jsonify({"status": "fail", "message": msg}), 400

if __name__ == '__main__':
    app.run()
