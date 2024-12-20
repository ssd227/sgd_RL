"""
Alpha Go Zeros training cycle

TODO layz Linear 被冲刷一遍数据后就不能初始化了
    本来想省事的结果更费劲了
"""

HOME_PATH = '/playground/sgd_deep_learning/sgd_rl/go/'
import sys
sys.path.append(HOME_PATH+'python')

import glob
import os
import time
import datetime
import hashlib
import uuid
import torch
from datetime import datetime
from collections import namedtuple
from scipy.stats import binomtest

from dlgo import scoring, zero
from dlgo import GameState, Player, Point


#############################################################################
#########################  helper functions  ################################
# 生成随机的文件名
def random_hash():
    # 生成随机字符串作为文件名
    random_filename = str(uuid.uuid4())
    # 使用 hashlib 计算文件名的哈希值
    hash_object = hashlib.md5(random_filename.encode())
    hash_value = hash_object.hexdigest()
    return hash_value

def list_experience_files(data_dir):
    files = []
    base = data_dir + '*.pth'
    for experience_file in glob.glob(base):
        files.append(experience_file)                    
    return files

def checkpoint_rename(current_name, new_name):
    try:
        os.rename(current_name, new_name)
        print(f"{current_name} 已成功重命名为 {new_name}")
    except FileNotFoundError:
        print(f"{current_name} 不存在，无法重命名")
    except FileExistsError:
        print(f"{new_name} 已经存在，请使用不同的新名称")

#-----------------------------------------------
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}

def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def name(player):
    if player == Player.black:
        return 'B'
    return 'W'

def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

#############################################################################
#########################  Part1: self paly  ################################
def self_play(args):
    agent_filename = args.learning_agent
    experience_dir = args.experience_dir
    experience_filename = experience_dir + random_hash() +".pth"
    num_games = args.num_sim_games
    board_size = args.board_size
    
    device = args.device
    model = args.model
    encoder = args.encoder
    rounds_per_move = args.rounds_per_move
    c = args.c

    agent1, agent2 = None, None
    if not os.path.exists(agent_filename):
        agent1 = zero.ZeroAgent(model, encoder, rounds_per_move, c, device=device) 
        agent2 = zero.ZeroAgent(model, encoder, rounds_per_move, c, device=device)
    else:
        agent1 = zero.load_agent(model, encoder, agent_filename, rounds_per_move, c, device=device)
        agent2 = zero.load_agent(model, encoder, agent_filename, rounds_per_move, c, device=device)
    assert (agent1 is not None) and (agent2 is not None)
    
    agent1.serialize(agent_filename)

    collector1 = zero.ZeroExperienceCollector()
    collector2 = zero.ZeroExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)
    
    #----------------------------------------------------------------
    t1 = time.time()
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode() # 开始记录， 不是agent负责
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.black:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
    print("simulatinon of {} games , cost_time:{:.3f}s.".format(num_games, time.time()-t1))

    # save to disk
    experience = zero.combine_experience([collector1, collector2]) # 整合所有训练数据
    experience.serialize(experience_filename) # 序列化存储
    print("collect {} samples".format(len(experience)))


############################################################################
#########################  Part2: train agz  ################################
def train_agz(args):
    # init
    encoder = args.encoder
    model = zero.agznet(encoder.num_planes(), moves_num=encoder.num_moves())
    
    device = args.device
    rounds_per_move = args.rounds_per_move
    c = args.c
    learning_rate = args.lr
    batch_size = args.batch_size
    
    # checkpoint
    learning_agent_filename = args.learning_agent
    updated_agent_filename = args.updated_agent
    
    experience_files = list_experience_files(args.experience_dir)
    print(learning_agent_filename)
    learning_agent = zero.load_agent(model, encoder, learning_agent_filename, rounds_per_move, c, device=device)
    print("part load model sucess!")

    # 读取数据训练
    for exp_filename in experience_files:
        exp_buffer = zero.load_experience(exp_filename)

        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)

    # 模型再保存
    learning_agent.serialize(updated_agent_filename)

###########################################################################
#########################  Part3: eval agz  ################################
def eval_agz_bot(args):
    device = args.device
    encoder = args.encoder
    model = zero.agznet(encoder.num_planes(), moves_num=encoder.num_moves())
    
    num_eval_games = args.num_eval_games
    rounds_per_move = args.rounds_per_move
    c = args.c

    agent1_checkpoint = args.updated_agent # new
    agent2_checkpoint = args.learning_agent # old

    agent1 = zero.load_agent(model, encoder,agent1_checkpoint, rounds_per_move, c, device=device)  # new
    
    model = zero.agznet(encoder.num_planes(), moves_num=encoder.num_moves())
    agent2 = zero.load_agent(model, encoder, agent2_checkpoint, rounds_per_move, c, device=device) # old
   

    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(num_eval_games):
        print('Simulating game %d/%d...' % (i + 1, num_eval_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))
    
    return wins, losses

###########################################################################
#########################  main loop  ################################
def main_loop():
    # 参数变量
    class args:
        data_home_path = HOME_PATH+'data/agz/' # 数据 模型存放目录
        board_size = 9 # 缩小计算量, 保证算法的验证速度
        num_sim_games = 1 # 每轮模拟数据新增数据量
        num_sim_tasks = 1 # 模拟x轮
        num_eval_games = 2 # 评估模型对局数，可以动态调整
        
        rounds_per_move = 20 # agz agent 每步探索次数
        c = 2.0 # UCT 协调系数（超参数）

        # 序列化文件
        learning_agent = data_home_path + 'agent_checkpoint.pth'
        updated_agent = data_home_path+ 'agent_checkpoint_update.pth'
        experience_dir = data_home_path + 'experience/'
        
        # 配套模型&encoder
        encoder = zero.ZeroEncoder(board_size)  # 注意这里的转变
        model = zero.agznet(input_channel_num=encoder.num_planes(), 
                                moves_num=encoder.num_moves())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型训练参数
        lr = 0.001
        batch_size = 1024
        
        run_p1 = 1
        run_p2 = 1
        run_p3 = 1
        
        # log相关参数
        # log_show_borad = False
    
    for dir_path in [args.data_home_path, args.experience_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print('--'*10 + '\nPart0: 数据准备')
    # print("\tdevice:", args.device)
    print('\tagent checkpoint文件存在:', os.path.exists(args.learning_agent))
    print('\t训练数据文件(experience)存在:', os.path.exists(args.experience_dir))

    # 全局变量
    global BOARD_SIZE
    BOARD_SIZE = args.board_size

    '''
        Training-Cycle
        
        依次执行: 对局数据生成 ==> model.train ==> 模型评估
        需要人工介入时，暂停程序并给出不确定日志
    '''
    while True:
        #-------------Part1 self play -----------------------
        if args.run_p1:
            print('--'*10 + '\nPart1: 对局数据收集')
            for i in range(args.num_sim_tasks):
                self_play(args=args) # 单线程版本
        
        #-------------Part2 train agent -----------------------
        if args.run_p2:
            print('--'*10 + '\nPart2: 训练agent')
            train_agz(args=args)
        
        #-------------Part3 eval pg bot -----------------------
        if args.run_p3:
            print('--'*10 + '\nPart3: 模型评估')
            wins, losses = eval_agz_bot(args=args)
        
        #-------------Part4 更新进度 -----------------------
        confidence = binomtest(wins, wins+losses, 0.5)
        print("同水平可信度",confidence)
        
        # 高可信度获胜 
        if wins > losses  and confidence.pvalue < 0.06:
            print("高优agent可信度: {:.3f}".format(1-confidence.pvalue))
            # 删掉learning模型，update模型重命名
            current_time = datetime.now()
            history_name = args.learning_agent+current_time
            checkpoint_rename(args.learning_agent, history_name)
            checkpoint_rename(args.updated_agent, args.learning_agent)
            continue
        else:
            break


if __name__ == '__main__':
    main_loop()


