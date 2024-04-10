"""
q learning training cycle

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

from dlgo import agent
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.encoders import get_encoder_by_name
from dlgo.networks import qnet_small


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

    # nametuple todo moves作用？margin作用？
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
    temperature = args.temperature
    device = args.device
    model = args.model
    encoder_name = args.encoder_name

    agent1, agent2 = None, None
    if not os.path.exists(agent_filename):
        encoder = get_encoder_by_name(name=encoder_name, board_size=board_size)
        agent1 = rl.load_q_agent(model=model, encoder=encoder, device=device)
        agent2 = rl.load_q_agent(model=model, encoder=encoder, device=device)
    else: 
        agent1 = rl.load_q_agent(model=model, save_path=agent_filename, device=device)
        agent2 = rl.load_q_agent(model=model, save_path=agent_filename, device=device)
    assert (agent1 is not None) and (agent2 is not None)
    
    agent1.serialize(agent_filename)

    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)
    # 增加模型随机性
    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)
    
    #----------------------------------------------------------------
    t1 = time.time()
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
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
    experience = rl.combine_experience([collector1, collector2]) # 整合所有训练数据
    experience.serialize(experience_filename) # 序列化存储
    print("collect {} samples".format(len(experience)))


############################################################################
#########################  Part2: train q value ################################
def train_q(args):
    # init 
    model = args.model
    device = args.device
    # 训练参数
    learning_rate = args.lr
    batch_size = args.batch_size
    # checkpoint
    learning_agent_filename = args.learning_agent
    updated_agent_filename = args.updated_agent
    
    experience_files = list_experience_files(args.experience_dir) 
    learning_agent = rl.load_q_agent(model=model, save_path=learning_agent_filename, device=device)
    
    # 读取数据训练
    for exp_filename in experience_files:
        exp_buffer = rl.load_experience(exp_filename)
        
        learning_agent.train(
            exp_buffer,
            lr=learning_rate,
            batch_size=batch_size)

    # 模型再保存
    learning_agent.serialize(updated_agent_filename)

###########################################################################
#########################  Part3: eval q  ################################
def eval_pg_bot(args):
    device = args.device
    model = args.model
    num_eval_games = args.num_eval_games
    
    new_agent_checkpoint = args.updated_agent
    old_agent_checkpoint = args.learning_agent
    new_agent = rl.load_q_agent(model=model, save_path=new_agent_checkpoint, device=device)
    old_agent = rl.load_q_agent(model=model, save_path=old_agent_checkpoint, device=device)

    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(num_eval_games):
        print('Simulating game %d/%d...' % (i + 1, num_eval_games))
        if color1 == Player.black:
            black_player, white_player = new_agent, old_agent
        else:
            white_player, black_player = new_agent, old_agent
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
        data_home_path = HOME_PATH+'data/q/' # 数据 模型存放目录
        board_size = 9 # 缩小计算量, 保证算法的验证速度
        num_sim_games = 100 # 每轮模拟的对局数
        num_sim_tasks = 10 # 模拟共执行N次
        num_eval_games = 50 # 评估模型对局数
        
        # 序列化文件
        learning_agent = data_home_path + 'agent_checkpoint.pth'
        updated_agent = data_home_path+ 'agent_checkpoint_update.pth'
        experience_dir = data_home_path + 'experience/'
        
        # 配套模型&encoder
        encoder_name = 'sevenplane'
        model = qnet_small(input_channel_num=7,
                board_size=board_size,
                embedding_dim=int(board_size*board_size/2)) # todo 每次都是新模型？？
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型训练参数
        lr = 0.01
        batch_size = 512
        
        run_p1 = 1
        run_p2 = 1
        run_p3 = 0
        
        # agent
        temperature = 0.1 # 增加agent move的随机性
        
        # log相关参数
        # log_show_borad = False
    
    for dir_path in [args.data_home_path, args.experience_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print('--'*10 + '\nPart0: 数据准备')
    print("\tdevice:", args.device)
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
            for _ in range(args.num_sim_tasks):
                self_play(args=args)
        
        #-------------Part2 train agent -----------------------
        if args.run_p2:
            print('--'*10 + '\nPart2: 训练agent')
            train_q(args=args)
        
        #-------------Part3 eval q bot -----------------------
        if args.run_p3:
            print('--'*10 + '\nPart3: 模型评估')
            wins, losses = eval_pg_bot(args=args)
            confidence = binomtest(wins, wins+losses, 0.5)
            print("同水平可信度", confidence)
        
            # 高可信度获胜
            if wins > losses and confidence.pvalue < 0.06:
                print("高优agent可信度: {:.3f}".format(1-confidence.pvalue))
        
        #-------------Part4 更新进度 -----------------------
        print('--'*10 + '\nPart4: 模型、数据 update')

        # 删掉learning模型，update模型重命名
        history_name = args.learning_agent + str(datetime.now())
        checkpoint_rename(args.learning_agent, history_name)
        checkpoint_rename(args.updated_agent, args.learning_agent)
        
        # todo 数据部分是保留呢，还是删掉使用新的数据。
        '''
            # 每一轮删除experience下的所有文件
                理由: 我们希望模型在众多数据的更新中，平衡掉一些错误的训练数据
                    如果每次叠加上一轮的训练数据进行模型训练，
                    前一轮的数据比后一轮的数据见的多，
                    模型跟倾向于对旧数据的拟合。

            缺点：生成数据太耗时，训练模型却非常快。
                TODO 如何通过简单的方式进行架构优化。
                已知 pytorch gpu model没法在ptyhon的多个线程上运行。
                
            TODO 暂时不确定模型在训练过程中是不是一致性的变好, lr和随机性探索的调节也需要实验验证。
        '''


if __name__ == '__main__':
    main_loop()


