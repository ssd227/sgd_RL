from __future__ import absolute_import

import os.path
import tarfile
import gzip
import glob
import shutil
import numpy as np

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler


'''
# 单线程sgf数据预处理

'''
class GoDataProcessor:
    def __init__(self, encoder='oneplane', data_directory='data'):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory


    def load_go_data(self, data_type='train',  # 'train' or 'test'.
                     num_samples=1000):  # 加载多少盘对局
        '''
        默认直接开始下载文件
            We download all games from KGS to our local data directory. 
            If data is available, it won't be downloaded again.
        '''
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()

        # 在全量tart数据中直接采样各sgf文件，'train'和'test'集合对应不同日期的zip文件
        sampler = Sampler(data_dir=self.data_dir)
        data = sampler.draw_data(data_type, num_samples) # List[(tar_file_name, game_index_of_subfile_list)]

        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in data:
            zip_names.add(filename)
            
            # map {zip_name : index}  index指代tar文件中单个sgf所在的file_idx
            indices_by_zip_name.setdefault(filename, [])
            indices_by_zip_name[filename].append(index) 

        # 然后分别处理单个zip文件，把采样到的sfg文件进一步处理成训练数据（通过index找到指定的sfg文件）
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name): # 还没生成数据文件 todo 这个文件判定的意义
                # The zip files are then processed individually.
                self.process_zip(zip_name, data_file_name, indices_by_zip_name[zip_name])
        
        # # Features and labels from each zip are then aggregated and returned.
        # features_and_labels = self.consolidate_games(data_type, data)
        # return features_and_labels

    # Unpack the `gz` file into a `tar` file.
    def unzip_data(self, zip_file_name):
        # 读取gz文件
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)
        # 删掉".gz" 保留.tar文件名
        tar_file = zip_file_name[0:-3]
        # gz文件写入新建的tar文件
        with open(self.data_dir + '/' + tar_file, 'wb') as this_tar:
            # Copy the contents of the unpacked file into the `tar` file.
            shutil.copyfileobj(this_gz, this_tar)  
        return tar_file

    # 处理每一个单独的zip file
    def process_zip(self, zip_file_name, data_file_name, game_list):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames() # 压缩文件的name list
        
        # 当前zip文件中被采样到的所有sfg（单局）文件所包含的所有样本数据（game_state, move）
        total_examples = self.num_total_examples(zip_file, game_list, name_list)
        # 训练集单个样本x的维度
        shape = self.encoder.shape()
        
        # numpy格式的(features, labels)数据
        feature_shape = np.insert(shape, 0, np.asarray([total_examples])) #[Batch, channel, X, Y] todo 转为pytorch高效处理的顺序
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,)) # np一维
        print("features shape:{}, labels shape:{}.".format(features.shape, labels.shape))

        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            
            # Read the SGF content as string, after extracting the zip file.
            sgf_content = zip_file.extractfile(name).read() # .sgf文件-> 字符串
            sgf = Sgf_game.from_string(sgf_content)
            
            # Infer the initial game state by applying all handicap stones.
            game_state, first_move_done = self.get_handicap(sgf) # 初始化游戏状态, 并设置让子

            # Iterate over all moves in the SGF file.
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    # 构造move
                    if move_tuple is not None:  # Read the coordinates of the stone to be played...
                        row, col = move_tuple # 已经解析成坐标(但下标从0开始)
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()  # W[], indicating a pass
                        
                    if first_move_done and point is not None:
                        # encode the current game state as features
                        features[counter] = self.encoder.encode(game_state)
                        # the next move as label for the features.
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    # 执行move, 获得下一步状态(game_state) 
                    game_state = game_state.apply_move(move)
                    first_move_done = True

        #################### 存储特征数据到多个文件，方便后续快速读取batch ####################
        # 文件名模板
        dir_path = self.data_dir + '/' + self.encoder_string
        feature_file_base = dir_path + '/' + data_file_name + '_features_%d'
        label_file_base = dir_path + '/' + data_file_name + '_labels_%d'
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        chunk = 0  # Due to files with large content, split up after chunksize
        chunksize = 1024
        while features.shape[0] >= chunksize:  # 只要样本数还>1024就继续截断保存，最后剩余样本直接丢弃。
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            # 截断已解析数据
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            # 保存到chunk文件
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)


    def consolidate_games(self, data_type, samples): # samples: List[(tar_file_name, game_index_of_subfile_list)]
        # 所有tar文件
        files_needed = set(file_name for file_name, index in samples)

        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)

        feature_list = []
        label_list = []
        
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '') # todo 这句没什么用
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base): # 这个正则匹配的用法还挺方便
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                
                x = x.astype('float32')
                y = y.astype(int)
                
                feature_list.append(x)
                label_list.append(y)

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        print('--consolidate--')
        print("features shape:{}, labels shape:{}.".format(features.shape, labels.shape))
        
        np.save('{}/features_{}.npy'.format(self.data_dir, data_type), features)
        np.save('{}/labels_{}.npy'.format(self.data_dir, data_type), labels)

        return features, labels

    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            # 让子(黑子下了N步)
            for setup in sgf.get_root().get_setup_stones(): # setup是什么东西,为什么有很多
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
                    
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move) # 白子先行, 无上一个状态, last_move=move
            
        return game_state, first_move_done


    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True # 保证棋盘上有棋子, 走下一步 (才算作第一个训练数据)
                total_examples = total_examples + num_moves # 统计N盘棋,一共可以收集到多少步的训练数据pair (game_state : move)
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples