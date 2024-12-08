'''
Tree search + Pruning

Reducing search depth with position evaluation


'''

import random
from dlgobang.agent import Agent

__all__ = ['DepthPrunedAgent',]

MIN_SCORE = -999999
MAX_SCORE = -MIN_SCORE

def best_result(game_state, max_depth, eval_fn):
    if game_state.is_over(): # 真实游戏获胜，获得最大局面评估
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE                         
        else:            
            return MIN_SCORE

    if max_depth == 0: # 达到最大look ahead深度，停止tree search
        return eval_fn(game_state) # 使用局面评估函数 eval_fn

    best_so_far = MIN_SCORE # init 最差评估分数（对手获胜）
    for candidate_move in game_state.legal_moves(): 
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = best_result(next_state, max_depth-1, eval_fn) # 递归求解对手最优
        our_result = -1 * opponent_best_result
        if our_result > best_so_far:
            best_so_far = our_result
    return best_so_far


class DepthPrunedAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        super().__init__()
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None

        for possible_move in game_state.legal_moves():
            print('DepthPrunedAgent is search Move', possible_move)
            next_state = game_state.apply_move(possible_move)
            opponent_best_outcome = best_result(next_state, self.max_depth, self.eval_fn)
            our_best_outcome = -1 * opponent_best_outcome
            
            if (best_score is None) or (our_best_outcome > best_score):
                best_moves = [possible_move] # reset
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                best_moves.append(possible_move) # 同样好的move
            else:
                pass # 非当前最好move

        return random.choice(best_moves) # 能够保证best_moves有值可以返回