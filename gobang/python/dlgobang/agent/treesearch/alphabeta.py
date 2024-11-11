'''
Tree search + Pruning
Reducing search width with alpha-beta pruning

alpha beta对应当前最好的best_black 和 best_white

比如当前搜索黑子的move1，如果当前 best black = b
当继续搜索黑子move2的时候，只要出现-w比b的结果更差，就可以停止该分支的tree search
    因为，继续递归白子追求更大的w1，那么-w1比当前的-w要更差

直观解释:已发现比目前局面更劣势局面，提前停止对手的tree search
'''


import random
from dlgobang.agent import Agent
from dlgobang.game import Player

__all__ = ['AlphaBetaAgent',]

MIN_SCORE = -999999
MAX_SCORE = -MIN_SCORE


def alpha_beta_result(game_state, max_depth, best_black, best_white, eval_fn):
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE

    if max_depth == 0:
        return eval_fn(game_state)

    best_so_far = MIN_SCORE
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best = alpha_beta_result(next_state, max_depth-1, best_black, best_white, eval_fn) # alphs/beta值向下递归传递
        our_result = -1 * opponent_best
        best_so_far = max(best_so_far, our_result) # 当前loop, 已探索move对应的最优棋局评分

        ########################
        ## 使用alpha beta值，决定是否提前停止tree search
        ## 注意，如果上一轮是黑子，那么函数传入的黑值用来提前停止当前白子的搜索
        #       白值在本轮更新，用于下一轮黑子tree search 提前停止的标定。反之亦然
        #  疑问，函数传入的历史白值（上上轮白子遍历出的结果）有什么作用
        #    答: 每一次新的tree子节点遍历，都存在一个当前最优tree_path的标定值，这个值就是历史白值
        
        if game_state.next_player == Player.WHITE: # 白子搜索
            best_white = max(best_white, best_so_far)  # 更新最佳白值
            if (-best_so_far) < best_black: # 比上轮黑子最佳差，停止搜索（继续搜索->更好的白值->更差的黑值）
                return best_so_far
            
        elif game_state.next_player == Player.BLACK: # 黑子搜索
            best_black = max(best_black, best_so_far) # 更新最佳黑值
            if (-best_so_far) < best_white: # 比上轮白子最佳差，停止搜索（继续搜索->更好的黑值->更差的白值）
                return best_so_far

    return best_so_far


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        super().__init__()
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        best_black = MIN_SCORE
        best_white = MIN_SCORE
       
        # Loop over all legal moves.
        for possible_move in game_state.legal_moves():
            print('AlphaBetaAgent is search Move', possible_move)
            next_state = game_state.apply_move(possible_move)
            opponent_best = alpha_beta_result(
                                        next_state, self.max_depth,
                                        best_black, best_white, # 初始状态-MIN_SCORE
                                        self.eval_fn)
            
            # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best
            if (not best_moves) or our_best_outcome > best_score:
                # This is the best move so far.
                best_moves = [possible_move]
                best_score = our_best_outcome
                
                # 同步更新alpha beta 值
                if game_state.next_player == Player.BLACK:
                    best_black = best_score
                elif game_state.next_player == Player.WHITE:
                    best_white = best_score
                    
            elif our_best_outcome == best_score:
                best_moves.append(possible_move) # as good as our previous
                
        # For variety, randomly select among all equally good moves.
        return random.choice(best_moves)
