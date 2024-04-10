import random

from dlgo.agent import Agent
from dlgo.gotypes import Player

__all__ = [
    'AlphaBetaAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999


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
        opponent_best_result = alpha_beta_result(next_state, max_depth - 1,
                                                best_black, best_white, eval_fn)
        our_result = -1 * opponent_best_result

        if our_result > best_so_far: # 当前搜索状态下的最好值 best_so_far
            best_so_far = our_result

        if game_state.next_player == Player.white: # 白子在搜索状态
            if best_so_far > best_white:
                best_white = best_so_far # 更新最佳的白子值
            outcome_for_black = -1 * best_so_far
            if outcome_for_black < best_black: # 黑子值已比上一轮黑子最佳差了，停止搜索（继续搜索->更好的白色值->更差的黑色值）
                return best_so_far

        elif game_state.next_player == Player.black: # 黑子在搜索状态
            if best_so_far > best_black:
                best_black = best_so_far # 更新最佳的黑子值
            outcome_for_white = -1 * best_so_far
            if outcome_for_white < best_white: # 白子值已比上一轮白子最佳差了，停止搜索（继续搜索->更好的黑色值->更差的白色值）
                return best_so_far

    return best_so_far


class AlphaBetaAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        best_black = MIN_SCORE
        best_white = MIN_SCORE
       
        # Loop over all legal moves.
        for possible_move in game_state.legal_moves():
            
            next_state = game_state.apply_move(possible_move)
            opponent_best_outcome = alpha_beta_result(
                                        next_state, self.max_depth,
                                        best_black, best_white, # 初始状态-MIN_SCORE
                                        self.eval_fn)
            # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # This is the best move so far.
                best_moves = [possible_move]
                best_score = our_best_outcome
                # 同步更新best next_player
                if game_state.next_player == Player.black:
                    best_black = best_score
                elif game_state.next_player == Player.white:
                    best_white = best_score
                    
            elif our_best_outcome == best_score:
                # This is as good as our previous best move.
                best_moves.append(possible_move)
        # For variety, randomly select among all equally good moves.
        return random.choice(best_moves)
