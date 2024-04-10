import enum
import random

from ..agent import Agent

__all__ = [
    'MinimaxAgent',
]


class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3


def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw


def best_result(game_state):
    # 递归的结束
    if game_state.is_over():
        # Game is already over.
        if game_state.winner() == game_state.next_player:
            # We win!
            return GameResult.win
        elif game_state.winner() is None:
            # A draw.
            return GameResult.draw
        else:
            # Opponent won.
            return GameResult.loss

    # 当前局面没结束，遍历所有legal moves，找best result (递归了所有可能性，复杂度是N！)
    best_result_so_far = GameResult.loss
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)     # try legal move
        opponent_best_result = best_result(next_state)         # 对手最优解
        our_result = reverse_game_result(opponent_best_result) # Whatever our opponent wants, we want the opposite.
        if our_result.value > best_result_so_far.value:        # 所有legal move中，找出收益最大的结果
            best_result_so_far = our_result
    return best_result_so_far


class MinimaxAgent(Agent):
    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        
        # Loop over all legal moves.
        for possible_move in game_state.legal_moves():
            # 评估 候选move
            next_state = game_state.apply_move(possible_move)
            opponent_best_outcome = best_result(next_state)
            our_best_outcome = reverse_game_result(opponent_best_outcome)
            
            # append result
            if our_best_outcome == GameResult.win:
                winning_moves.append(possible_move)
            elif our_best_outcome == GameResult.draw:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
        
        # 从优到劣，随机select_move
        if winning_moves:
            return random.choice(winning_moves)
        if draw_moves:
            return random.choice(draw_moves)
        return random.choice(losing_moves)