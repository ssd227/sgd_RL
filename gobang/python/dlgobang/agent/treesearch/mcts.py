'''
Tree search
Evaluating game states with Monte Carlo tree search

'''
from tqdm import tqdm
import math
import random
from dlgobang.agent import Agent, RandomBot
from dlgobang import Player, Move, GameState
from dlgobang.utils import coords_from_point

__all__ = ['MCTSAgent',]

def fmt(x):
    if x is Player.BLACK:
        return 'B'
    if x is Player.WHITE:
        return 'W'
    if x.is_pass:
        return 'pass'
    if x.is_resign:
        return 'resign'
    return coords_from_point(x.point)


def show_tree(node, indent='', max_depth=3):
    if max_depth < 0:
        return
    if node is None:
        return
    if node.parent is None:
        print('%sroot' % indent)
    else:
        player = node.parent.game_state.next_player
        move = node.move
        print('%s%s %s %d %.3f' % (
            indent, fmt(player), fmt(move),
            node.num_rollouts,
            node.winning_frac(player),
        ))
    for child in sorted(node.children, key=lambda n: n.num_rollouts, reverse=True):
        show_tree(child, indent + '  ', max_depth - 1)


class MCTSNode(object):
    def __init__(self, game_state:GameState, parent=None, move:Move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move

        self.unvisited_moves = game_state.legal_moves()
        self.children = []

        # Statistics about the rollouts that started from this node.
        self.num_rollouts = 0
        self.win_counts = {Player.BLACK: 0, Player.WHITE: 0, 'Tie': 0}

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner:Player):
        if winner:
            self.win_counts[winner] += 1
        else:
            self.win_counts['Tie'] += 1 # winner此时为None，棋盘下满，平局
        self.num_rollouts += 1

    # Helper methods to access useful MCTS tree properties
    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        super().__init__()
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        # loop for a fixed number of rounds for each turn
        # to repeatedly generate rollouts

        for _ in tqdm(range(self.num_rounds)):
            node = root
            
            # 优先展开高层node的legal move，没结束&不能add child时，按照UCT的方式选择子node，继续tree search 
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Add a new child node into the tree. 
            # 注意：if后，允许game_state is over的情况。相当于这条game trace的结果重复了一遍
            if node.can_add_child():
                node = node.add_random_child()

            # Simulate a random game from this node.
            winner = self.simulate_random_game(node.game_state)

            # Propagate scores back up the tree.
            while node is not None:
                node.record_win(winner)
                node = node.parent

        # Having performed as many MCTS rounds as we have time for, we
        # now pick a move.
        scored_moves = [ (child.winning_frac(game_state.next_player), child.move, child.num_rollouts)
                            for child in root.children] # list of tuple (win_rate, move, num_rollouts)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n)) # log top10高胜率候选move

        assert len(scored_moves) > 0 , 'no legal move in MCTS.select_move'
        best_win_rate = scored_moves[0][0]
        best_move = scored_moves[0][1] 
        print('Select move %s with win pct %.3f' % (best_move, best_win_rate))
        
        return best_move # 不存在返回None:Move的情况

    def select_child(self, node):
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        best_score = -1
        best_child = None

        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(node.game_state.next_player) # exploit
            exploration_factor = math.sqrt(math.log(total_rollouts) / child.num_rollouts) # explore
            uct_score = win_percentage + self.temperature * exploration_factor # UCT formula
            
            # find child with max utc_score
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.BLACK: RandomBot(),
            Player.WHITE: RandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()