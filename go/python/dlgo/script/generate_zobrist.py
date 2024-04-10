import random
from ..gotypes import Player, Point


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white

MAX63 = 0x7fffffffffffffff

def generate_hash():
    table = {}
    empty_board = 0
    for row in range(1, 20):
        for col in range(1, 20):
            for state in (Player.black, Player.white, None):
                code = random.randint(0, MAX63)
                table[Point(row, col), state] = code
    out_str = []
    out_str.append('from .gotypes import Player, Point')
    out_str.append('')
    out_str.append("__all__ = ['HASH_CODE', 'EMPTY_BOARD']")
    out_str.append('')
    out_str.append('HASH_CODE = {')
    for (pt, state), hash_code in table.items():
        out_str.append('    (%r, %s): %r,' % (pt, to_python(state), hash_code))
    out_str.append('}')
    out_str.append('')
    out_str.append('EMPTY_BOARD = %d' % random.randint(empty_board, MAX63))
    
    return [line+'\n' for line in out_str]
