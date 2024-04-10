from dlgo.gotypes import Point

# 眼的判定
def is_point_an_eye(board, point, color):
    if board.get(point) is not None:
        return False

    # All adjacent points must contain friendly stones.
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor): # 存在棋盘上的邻接位置（上下左右）
            neighbor_color = board.get(neighbor)
            if neighbor_color != color: #邻接的棋子必须全部是同色（必须有子）
                return False

    # We must control 
        # three out of four corners if the point is in the middle of the board;
        # on the edge, you must control all corners.        
    friendly_corners = 0
    off_board_corners = 0
    
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    
    # 确认四个角的棋子颜色
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4 # Point is on the edge or corner.
    return friendly_corners >= 3 # Point is in the middle.