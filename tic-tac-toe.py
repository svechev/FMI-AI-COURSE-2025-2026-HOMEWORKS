from math import inf
from os import getenv
from timeit import default_timer

pretty_row = "+---+---+---+"


def read_input_board():
    _ = input()
    row1 = [symbol for symbol in input().split() if symbol in ['O', 'X', '_']]
    _ = input()
    row2 = [symbol for symbol in input().split() if symbol in ['O', 'X', '_']]
    _ = input()
    row3 = [symbol for symbol in input().split() if symbol in ['O', 'X', '_']]
    _ = input()
    board = [row1, row2, row3]
    depth = 10 - board[0].count('_') - board[1].count('_') - board[2].count('_')
    return board, depth


def print_board(board):
    for row in board:
        print(pretty_row)
        print("| " + row[0] + " | " + row[1] + " | " + row[2] + " |")
    print(pretty_row)


def valid_moves(board):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                moves.append((i, j))
    return moves


def is_terminal(board, depth):
    end_states = [['X', 'X', 'X'], ['O', 'O', 'O']]
    if computer_symbol == 'X':
        end_states.reverse()        # first row will be player's symbols, second row is computer
    for row in board:
        if row in end_states:
            return (10 - depth) if row == end_states[1] else -1 * (10 - depth)
    diag1, diag2 = [], []
    for i in range(3):
        diag1.append(board[i][i])
        diag2.append(board[2-i][i])
        col = [row[i] for row in board]
        if col in end_states:
            return (10 - depth) if col == end_states[1] else -1 * (10 - depth)
    if diag1 in end_states:
        return (10 - depth) if diag1 == end_states[1] else -1 * (10 - depth)
    if diag2 in end_states:
        return (10 - depth) if diag2 == end_states[1] else -1 * (10 - depth)

    # no winning configuration
    if depth > 9:
        return 0  # end; draw
    else:
        return None


def player_move(board):  # with input validation
    is_valid = False
    while not is_valid:
        nums = input().split()
        if len(nums) != 2 or (not nums[0].isnumeric()) or (not nums[1].isnumeric()):
            print("Invalid input")
            continue
        x, y = int(nums[0]) - 1, int(nums[1]) - 1
        if not 0 <= x <= 2 or not 0 <= y <= 2 or board[x][y] != '_':
            print("Invalid move")
        else:
            board[x][y] = symbol
            is_valid = True


def alpha_beta_search(board, depth, symbol):
    v, move = max_value(board, -inf, inf, None, symbol, depth)
    return move


def max_value(board, alpha, beta, last_move, symbol, depth):
    score = is_terminal(board, depth)
    if score != None:
        return score, last_move
    v = -inf
    for (x, y) in valid_moves(board):
        board[x][y] = symbol
        new_symbol = 'X' if symbol == 'O' else 'O'
        new_v, _ = min_value(board, alpha, beta, (x, y), new_symbol, depth + 1)
        if new_v > v:
            v = new_v
            optimal_move = (x, y)
        if v >= beta:
            board[x][y] = "_"
            return v, (x, y)
        alpha = max(alpha, v)
        board[x][y] = "_"
    return v, optimal_move


def min_value(board, alpha, beta, last_move, symbol, depth):
    score = is_terminal(board, depth)
    if score != None:
        return score, last_move
    v = inf
    for (x, y) in valid_moves(board):
        board[x][y] = symbol
        new_symbol = 'X' if symbol == 'O' else 'O'
        new_v, _ = max_value(board, alpha, beta, (x, y), new_symbol, depth + 1)
        if new_v < v:
            v = min(v, new_v)
            optimal_move = (x, y)
        if v <= alpha:
            board[x][y] = "_"
            return v, (x, y)
        beta = min(beta, v)
        board[x][y] = "_"
    return v, optimal_move


def make_move(board, depth, symbol):
    x, y = alpha_beta_search(board, depth, symbol)
    board[x][y] = symbol


mode = input()
if mode == "JUDGE":
    computer_symbol = input()[5]
    board, depth = read_input_board()

    start = default_timer()
    score = is_terminal(board, depth)
    if score is None:
        result = alpha_beta_search(board, depth, computer_symbol)
    end = default_timer()
    if getenv("FMI_TIME_ONLY") == "1":
        print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
    else:
        print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
        if score is None:
            print(result[0] + 1, result[1] + 1)
        else:
            print(-1)


elif mode == "GAME":
    symbol = input()[6]     # who is first
    human = input()[6]
    computer_symbol = 'O' if human == 'X' else 'X'    # who is human
    board, depth = read_input_board()

    score = is_terminal(board, depth)    # check if we were given a terminal board for some reason
    if score is None:
        winner = None
    else:
        winner = computer_symbol if score > 0 else 'X' if computer_symbol == 'O' else 'O'

    while depth < 10 and not winner:
        if symbol == computer_symbol:
            make_move(board, depth, symbol)
        else:
            player_move(board)
        print_board(board)
        symbol = 'O' if symbol == 'X' else 'X'
        score = is_terminal(board, depth)
        if score != None:
            winner = computer_symbol if score > 0 else 'X' if computer_symbol == 'O' else 'O'
        depth += 1

    if winner:
        print(f"WINNER: {winner}")
    else:
        print("DRAW")


'''      blank board
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
| _ | _ | _ |
+---+---+---+
'''