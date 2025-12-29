from math import sqrt, inf
from copy import deepcopy
import timeit
import os

possible_moves = {
    "start": ["left", "up", "down", "right"],
    "left": ["left", "up", "down"],
    "right": ["right", "up", "down"],
    "up": ["left", "up", "right"],
    "down": ["left", "right", "down"]
}


def target_col(num):
    if goal == -1 or goal == n:  # 0 must be bottom right
        return (num - 1) % size
    elif goal == 0:  # 0 must be top left
        return num % size
    else:  # 0 must be in the middle
        if num <= n // 2:
            return (num - 1) % size
        else:
            return num % size


def target_row(num):
    if goal == -1 or goal == n:  # 0 must be bottom right
        return (num - 1) // size
    elif goal == 0:  # 0 must be top left
        return num // size
    else:  # 0 must be in the middle
        if num <= n // 2:
            return (num - 1) // size
        else:
            return num // size


def manhattan(i, j):
    elem = board[i][j]
    if goal == -1 or goal == n:  # 0 must be bottom right
        return abs(i - (elem - 1) // size) + abs(j - (elem - 1) % size)
    elif goal == 0:  # 0 must be top left
        return abs(i - (elem // size)) + abs(j - (elem % size))
    else:  # 0 must be in the middle
        if elem <= n // 2:
            return abs(i - (elem - 1) // size) + abs(j - (elem - 1) % size)
        else:
            return abs(i - (elem // size)) + abs(j - (elem % size))


def is_solvable():
    inversions = 0
    elements = [el for sublist in board for el in sublist]
    for (index, num) in enumerate(elements):
        if num > 0:
            for next_num in elements[index + 1:]:
                if num > next_num > 0:
                    inversions += 1
    if size % 2 == 1:
        return inversions % 2 == 0
    else:
        if goal == 0:
            return (inversions + blank[0]) % 2 == 0
        else:
            return (inversions + blank[0]) % 2 == 1


def check_move(move, blank):
    x, y = blank[0], blank[1]
    match move:
        case "left":
            if y == size - 1:
                return None
            num = board[x][y+1]
            return -1 if y+1 > target_col(num) else 1
        case "right":
            if y == 0:
                return None
            num = board[x][y-1]
            return -1 if y-1 < target_col(num) else 1
        case "up":
            if x == size - 1:
                return None
            num = board[x+1][y]
            return -1 if x+1 > target_row(num) else 1
        case "down":
            if x == 0:
                return None
            num = board[x-1][y]
            return -1 if x-1 < target_row(num) else 1
        case _:
            return None


def make_move(move, blank):
    x, y = blank[0], blank[1]
    match move:
        case "left":
            board[x][y], board[x][y+1] = board[x][y+1], board[x][y]
            blank[1] += 1
        case "right":
            board[x][y], board[x][y-1] = board[x][y-1], board[x][y]
            blank[1] -= 1
        case "up":
            board[x][y], board[x+1][y] = board[x+1][y], board[x][y]
            blank[0] += 1
        case "down":
            board[x][y], board[x-1][y] = board[x-1][y], board[x][y]
            blank[0] -= 1
    moves.append(move)


def move_back(move, blank):
    x, y = blank
    match move:
        case "right":
            board[x][y], board[x][y+1] = board[x][y+1], board[x][y]
            blank[1] += 1
        case "left":
            board[x][y], board[x][y-1] = board[x][y-1], board[x][y]
            blank[1] -= 1
        case "down":
            board[x][y], board[x+1][y] = board[x+1][y], board[x][y]
            blank[0] += 1
        case "up":
            board[x][y], board[x-1][y] = board[x-1][y], board[x][y]
            blank[0] -= 1
    moves.pop()


def procedure(manh_dist, blank):
    if not is_solvable():
        return -1
    bound = manh_dist
    visited = []
    while True:
        t = search(visited, 0, bound, manh_dist, blank)
        if t == True:
            return bound
        bound = t


def search(visited, g, bound, manh_dist, blank):
    f = g + manh_dist

    if f > bound:
        return f   # not too long paths
    if manh_dist == 0:
        return True   # we got it
    minimum = inf

    # get successors
    good_moves = []
    bad_moves = []

    # where we can move - good vs bad moves, valid moves, also we don't want to reset the previous move
    next_moves = possible_moves[moves[-1]] if moves else possible_moves["start"]
    for possible_move in next_moves:
        checked_move = check_move(possible_move, blank)
        if checked_move == -1:
            good_moves.append(possible_move)
        elif checked_move == 1:
            bad_moves.append(possible_move)

    for move in good_moves:      # check good moves first
        make_move(move, blank)
        manh_dist -= 1

        if board not in visited:
            visited.append(deepcopy(board))
            t = search(visited, g + 1, bound, manh_dist, blank)
            if t == True:
                return True
            if t < minimum:
                minimum = t
            visited.remove(board)

        move_back(move, blank)
        manh_dist += 1

    for move in bad_moves:    # check bad moves after that
        make_move(move, blank)
        manh_dist += 1

        if board not in visited:
            visited.append(deepcopy(board))
            t = search(visited, g + 1, bound, manh_dist, blank)
            if t == True:
                return True
            if t < minimum:
                minimum = t
            visited.remove(board)

        move_back(move, blank)
        manh_dist -= 1

    return minimum


n = int(input())
size = round(sqrt(n+1))
goal = int(input())
board = []
for _ in range(round(sqrt(n + 1))):
    row = [int(x) for x in input().split()]
    board.append(row)

start = timeit.default_timer()     # start timer here ------------
moves = []
blank = []
manh_dist = 0
for i in range(size):
    for j in range(size):
        if board[i][j] == 0:
            blank = [i, j]
        else:
            elem = board[i][j]
            if elem > 0:
                manh_dist += manhattan(i, j)

bound_res = procedure(manh_dist, blank)
end = timeit.default_timer()       # end timer here -------------
if os.getenv("FMI_TIME_ONLY") == "1":
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
else:
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
    print(bound_res)
    for move in moves:
        print(move)
