from os import getenv
from timeit import default_timer


def frog_leap(n):
    frogs = [*['>' for _ in range(n)], '_', *['<' for _ in range(n)]]
    res = ["".join(frogs)]


    moves1 = [(i, 'r' if i % 2 else 'l') for i in range(1, n + 1)]
    middle_move = (n, 'l' if n % 2 else 'r')
    moves2 = [(i, 'r' if i % 2 else 'l') for i in range(n, 0, -1)]
    moves = [*moves1, middle_move, *moves2]

    blank = n

    for (i, move) in moves:
        for _ in range(i):
            if move == 'r':
                if frogs[blank - 1] == '>':
                    frogs[blank - 1], frogs[blank] = frogs[blank], frogs[blank - 1]
                    blank -= 1
                else:
                    frogs[blank - 2], frogs[blank] = frogs[blank], frogs[blank - 2]
                    blank -= 2
            else:  # move == 'l'
                if frogs[blank + 1] == '<':
                    frogs[blank + 1], frogs[blank] = frogs[blank], frogs[blank + 1]
                    blank += 1
                else:
                    frogs[blank + 2], frogs[blank] = frogs[blank], frogs[blank + 2]
                    blank += 2
            res.append("".join(frogs))
    return res


n = int(input())
start = default_timer()
res = frog_leap(n)
end = default_timer()
if getenv("FMI_TIME_ONLY") == "1":
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
else:
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
    for x in res:
        print(x)
