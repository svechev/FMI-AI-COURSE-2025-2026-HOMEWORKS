from os import getenv
from timeit import default_timer


def frog_leap_dfs(frogs, blank):
    res.append("".join(frogs))
    if res[-1] == goal_state:
        return True

    if blank + 2 <= 2 * n and frogs[blank + 2] == '<':  # trying jumping to the left 2 spots
        frogs[blank + 2], frogs[blank] = frogs[blank], frogs[blank + 2]
        if frog_leap_dfs(frogs, blank + 2):
            return True
        frogs[blank + 2], frogs[blank] = frogs[blank], frogs[blank + 2]

    if blank - 2 >= 0 and frogs[blank - 2] == '>':  # trying jumping to the right 2 spots
        frogs[blank - 2], frogs[blank] = frogs[blank], frogs[blank - 2]
        if frog_leap_dfs(frogs, blank - 2):
            return True
        frogs[blank - 2], frogs[blank] = frogs[blank], frogs[blank - 2]

    if blank-1 >= 0 and frogs[blank - 1] == '>':     # trying jumping to the right 1 spot
        frogs[blank - 1], frogs[blank] = frogs[blank], frogs[blank - 1]
        if frog_leap_dfs(frogs, blank-1):
            return True
        frogs[blank - 1], frogs[blank] = frogs[blank], frogs[blank - 1]

    if blank+1 <= 2*n and frogs[blank + 1] == '<':    # trying jumping to the left 1 spot
        frogs[blank + 1], frogs[blank] = frogs[blank], frogs[blank + 1]
        if frog_leap_dfs(frogs, blank + 1):
            return True
        frogs[blank + 1], frogs[blank] = frogs[blank], frogs[blank + 1]



    res.pop()
    return False


def frog_leap(n):
    frogs = [*['>' for _ in range(n)], '_', *['<' for _ in range(n)]]

    frog_leap_dfs(frogs, n)
    return res


n = int(input())

start = default_timer()
goal_state = '<'*n + '_' + '>'*n
res = []
frog_leap(n)
end = default_timer()
if getenv("FMI_TIME_ONLY") == "1":
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
else:
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
    for x in res:
        print(x)
