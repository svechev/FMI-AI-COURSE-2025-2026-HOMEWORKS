import timeit
import os
import random


POPULATION_SIZE = 7000
curr_gen_number = 0
ITERATIONS = 20
print_on_step = ITERATIONS // 10
numbers_to_print = []


MUTATE = 0.2
crossover = "partially-mapped"
# crossover = "one-point"
# crossover = "two-point"

#selection = "ranking"
selection = "tournament"


def distance(p1, p2):
    return (abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2) ** 0.5


def make_children(parent1, parent2, crossover_method=crossover):
    if crossover_method == "one-point":
        swap_index = random.randint(1, n // 2 - 1)
        child1, child2 = [], []
        for j in range(swap_index):
            child1.append(parent1.cities[j])
            child2.append(parent2.cities[j])
        for k in range(n):
            if parent2.cities[k] not in child1:
                child1.append(parent2.cities[k])
            if parent1.cities[k] not in child2:
                child2.append(parent1.cities[k])
    elif crossover_method == "two-point":
        children = [[None for _ in range(n)], [None for _ in range(n)]]
        for j in range(n // 3, 2 * n // 3):
            children[0][j] = parent1.cities[j]
            children[1][j] = parent2.cities[j]
        i1, i2 = 0, 0
        for k in range(n):
            if parent2.cities[k] not in children[0]:
                children[0][i1] = parent2.cities[k]
                i1 += 1
                if i1 == n // 3:
                    i1 = 2 * n // 3
            if parent1.cities[k] not in children[1]:
                children[1][i2] = parent1.cities[k]
                i2 += 1
                if i2 == n // 3:
                    i2 = 2 * n // 3
        child1, child2 = children[0], children[1]

    elif crossover_method == "partially-mapped":
        children = [[None for _ in range(n)], [None for _ in range(n)]]
        for j in range(n // 4):
            children[0][j] = parent1.cities[j]
            children[1][j] = parent2.cities[j]
        in_child2_not_child1 = [city for city in children[1] if city not in children[0]]
        in_child1_not_child2 = [city for city in children[0] if city not in children[1]]

        for diff_city in in_child2_not_child1:  # map for top child
            city_index = parent2.cities.index(diff_city)
            top_city = parent1.cities[city_index]
            while city_index < n // 4:
                city_index = parent2.cities.index(top_city)
                top_city = parent1.cities[city_index]
            children[0][city_index] = diff_city

        for diff_city in in_child1_not_child2:    # map for bottom child
            city_index = parent1.cities.index(diff_city)
            bottom_city = parent2.cities[city_index]
            while city_index < n // 4:
                city_index = parent1.cities.index(bottom_city)
                bottom_city = parent2.cities[city_index]
            children[1][city_index] = diff_city

        for i in range(n // 4, n):      # place the rest
            if children[0][i] is None:
                children[0][i] = parent2.cities[i]
            if children[1][i] is None:
                children[1][i] = parent1.cities[i]

        child1, child2 = children[0], children[1]

    return Route(child1), Route(child2)


class City:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name

    def __repr__(self):
        if self.name:
            return f"{self.name}"
        else:
            return f"({self.x}, {self.y})"


class Route:
    def __init__(self, cities):
        self.cities = cities
        self.path_len = 0
        for i in range(len(cities)-1):
            self.path_len += distance((cities[i].x, cities[i].y), (cities[i+1].x, cities[i+1].y))

    def __repr__(self):
        return str(self.cities) + f" len: {self.get_path_len}"

    @property
    def get_path_len(self):
        return self.path_len


class Population:
    def __init__(self, population):
        self.population = population
        self.population.sort(key=lambda x: x.path_len)

    def __repr__(self):
        res = ""
        for route in self.population:
            res += str(route) + "\n"
        return res[:-1]

    def mutation(self):
        for route in self.population:
            if random.random() < MUTATE:
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                while i == j:
                    i, j = random.randint(0, n-1), random.randint(0, n-1)
                route.cities[i], route.cities[j] = route.cities[j], route.cities[i]
        return Population(self.population)

    def selection(self, selection_method=selection):
        parents = []
        if selection_method == "ranking":
            for i in range(POPULATION_SIZE // 2):
                if len(parents) >= POPULATION_SIZE:
                    break
                for j in range(i+1, i+1+(POPULATION_SIZE // 5)):
                    if len(parents) >= POPULATION_SIZE:
                        break
                    parents.append(self.population[i])
                    parents.append(self.population[j])
        elif selection_method == "tournament":
            for _ in range(POPULATION_SIZE):
                bracket = random.sample(self.population, 3)
                match1, player3 = bracket[0:2], bracket[2]
                winner1 = match1[0] if match1[0].path_len < match1[1].path_len else match1[1]
                winner = winner1 if winner1.path_len < player3.path_len else player3
                parents.append(winner)

        return parents

    def crossover(self):
        parents = self.selection()
        children = []
        parents_len = len(parents)
        for i in range(parents_len // 2):
            new_children = make_children(parents[i], parents[i+1], crossover_method=crossover)
            children.append(new_children[0])
            children.append(new_children[1])
        new_gen = Population(children)
        return new_gen


population = []

info = input()

if info.isnumeric():  # unnamed random 2d cities
    infobase = None
    cities = []
    n = int(info)
    for _ in range(n):
        x = random.random()
        y = random.random()
        cities.append(City(x, y))
    for _ in range(POPULATION_SIZE):
        route = Route(random.sample(cities, n))
        population.append(route)

else:  # named cities
    infobase = info
    n = int(input())
    cities = []
    for _ in range(n):
        city_info = input().split()
        city = City(float(city_info[1]), float(city_info[2]), name=city_info[0])
        cities.append(city)
    for _ in range(POPULATION_SIZE):
        route = Route(random.sample(cities, n))
        population.append(route)


start = timeit.default_timer()     # start timer here ------------

gen = Population(population)
min_route, min_len = gen.population[0], gen.population[0].path_len

for _ in range(ITERATIONS):
    prev_top = gen.population[:50]
    gen = gen.crossover().mutation()
    new_routes = gen.population[:POPULATION_SIZE-50]
    new_population = prev_top + new_routes       # keep the previous top 50
    gen = Population(new_population)
    curr_gen_number += 1

    new_short_len = gen.population[0].path_len
    if new_short_len < min_len:
        min_route = gen.population[0]
        min_len = new_short_len
    if curr_gen_number % print_on_step == 0:
        numbers_to_print.append(min_len)

end = timeit.default_timer()       # end timer here -------------


if os.getenv("FMI_TIME_ONLY") == "1":
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
else:
    print(f"# TIMES_MS: alg={(end - start) * 1000}ms")
    for num in numbers_to_print:
        print(num)
    print(min_len)

    print("")
    if infobase:
        res = ""
        for i in range(n - 1):
            res += f"{min_route.cities[i]} -> "
        print(res + f"{min_route.cities[n - 1]}")

    print(min_len)
