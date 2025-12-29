import math

import pandas as pd
import numpy as np
from random import shuffle

LAMBDA = 1


def fill_unknown(entries):
    for r in range(len(entries)):
        for c in range(1, len(entries[0])):  # column 0 can't be "?"
            if entries[r][c] == '?':
                filtered_answers = [entry[c] for entry in entries if entry[0] == entries[r][0]]  # only reps or dems
                yes_count = filtered_answers.count('y')
                no_count = filtered_answers.count('n')
                entries[r][c] = 'y' if yes_count > no_count else 'n'


def get_likelihood_table(data):
    republicans = data[data[:, 0] == "republican"]
    democrats = data[data[:, 0] == "democrat"]

    reps_num = republicans.shape[0]
    dems_num = democrats.shape[0]

    P_rep = reps_num / data.shape[0]
    P_dem = dems_num / data.shape[0]

    l_hood_table = []
    cols = df.columns[1:]
    answers = ['n', 'y', '?'] if not FILTER_DATA else ['n', 'y']
    num_values = len(answers)

    denum_reps = reps_num + num_values * LAMBDA
    denum_dems = dems_num + num_values * LAMBDA

    for i in range(len(cols)):   # i-th feature
        res = {answer: {} for answer in answers}
        for answer in answers:
            ans_prob_rep = (republicans[republicans[:, i+1] == answer].shape[0] + LAMBDA) / denum_reps
            ans_prob_dem = (democrats[democrats[:, i+1] == answer].shape[0] + LAMBDA) / denum_dems
            res[answer]["republican"] = ans_prob_rep
            res[answer]["democrat"] = ans_prob_dem
        l_hood_table.append(res)

    return l_hood_table, P_rep, P_dem


def classifier(l_hood_table, P_rep, P_dem, Xt):
    rep_chance = math.log2(P_rep)
    dem_chance = math.log2(P_dem)
    for col_num in range(len(l_hood_table)):
        answer = Xt[col_num]
        rep_chance += math.log2(l_hood_table[col_num][answer]["republican"])
        dem_chance += math.log2(l_hood_table[col_num][answer]["democrat"])
    return "republican" if rep_chance > dem_chance else "democrat"


def validate_set(l_hood_table, P_rep, P_dem, data):
    successes = 0
    set_size = data.shape[0]
    Xs = data[:, 1:]
    Cs = data[:, 0]
    for k in range(set_size):
        assumed_class = classifier(l_hood_table, P_rep, P_dem, Xs[k])
        if assumed_class == Cs[k]:
            successes += 1

    return round(successes / set_size, 2) * 100


def split_data_10fold(entries):
    sets = [[] for _ in range(10)]
    shuffle(entries)
    possible_vals = set([entry[0] for entry in entries])
    for val in possible_vals:
        entries_with_val = [entry for entry in entries if entry[0] == val]

        needed_size = len(entries_with_val) // 10
        set_sizes = [needed_size for _ in range(10)]
        for i in range(len(entries_with_val) % 10):
            set_sizes[i] += 1

        start_index = 0
        for i in range(10):
            end_index = start_index + set_sizes[i]
            sets[i] += entries_with_val[start_index:end_index]
            start_index = end_index

    return sets


def cross_validation_10fold(data):
    sets = split_data_10fold(data)
    accuracy_sum = 0
    accuracies = []
    for i in range(10):
        test_set = sets[i]
        train_set = []
        for k in range(10):
            if i == k:
                continue
            train_set += sets[k]
        new_table, new_P_rep, new_P_dem = get_likelihood_table(np.array(train_set))

        test_i_accuracy = validate_set(new_table, new_P_rep, new_P_dem, np.array(test_set))
        accuracies.append(test_i_accuracy)
        print(f"Accuracy Fold {i+1}: {test_i_accuracy}%")
        accuracy_sum += test_i_accuracy

    return round(accuracy_sum / 10, 2), np.std(accuracies, dtype=float)


def split_data(entries):
    train, test = [], []
    classes = set([entry[0] for entry in entries])

    for cls in classes:
        cls_entries = [entry for entry in entries if entry[0] == cls]
        split_at = int(len(cls_entries) * 0.8)
        train += cls_entries[:split_at]
        test += cls_entries[split_at:]

    return train, test


df = pd.read_csv("house-votes-84.data")
df = df.sample(frac=1)
entries = df.values.tolist()
FILTER_DATA = False


start = int(input())
if start == 1:
    fill_unknown(entries)
    FILTER_DATA = True
elif start != 0:
    raise "Invalid input! Must be 0 or 1"

train, test = split_data(entries)

table, P_rep, P_dem = get_likelihood_table(np.array(train))


print("1. Train Set Accuracy:")
train_acc = validate_set(table, P_rep, P_dem, np.array(train))
print(f"Accuracy: {train_acc}%\n")


print("10-Fold Cross-Validation Results:\n")
avg_acc_10fold, sd = cross_validation_10fold(np.array(entries))
print(f"\nAverage Accuracy: {avg_acc_10fold}%")
print(f"Standard Deviation: {sd:.2f}%\n")

print("2. Test Set Accuracy:")
test_acc = validate_set(table, P_rep, P_dem, np.array(test))
print(f"Accuracy: {test_acc}%")
