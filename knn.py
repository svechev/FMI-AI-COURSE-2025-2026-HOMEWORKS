import pandas as pd
import numpy as np
from random import shuffle, choice

# опитах с min-max скалиране, резултатите са подобни дори и без нормализация
# предполагам, че е защото стойностите в различните категории са близки


def scaling(entries):
    for col in range(len(entries[0])-1):
        col_i = [entry[col] for entry in entries]
        max_val, min_val = max(col_i), min(col_i)
        denum = max_val - min_val
        for row_index, entry in enumerate(entries):
            entries[row_index][col] = (entry[col] - min_val) / denum


def distance(x, y):
    return sum([abs(xi-yi) ** 2 for xi, yi in zip(x, y)]) ** 0.5


def get_majority_label(neighbour_list):
    labels = [x[1] for x in neighbour_list]
    hist = {label: labels.count(label) for label in labels}
    max_labels = [k for (k, v) in hist.items() if v == max(hist.values())]
    return choice(max_labels)


def knn(xs, ys, xt, k):
    distance_pair_list = []
    for xi, yi in zip(xs, ys):
        di = distance(xi, xt)
        distance_pair_list.append((xi, yi, di))
    distance_pair_list.sort(key=lambda x: x[2])
    k_neighbours_list = distance_pair_list[:k]
    return get_majority_label(k_neighbours_list)


def split_data_10fold(entries):
    sets = [[] for _ in range(10)]
    shuffle(entries)
    possible_vals = set([entry[4] for entry in entries])
    for val in possible_vals:
        entries_with_val = [entry for entry in entries if entry[4] == val]

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


def cross_validation_10fold(entries):
    sets = split_data_10fold(entries)
    accuracy_sum = 0
    accuracies = []
    for i in range(10):
        test_set = sets[i]
        train_set = []
        for k in range(10):
            if k == i:
                continue
            train_set += sets[k]
        test_i_accuracy = set_validation(train_set, test_set)
        accuracies.append(test_i_accuracy)
        print(f"Accuracy fold {i+1}: {test_i_accuracy}%")
        accuracy_sum += test_i_accuracy

    return round(accuracy_sum / 10, 2), np.std(accuracies, dtype=float)


def set_validation(train_set, test_set):
    successes = 0
    train_Xs, train_Ys = [entry[:-1] for entry in train_set], [entry[-1] for entry in train_set]
    for entry in test_set:
        test_Xi, test_Yi = entry[:-1], entry[-1]
        knn_res = knn(train_Xs, train_Ys, test_Xi, K)
        if knn_res == test_Yi:
            successes += 1
    return round(successes / len(test_set), 2) * 100


def split_data(entries):
    train, test = [], []
    classes = set([entry[-1] for entry in entries])

    for cls in classes:
        cls_entries = [entry for entry in entries if entry[-1] == cls]
        split_at = int(len(cls_entries) * 0.8)
        train += cls_entries[:split_at]
        test += cls_entries[split_at:]

    return train, test


df = pd.read_csv("iris.data", delimiter=",")
entries = df.values.tolist()
# scaling(entries)
shuffle(entries)
K = int(input())

train, test = split_data(entries)

print("1. Train Set Accuracy:")
train_acc = set_validation(train, train)
print(f"Accuracy: {train_acc}%\n")


print("2. 10-Fold Cross-Validation Results:\n")

avg_acc_10fold, sd = cross_validation_10fold(entries)

print(f"\nAverage Accuracy: {avg_acc_10fold}%")
print(f"Standard Deviation: {sd:.2f}%\n")

print("3. Test Set Accuracy:")
test_acc = set_validation(train, test)
print(f"Accuracy: {test_acc}%")
