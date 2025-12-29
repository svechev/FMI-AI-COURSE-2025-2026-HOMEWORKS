import math

import pandas as pd
import numpy as np
from random import shuffle


K = 3
N = 5
G = 0.01
implemented_pre_pruning = ["K", "N", "G"]
implemented_post_pruning = ["E"]
pruning = []


col_names = ["Class",
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat"]
target_col = col_names.index("Class")


def filter_data(entries):
    for row in range(len(entries)):
        for col in range(1, 10):  # cols 1 to 9 are the attributes
            if entries[row][col] == "?":
                entries_with_class = [entry for entry in entries if entry[0] == entries[row][0]]
                all_vals = [entry[col] for entry in entries_with_class if entry[col] != "?"]
                unique_vals = set(all_vals)
                vals_hist = {val: all_vals.count(val) for val in unique_vals}
                most_common = max(vals_hist, key=vals_hist.get)
                entries[row][col] = most_common


class Node:
    def __init__(self, entries, cols_to_check, depth):
        self.depth = depth
        self.entries = entries
        self.cols_to_check = cols_to_check
        self.next_attr = None
        self.children = []
        self.answer = None
        self.is_leaf = False

    def build_tree(self):
        # check if we need to cut:

        # first check if the answer is the same for every entry
        target_vals = [entry[target_col] for entry in self.entries]
        unique_target_vals = set(target_vals)
        if len(unique_target_vals) == 1:
            self.answer = target_vals[0]  # if it's one, doesn't matter which one we get
            self.is_leaf = True
            return

        vals_left = {val: target_vals.count(val) for val in unique_target_vals}
        self.answer = max(vals_left, key=vals_left.get)

        # check if we have no more columns to check or pre-pruning
        if len(self.cols_to_check) == 0:
            self.is_leaf = True
            return
        if "K" in pruning and len(self.entries) < K:
            self.is_leaf = True
            return
        if "N" in pruning and self.depth >= N:
            self.is_leaf = True
            return

        # we can build the tree now
        if len(self.cols_to_check) == 1:   # 1 column - just return the column
            best_col = self.cols_to_check[0]
            col_to_check = col_names.index(best_col)
            inf_gain = info_gain(self.entries, col_to_check, target_col)
        else:
            best_col, inf_gain = get_attr_best_info_gain(self.entries, self.cols_to_check, target_col)

        # low information gain?
        if "G" in pruning and inf_gain < G:
            self.is_leaf = True
            return

        self.next_attr = best_col
        self.cols_to_check.remove(best_col)  # we don't need to check it in the future
        col_to_check = col_names.index(best_col)

        next_attr_vals = set(entry[col_to_check] for entry in self.entries)

        for val in next_attr_vals:
            filtered_entries = [entry for entry in self.entries if entry[col_to_check] == val]
            self.children.append((val, Node(filtered_entries, list(self.cols_to_check), self.depth+1)))
        for _, child in self.children:
            child.build_tree()


def get_id3_tree(entries, cols_to_check):
    tree = Node(entries, cols_to_check, 0)
    tree.build_tree()
    return tree


def prune(curr_node, full_tree, validation_set, prev_best_acc):
    for _, child in curr_node.children:
        prune(child, full_tree, validation_set, prev_best_acc)
    if not curr_node.is_leaf:
        curr_node.is_leaf = True

        new_acc = check_accuracy_on_set(full_tree, validation_set)
        if new_acc >= prev_best_acc[0]:
            prev_best_acc[0] = new_acc
            curr_node.children = []
        else:
            curr_node.is_leaf = False


def get_id3_tree_REP(entries, cols_to_check):  # entries here is the original train data
    train, validation = split_data(entries, 0.9)
    tree = get_id3_tree(entries, cols_to_check)  # tree is the first node
    prev_acc = [check_accuracy_on_set(tree, validation)]
    prune(tree, tree, validation, prev_acc)
    return tree


def get_answer(tree, entry):
    if tree.is_leaf:  # leaf -> answer
        return tree.answer
    else:  # move to the correct subtree
        col_to_check = col_names.index(tree.next_attr)
        match_node = [(edge, node) for (edge, node) in tree.children if edge == entry[col_to_check]]

        if len(match_node) == 1:
            return get_answer(match_node[0][1], entry)
        elif len(match_node) == 0:  # corner case - new value for the attribute, not seen in the train set
            child_answers = [child.answer for _, child in tree.children]
            if child_answers:
                vals_hist = {val: child_answers.count(val) for val in set(child_answers)}
                return max(vals_hist, key=vals_hist.get)
            else:
                return tree.answer


def entropy(entries, target_col):
    all_values = [entry[target_col] for entry in entries]
    unique_values = set(all_values)
    values_count = len(all_values)

    entr = 0
    for value in unique_values:
        p_v = all_values.count(value) / values_count
        entr += p_v * math.log2(p_v) * (-1)
    return entr


def entropy_joint(entries, col_to_check, target_col):
    possible_values = [entry[col_to_check] for entry in entries]
    unique_values = set(possible_values)
    values_count = len(possible_values)

    entr_joint = 0
    for value in unique_values:
        p_val = possible_values.count(value) / values_count
        filtered_data = [entry for entry in entries if entry[col_to_check] == value]
        H_Sv = entropy(filtered_data, target_col)
        entr_joint += p_val * H_Sv
    return entr_joint


def info_gain(entries, col_to_check, target_col):
    return entropy(entries, target_col) - entropy_joint(entries, col_to_check, target_col)


def get_attr_best_info_gain(entries, cols_to_check, target_col):
    max_gain = -1
    best_col = None

    for col_name in cols_to_check:
        col_index = col_names.index(col_name)
        inf_gain = info_gain(entries, col_index, target_col)
        #print(inf_gain)
        if inf_gain > max_gain:
            max_gain = inf_gain
            best_col = col_name

    return best_col, max_gain


def check_accuracy_on_set(tree, entries):
    successes = 0
    set_size = len(entries)
    for entry in entries:
        assumed_class = get_answer(tree, entry)
        if assumed_class == entry[target_col]:
            successes += 1

    return round(successes / set_size * 100, 2)


def cross_validation_10fold(entries, cols_to_check):
    sets = split_data_10fold(entries)
    accuracy_sum = 0
    accuracies = []
    for i in range(10):
        test_set = sets[i]
        train_set = []
        for k in range(10):
            if i == k:
                continue
            train_set += sets[k]

        if "E" in pruning:
            new_tree = get_id3_tree_REP(train_set, list(cols_to_check))
        else:
            new_tree = get_id3_tree(train_set, list(cols_to_check))
        test_i_accuracy = check_accuracy_on_set(new_tree, test_set)
        accuracies.append(test_i_accuracy)
        print(f"Accuracy Fold {i+1}: {test_i_accuracy}%")
        accuracy_sum += test_i_accuracy

    return round(accuracy_sum / 10, 2), np.std(accuracies, dtype=float)


def split_data(entries, train_ratio=0.8):
    train, test = [], []
    classes = set([entry[target_col] for entry in entries])

    for cls in classes:
        cls_entries = [entry for entry in entries if entry[target_col] == cls]
        split_at = int(len(cls_entries) * train_ratio)
        train += cls_entries[:split_at]
        test += cls_entries[split_at:]

    return train, test


def split_data_10fold(entries):
    sets = [[] for _ in range(10)]
    shuffle(entries)
    possible_vals = set([entry[target_col] for entry in entries])
    for val in possible_vals:
        entries_with_val = [entry for entry in entries if entry[target_col] == val]

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


command = input().split()
match int(command[0]):
    case 0 if len(command) == 1:
        pruning += implemented_pre_pruning
    case 1 if len(command) == 1:
        pruning += implemented_post_pruning
    case 2 if len(command) == 1:
        pruning += implemented_pre_pruning
        pruning += implemented_post_pruning
    case _:
        pruning += command[1:]   # letters are passed, add them to the pruning methods


df = pd.read_csv("breast-cancer.data")
df = df.sample(frac=1)
entries = df.values.tolist()
filter_data(entries)


cols_to_check = list(col_names[1:])

train, test = split_data(entries)

if "E" in pruning:
    tree = get_id3_tree_REP(train, list(cols_to_check))
else:
    tree = get_id3_tree(train, list(cols_to_check))

print("1. Train Set Accuracy:")
train_acc = check_accuracy_on_set(tree, train)
print(f"Accuracy: {train_acc}%\n")


print("10-Fold Cross-Validation Results:\n")

avg_acc_10fold, sd = cross_validation_10fold(entries, cols_to_check)

print(f"\nAverage Accuracy: {avg_acc_10fold}%")
print(f"Standard Deviation: {sd:.2f}%\n")

print("2. Test Set Accuracy:")
test_acc = check_accuracy_on_set(tree, test)
print(f"Accuracy: {test_acc}%")
