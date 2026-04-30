"""
display_tree.py
───────────────
Builds a Decision Tree from heart.csv (full dataset) and prints a
text-based diagram for inclusion in the report.

Run with:   python display_tree.py

Fully self-contained — does NOT import from program.py, so it can
safely coexist in the same Ed workspace without causing circular imports.
"""

import math
from collections import Counter

TRAINING_FILE = 'heart.csv'

FEATURE_NAMES = [
    'age', 'anaemia', 'CPK', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine',
    'serum_sodium', 'sex', 'smoking',
]


def load_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split(',')
            data.append((row[:-1], row[-1]))
    return data


def entropy(examples):
    if not examples:
        return 0.0
    total = len(examples)
    counts = Counter(label for _, label in examples).values()
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def information_gain(examples, feature_idx):
    total = len(examples)
    partitions = {}
    for features, label in examples:
        partitions.setdefault(features[feature_idx], []).append((features, label))
    weighted = sum((len(s) / total) * entropy(s) for s in partitions.values())
    return entropy(examples) - weighted


def majority_class(examples):
    if not examples:
        return None
    counter = Counter(label for _, label in examples)
    return min(counter, key=lambda c: (-counter[c], c))


def build_tree(examples, feature_indices, default):
    if not examples:
        return default
    labels = [label for _, label in examples]
    if len(set(labels)) == 1:
        return labels[0]
    if not feature_indices:
        return majority_class(examples)
    best = max(feature_indices, key=lambda i: information_gain(examples, i))
    node_default = majority_class(examples)
    remaining = [i for i in feature_indices if i != best]
    node = {'feature': best, 'default': node_default, 'children': {}}
    for value in sorted(set(f[best] for f, _ in examples)):
        subset = [(f, l) for f, l in examples if f[best] == value]
        node['children'][value] = build_tree(subset, remaining, node_default)
    return node


def classify_instance(tree, instance):
    if isinstance(tree, str):
        return tree
    value = instance[tree['feature']]
    if value in tree['children']:
        return classify_instance(tree['children'][value], instance)
    return tree['default']


def print_tree(node, indent=0):
    prefix = '  ' * indent
    if isinstance(node, str):
        print(f'{prefix}-> {node}')
        return
    feat_name = FEATURE_NAMES[node['feature']]
    for value, subtree in sorted(node['children'].items()):
        print(f'{prefix}{feat_name} = {value}:')
        print_tree(subtree, indent + 1)


if __name__ == '__main__':
    training = load_data(TRAINING_FILE)
    n_features = len(training[0][0])
    default = majority_class(training)
    tree = build_tree(training, list(range(n_features)), default)

    print('Decision Tree built on full heart.csv dataset')
    print('=' * 50)
    print_tree(tree)
    print()

    correct = sum(classify_instance(tree, f) == l for f, l in training)
    print(f'Training accuracy: {correct}/{len(training)} = {correct/len(training):.3f}')
