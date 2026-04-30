import math
from collections import Counter


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(filename, has_class=True):
    """Load CSV data. Returns list of (features, label) tuples if has_class,
    otherwise list of feature lists."""
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split(',')
            if has_class:
                data.append((row[:-1], row[-1]))
            else:
                data.append(row)
    return data


# ── KNN ───────────────────────────────────────────────────────────────────────

def euclidean_distance(a, b):
    """Euclidean distance for nominal attributes.

    Each attribute contributes 0 if values match, 1 if they differ.
    Distance = sqrt(number of mismatching attributes).
    """
    return math.sqrt(sum(x != y for x, y in zip(a, b)))


def classify_knn(training_filename, testing_filename, k):
    """K-Nearest Neighbour classifier using Euclidean distance for nominal data.

    Tie-breaking rules:
      - Distance ties: lower row index in the training file takes priority
        (guaranteed by stable sort preserving original insertion order).
      - Voting ties: predict 'died'.
    """
    training = load_data(training_filename, has_class=True)
    testing  = load_data(testing_filename,  has_class=False)

    results = []
    for test_instance in testing:
        # Pair each training instance with its distance; enumerate preserves row index
        distances = [
            (euclidean_distance(test_instance, features), row_idx, label)
            for row_idx, (features, label) in enumerate(training)
        ]
        # Sort by distance, then by row index for distance ties (stable, explicit)
        distances.sort(key=lambda x: (x[0], x[1]))

        k_nearest = distances[:k]
        votes = Counter(label for _, _, label in k_nearest)

        # Most votes wins; voting tie → predict 'died'
        max_votes = max(votes.values())
        if list(votes.values()).count(max_votes) > 1:
            prediction = 'died'
        else:
            prediction = max(votes, key=lambda c: votes[c])
        results.append(prediction)

    return results


# ── Decision Tree (ID3) ────────────────────────────────────────────────────────

def entropy(examples):
    """Shannon entropy of the class labels in examples."""
    if not examples:
        return 0.0
    total = len(examples)
    counts = Counter(label for _, label in examples).values()
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def information_gain(examples, feature_idx):
    """Information gain of splitting examples on the given feature index."""
    parent_entropy = entropy(examples)
    total = len(examples)

    # Partition examples by the feature's value
    partitions = {}
    for features, label in examples:
        v = features[feature_idx]
        partitions.setdefault(v, []).append((features, label))

    weighted_child_entropy = sum(
        (len(subset) / total) * entropy(subset)
        for subset in partitions.values()
    )
    return parent_entropy - weighted_child_entropy


def majority_class(examples):
    """Most common class in examples; tie-break predicts 'died'."""
    if not examples:
        return None
    counter = Counter(label for _, label in examples)
    # min on (-count, label) → highest count wins; 'died' < 'survived' breaks ties
    return min(counter, key=lambda c: (-counter[c], c))


def build_tree(examples, feature_indices, default):
    """Recursively build an ID3 decision tree.

    Returns either a leaf (string class label) or an internal node dict:
        {'feature': int, 'default': str, 'children': {value: subtree, ...}}
    """
    # No examples left — use the parent's majority class
    if not examples:
        return default

    labels = [label for _, label in examples]

    # Pure node — all examples have the same class
    if len(set(labels)) == 1:
        return labels[0]

    # No features left — return majority class
    if not feature_indices:
        return majority_class(examples)

    # Choose the feature with the highest information gain
    best_idx = max(feature_indices, key=lambda i: information_gain(examples, i))

    node_default = majority_class(examples)
    remaining    = [i for i in feature_indices if i != best_idx]

    node = {
        'feature':  best_idx,
        'default':  node_default,   # fallback for unseen values at test time
        'children': {}
    }

    for value in sorted(set(features[best_idx] for features, _ in examples)):
        subset = [(f, l) for f, l in examples if f[best_idx] == value]
        node['children'][value] = build_tree(subset, remaining, node_default)

    return node


def classify_instance(tree, instance):
    """Traverse the tree to classify a single instance."""
    if isinstance(tree, str):
        return tree
    value = instance[tree['feature']]
    if value in tree['children']:
        return classify_instance(tree['children'][value], instance)
    # Unseen feature value — fall back to the majority class stored at this node
    return tree['default']


def classify_dt(training_file, testing_file):
    """Decision Tree classifier built with ID3 (information gain)."""
    training = load_data(training_file, has_class=True)
    testing  = load_data(testing_file,  has_class=False)

    if not training:
        return []

    n_features = len(training[0][0])
    default    = majority_class(training)
    tree       = build_tree(training, list(range(n_features)), default)

    return [classify_instance(tree, inst) for inst in testing]

