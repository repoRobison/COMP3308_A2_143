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


# ── KNN+ ──────────────────────────────────────────────────────────────────────

def _ig_weights(training):
    """Compute sqrt-information-gain weights for each feature.

    Square-root scaling keeps strong predictors important without letting one
    attribute completely dominate the distance.
    """
    n = len(training[0][0])
    raw = [math.sqrt(max(information_gain(training, i), 0.0)) for i in range(n)]
    total = sum(raw)
    if total == 0:
        return [1.0] * n
    # Normalise so weights sum to n - a feature with average signal gets weight 1
    return [w / total * n for w in raw]


def _vdm_tables(training, alpha=1):
    """Pre-compute Value Difference Metric probability tables.

    For each feature value, store P(class | value). Laplace smoothing avoids
    over-trusting rare values in this small dataset.
    """
    labels = sorted(set(label for _, label in training))
    n_features = len(training[0][0])
    value_counts = [Counter() for _ in range(n_features)]
    class_counts = [{} for _ in range(n_features)]

    for features, label in training:
        for i, value in enumerate(features):
            value_counts[i][value] += 1
            class_counts[i].setdefault(value, Counter())[label] += 1

    global_counts = Counter(label for _, label in training)
    global_probs = {
        label: (global_counts[label] + alpha) / (len(training) + alpha * len(labels))
        for label in labels
    }

    tables = []
    for i in range(n_features):
        feature_table = {}
        for value, count in value_counts[i].items():
            denom = count + alpha * len(labels)
            feature_table[value] = {
                label: (class_counts[i][value][label] + alpha) / denom
                for label in labels
            }
        tables.append(feature_table)

    return labels, tables, global_probs


def _vdm_distance(a, b, labels, tables, global_probs, weights):
    """Value Difference Metric distance for categorical data."""
    total = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        x_probs = tables[i].get(x, global_probs)
        y_probs = tables[i].get(y, global_probs)
        value_diff = sum(abs(x_probs[label] - y_probs[label]) for label in labels)
        total += weights[i] * value_diff * value_diff
    return math.sqrt(total)


def classify_knn_plus(training_filename, testing_filename, k=21):
    """KNN+: Value Difference Metric + sqrt-information-gain feature weighting.

    VDM is designed for categorical data: two different values are considered
    close if they have similar class distributions in the training set. For
    this heart-failure dataset, that means values are compared by how similarly
    they predict 'died' or 'survived', rather than by a plain mismatch.

    Tie-breaking:
      • Distance ties: lower training-row index wins (stable sort).
      • Voting ties: predict 'died'.
    """
    training = load_data(training_filename, has_class=True)
    testing  = load_data(testing_filename,  has_class=False)

    weights = _ig_weights(training)
    labels, tables, global_probs = _vdm_tables(training)

    results = []
    for test_instance in testing:
        distances = []
        for row_idx, (features, label) in enumerate(training):
            d = _vdm_distance(test_instance, features, labels, tables, global_probs, weights)
            distances.append((d, row_idx, label))

        distances.sort(key=lambda x: (x[0], x[1]))
        k_nearest = distances[:k]
        votes = Counter(label for _, _, label in k_nearest)

        max_votes = max(votes.values())
        if list(votes.values()).count(max_votes) > 1:
            prediction = 'died'
        else:
            prediction = max(votes, key=lambda c: votes[c])
        results.append(prediction)

    return results


# ── DT+ ───────────────────────────────────────────────────────────────────────

def build_tree_plus(examples, feature_indices, default, max_depth, depth=0):
    """ID3 decision tree with max-depth pre-pruning.

    Identical to build_tree except: once max_depth is reached, stop splitting
    and return the current majority class. This keeps only the strongest
    high-level rules and avoids fitting tiny patient subgroups.
    """
    if not examples:
        return default

    labels = [label for _, label in examples]

    if len(set(labels)) == 1:
        return labels[0]

    # Pre-pruning: stop before the tree becomes too specific
    if depth >= max_depth:
        return majority_class(examples)

    if not feature_indices:
        return majority_class(examples)

    best_idx = max(feature_indices, key=lambda i: information_gain(examples, i))
    node_default = majority_class(examples)
    remaining    = [i for i in feature_indices if i != best_idx]

    node = {
        'feature':  best_idx,
        'default':  node_default,
        'children': {}
    }

    for value in sorted(set(features[best_idx] for features, _ in examples)):
        subset = [(f, l) for f, l in examples if f[best_idx] == value]
        node['children'][value] = build_tree_plus(
            subset, remaining, node_default, max_depth, depth + 1
        )

    return node


def classify_dt_plus(training_filename, testing_filename, max_depth=2):
    """DT+: ID3 with max-depth pre-pruning.

    Improvement over plain ID3:
      The tree is only allowed to grow to max_depth. This is useful here
      because the dataset has only ~300 patients and multi-way categorical
      splits can quickly create small, noisy branches. A shallow tree keeps
      the strongest rules and reduces overfitting.
    """
    training = load_data(training_filename, has_class=True)
    testing  = load_data(testing_filename,  has_class=False)

    if not training:
        return []

    n_features = len(training[0][0])
    default    = majority_class(training)
    tree       = build_tree_plus(training, list(range(n_features)), default, max_depth)

    return [classify_instance(tree, inst) for inst in testing]

