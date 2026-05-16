"""
display_tree.py
───────────────
Builds Decision Trees from heart.csv (full dataset): baseline ID3 (My DT)
and max-depth pruned ID3 (My DT+). Prints text diagrams and saves PNG images.

Run with:   python display_tree.py

Fully self-contained — does NOT import from program.py, so it can
safely coexist in the same Ed workspace without causing circular imports.
"""

import math
import os
from collections import Counter

TRAINING_FILE = 'heart.csv'
OUTPUT_DIR = 'images'
MY_DT_IMAGE = os.path.join(OUTPUT_DIR, 'my_dt.png')
MY_DT_PLUS_IMAGE = os.path.join(OUTPUT_DIR, 'my_dt_plus.png')

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
    """Baseline ID3 — matches program.build_tree."""
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


def build_tree_plus(examples, feature_indices, default, max_depth, depth=0):
    """My DT+ — matches program.build_tree_plus (max-depth pre-pruning)."""
    if not examples:
        return default
    labels = [label for _, label in examples]
    if len(set(labels)) == 1:
        return labels[0]
    if depth >= max_depth:
        return majority_class(examples)
    if not feature_indices:
        return majority_class(examples)
    best = max(feature_indices, key=lambda i: information_gain(examples, i))
    node_default = majority_class(examples)
    remaining = [i for i in feature_indices if i != best]
    node = {'feature': best, 'default': node_default, 'children': {}}
    for value in sorted(set(f[best] for f, _ in examples)):
        subset = [(f, l) for f, l in examples if f[best] == value]
        node['children'][value] = build_tree_plus(
            subset, remaining, node_default, max_depth, depth + 1
        )
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


class _LayoutCounter:
    __slots__ = ('n',)

    def __init__(self):
        self.n = 0


def _layout(node, depth, counter):
    """Assign integer leaf indices; internal x = mean of child x."""
    if isinstance(node, str):
        x = counter.n
        counter.n += 1
        return {'x': x, 'y': depth, 'leaf': True, 'label': node}
    kids = []
    xs = []
    for value in sorted(node['children'].keys()):
        child = _layout(node['children'][value], depth + 1, counter)
        kids.append((value, child))
        xs.append(child['x'])
    x_center = sum(xs) / len(xs)
    return {
        'x': x_center,
        'y': depth,
        'leaf': False,
        'feature': node['feature'],
        'kids': kids,
    }


def _draw_tree(ax, laid_out, x_scale, y_scale):
    """Draw edges and node boxes from layout dict."""

    def draw_edges(node):
        if node['leaf']:
            return node['x'], node['y']
        px, py = node['x'], node['y']
        for value, child in node['kids']:
            cx, cy = draw_edges(child)
            ax.plot(
                [px * x_scale, cx * x_scale],
                [py * y_scale, cy * y_scale],
                color='#444444',
                linewidth=1.2,
                zorder=1,
            )
            mx = (px + cx) / 2 * x_scale
            my = (py + cy) / 2 * y_scale
            ax.text(
                mx, my,
                str(value),
                fontsize=7,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#aaaaaa'),
                zorder=3,
            )
        return px, py

    def draw_nodes(node):
        px, py = node['x'] * x_scale, node['y'] * y_scale
        if node['leaf']:
            text = node['label']
            color = '#ffe6e6' if text == 'died' else '#e6ffe6'
        else:
            text = FEATURE_NAMES[node['feature']]
            color = '#e6f0ff'
        ax.text(
            px, py, text,
            fontsize=8 if node['leaf'] else 9,
            ha='center',
            va='center',
            fontweight='bold' if not node['leaf'] else 'normal',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color, edgecolor='#333333'),
            zorder=2,
        )
        if not node['leaf']:
            for _, child in node['kids']:
                draw_nodes(child)

    draw_edges(laid_out)
    draw_nodes(laid_out)


def save_tree_image(tree, path, title):
    """Render tree as PNG using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required for PNG export. Install with: pip install matplotlib'
        ) from exc

    rcParams['font.family'] = 'sans-serif'
    counter = _LayoutCounter()
    laid_out = _layout(tree, 0, counter)
    n_leaves = max(counter.n, 1)
    fig_w = max(10, n_leaves * 0.55)
    fig_h = max(6, (laid_out['y'] if isinstance(laid_out, dict) else 0) + 3)
    x_scale = 1.2
    y_scale = -1.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12)
    max_x = max(counter.n - 1, 0) * x_scale
    min_y = laid_out['y'] * y_scale if isinstance(laid_out, dict) else 0
    _draw_tree(ax, laid_out, x_scale=x_scale, y_scale=y_scale)
    pad_x = 1.0
    pad_y = 1.0
    ax.set_xlim(-pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, pad_y)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    training = load_data(TRAINING_FILE)
    n_features = len(training[0][0])
    default = majority_class(training)

    tree_dt = build_tree(training, list(range(n_features)), default)
    tree_dt_plus = build_tree_plus(
        training, list(range(n_features)), default, max_depth=2
    )

    print('My DT - baseline ID3 on full heart.csv')
    print('=' * 50)
    print_tree(tree_dt)
    correct = sum(classify_instance(tree_dt, f) == l for f, l in training)
    print(f'Training accuracy: {correct}/{len(training)} = {correct/len(training):.3f}')
    print()

    print('My DT+ - max_depth=2 ID3 on full heart.csv')
    print('=' * 50)
    print_tree(tree_dt_plus)
    correct_plus = sum(classify_instance(tree_dt_plus, f) == l for f, l in training)
    print(f'Training accuracy: {correct_plus}/{len(training)} = {correct_plus/len(training):.3f}')
    print()

    save_tree_image(tree_dt, MY_DT_IMAGE, 'My DT (baseline ID3)')
    save_tree_image(tree_dt_plus, MY_DT_PLUS_IMAGE, 'My DT+ (max_depth=2)')
    print(f'Saved: {MY_DT_IMAGE}')
    print(f'Saved: {MY_DT_PLUS_IMAGE}')
