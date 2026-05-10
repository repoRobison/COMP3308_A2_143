import argparse
import csv
import os
from collections import Counter

import program


FOLD_FILE = 'heart-folds.csv'
TRAIN_FILE = '_cv_train.csv'
TEST_FILE = '_cv_test.csv'
POSITIVE_CLASS = 'died'


def read_folds(filename):
    """Read heart-folds.csv into a list of folds, each a list of full rows."""
    folds = []
    current = None

    with open(filename) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('fold'):
                current = []
                folds.append(current)
            elif current is not None:
                current.append(line)

    return folds


def write_fold_files(folds, test_index):
    """Write temporary training/testing files for one CV fold.

    The classifiers in program.py expect filenames. The training file includes
    class labels; the testing file contains only feature values.
    """
    test_rows = folds[test_index]
    train_rows = [
        row
        for i, fold in enumerate(folds)
        if i != test_index
        for row in fold
    ]

    with open(TRAIN_FILE, 'w', newline='') as f:
        f.write('\n'.join(train_rows))

    truth = []
    with open(TEST_FILE, 'w', newline='') as f:
        for row in test_rows:
            parts = row.split(',')
            truth.append(parts[-1])
            f.write(','.join(parts[:-1]) + '\n')

    return truth


def class_metrics(y_true, y_pred, label):
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_predictions(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    per_class = {
        label: class_metrics(y_true, y_pred, label)
        for label in labels
    }
    macro_f1 = sum(metrics[2] for metrics in per_class.values()) / len(labels)
    positive_precision, positive_recall, positive_f1 = per_class[POSITIVE_CLASS]

    return {
        'accuracy': accuracy,
        'died_precision': positive_precision,
        'died_recall': positive_recall,
        'died_f1': positive_f1,
        'macro_f1': macro_f1,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def run_cross_validation(fold_file, k, k_plus, max_depth):
    folds = read_folds(fold_file)
    if len(folds) != 10:
        raise ValueError(f'Expected 10 folds, found {len(folds)}')

    classifiers = {
        f'KNN (k={k})': (
            lambda train, test: program.classify_knn(train, test, k)
        ),
        f'KNN+ (VDM+sqrtIG, k={k_plus})': (
            lambda train, test: program.classify_knn_plus(train, test, k_plus)
        ),
        'DT (ID3)': (
            lambda train, test: program.classify_dt(train, test)
        ),
        f'DT+ (max_depth={max_depth})': (
            lambda train, test: program.classify_dt_plus(
                train, test, max_depth=max_depth
            )
        ),
    }

    fold_results = []
    all_truth = {name: [] for name in classifiers}
    all_predictions = {name: [] for name in classifiers}

    try:
        for fold_index in range(len(folds)):
            truth = write_fold_files(folds, fold_index)

            for name, classifier in classifiers.items():
                predictions = classifier(TRAIN_FILE, TEST_FILE)
                metrics = evaluate_predictions(truth, predictions)

                fold_results.append({
                    'classifier': name,
                    'fold': fold_index + 1,
                    **metrics,
                })
                all_truth[name].extend(truth)
                all_predictions[name].extend(predictions)
    finally:
        for filename in (TRAIN_FILE, TEST_FILE):
            if os.path.exists(filename):
                os.remove(filename)

    summary = []
    for name in classifiers:
        rows = [row for row in fold_results if row['classifier'] == name]
        overall = evaluate_predictions(all_truth[name], all_predictions[name])
        summary.append({
            'classifier': name,
            'mean_fold_accuracy': mean([row['accuracy'] for row in rows]),
            'overall_accuracy': overall['accuracy'],
            'mean_died_precision': mean([row['died_precision'] for row in rows]),
            'mean_died_recall': mean([row['died_recall'] for row in rows]),
            'mean_died_f1': mean([row['died_f1'] for row in rows]),
            'mean_macro_f1': mean([row['macro_f1'] for row in rows]),
        })

    return fold_results, summary


def write_csv(filename, rows):
    if not rows:
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    print('10-fold cross-validation summary')
    print('-' * 95)
    print(
        f'{"Classifier":32}  {"Mean Acc":>8}  {"Overall":>8}  '
        f'{"Died P":>8}  {"Died R":>8}  {"Died F1":>8}  {"Macro F1":>8}'
    )
    print('-' * 95)
    for row in summary:
        print(
            f'{row["classifier"]:32}  '
            f'{row["mean_fold_accuracy"]:8.4f}  '
            f'{row["overall_accuracy"]:8.4f}  '
            f'{row["mean_died_precision"]:8.4f}  '
            f'{row["mean_died_recall"]:8.4f}  '
            f'{row["mean_died_f1"]:8.4f}  '
            f'{row["mean_macro_f1"]:8.4f}'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate classifiers using heart-folds.csv 10-fold CV.'
    )
    parser.add_argument('--folds', default=FOLD_FILE)
    parser.add_argument('--k', type=int, default=21)
    parser.add_argument('--k-plus', type=int, default=21)
    parser.add_argument('--max-depth', type=int, default=2)
    args = parser.parse_args()

    fold_results, summary = run_cross_validation(
        args.folds, args.k, args.k_plus, args.max_depth
    )

    write_csv('cv_fold_results.csv', fold_results)
    write_csv('cv_summary.csv', summary)
    print_summary(summary)
    print()
    print('Wrote cv_fold_results.csv and cv_summary.csv')


if __name__ == '__main__':
    main()
