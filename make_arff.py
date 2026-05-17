"""
Convert heart.csv into heart.arff for Weka.

Auto-detects each feature's unique values from the data, so the ARFF header
exactly matches what the file contains.
"""

from collections import defaultdict


INPUT_FILE  = 'heart.csv'
OUTPUT_FILE = 'heart.arff'
RELATION    = 'heart'

FEATURE_NAMES = [
    'age', 'anaemia', 'CPK', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine',
    'serum_sodium', 'sex', 'smoking',
]
CLASS_NAME = 'class'


def needs_quoting(value):
    """ARFF requires quotes around values containing special characters."""
    return any(c in value for c in '[]-, ') or value == ''


def quote(value):
    return f"'{value}'" if needs_quoting(value) else value


def main():
    with open(INPUT_FILE) as f:
        rows = [line.strip().split(',') for line in f if line.strip()]

    if not rows:
        raise ValueError(f'{INPUT_FILE} is empty')

    n_cols = len(rows[0])
    if n_cols != len(FEATURE_NAMES) + 1:
        raise ValueError(
            f'Expected {len(FEATURE_NAMES) + 1} columns, found {n_cols}'
        )

    column_values = defaultdict(set)
    for row in rows:
        for i, value in enumerate(row):
            column_values[i].add(value)

    lines = [f'@relation {RELATION}', '']

    for i, name in enumerate(FEATURE_NAMES):
        values = sorted(column_values[i])
        value_list = ','.join(quote(v) for v in values)
        lines.append(f'@attribute {name} {{{value_list}}}')

    class_values = sorted(column_values[len(FEATURE_NAMES)])
    class_list = ','.join(quote(v) for v in class_values)
    lines.append(f'@attribute {CLASS_NAME} {{{class_list}}}')

    lines.append('')
    lines.append('@data')

    for row in rows:
        lines.append(','.join(quote(v) for v in row))

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote {OUTPUT_FILE}')
    print(f'  {len(rows)} instances')
    print(f'  {len(FEATURE_NAMES)} features + 1 class')
    print('\nValue counts per attribute:')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name}: {sorted(column_values[i])}')
    print(f'  {CLASS_NAME}: {sorted(column_values[len(FEATURE_NAMES)])}')


if __name__ == '__main__':
    main()
