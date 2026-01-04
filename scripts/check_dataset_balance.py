"""
Quick script to check dataset balance
"""

import csv
from collections import Counter
from pathlib import Path

csv_path = Path('Data/Labels/dataset.csv')

if not csv_path.exists():
    print(f"❌ CSV file not found: {csv_path}")
    exit(1)

# Read CSV and count labels
labels = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row['label'])

# Count labels
label_counts = Counter(labels)

print("=" * 60)
print("Dataset Balance Check")
print("=" * 60)
print(f"\nTotal samples: {len(labels)}")
print(f"Number of classes: {len(label_counts)}")

print(f"\nPer label:")
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
max_count = sorted_labels[0][1]
min_count = sorted_labels[-1][1]
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

for label, count in sorted_labels:
    status = "[OK]" if count >= 20 else "[!]"
    if count == max_count:
        status = "[!!]"
    print(f"   {status} {label:12s}: {count:3d} samples")

print(f"\nBalance Analysis:")
print(f"   Most common: {sorted_labels[0][0]} ({max_count} samples)")
print(f"   Least common: {sorted_labels[-1][0]} ({min_count} samples)")
print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 2.0:
    print(f"\n   [!!] WARNING: Significant class imbalance!")
    print(f"      The model might favor '{sorted_labels[0][0]}'")
else:
    print(f"\n   [OK] Classes are relatively balanced")
    print(f"      If model still predicts only one class, the problem is NOT class imbalance.")
    print(f"      Possible causes:")
    print(f"      1. Videos are too similar (same start, same person)")
    print(f"      2. Model architecture issue")
    print(f"      3. Training parameters need adjustment")

print("=" * 60)

