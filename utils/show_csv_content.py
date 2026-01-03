"""Show CSV content to verify it's correct"""

import pandas as pd
from pathlib import Path
import sys
import os

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

csv_path = Path('Data/Labels/dataset.csv')
df = pd.read_csv(csv_path)

print("=" * 60)
print("CSV Content Verification")
print("=" * 60)
print(f"\nTotal entries: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")

print("\n" + "=" * 60)
print("Videos per word:")
print("=" * 60)
for label, count in df.groupby('label').size().sort_values(ascending=False).items():
    print(f"  {label:12s}: {count:3d} videos")

print("\n" + "=" * 60)
print("GoodBye files (should have 28, including 17-28):")
print("=" * 60)
goodbye = df[df['label'] == 'GOODBYE']
print(f"Total GoodBye entries: {len(goodbye)}")
print("\nAll GoodBye files:")
for path in sorted(goodbye['path'].tolist()):
    print(f"  {path}")

print("\n" + "=" * 60)
print("Files 17-28 specifically:")
print("=" * 60)
files_17_28 = [p for p in sorted(goodbye['path'].tolist()) 
               if any(str(i) in p for i in range(17, 29))]
for f in files_17_28:
    print(f"  {f}")

print(f"\n✅ Found {len(files_17_28)} files in range 17-28 (expected 12)")

