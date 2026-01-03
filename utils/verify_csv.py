"""Verify CSV contains all 28 videos per word"""

import sys
import os
import pandas as pd
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

csv_path = Path('Data/Labels/dataset.csv')
df = pd.read_csv(csv_path)

print("=" * 60)
print("CSV Verification")
print("=" * 60)
print(f"\nTotal entries: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")

print("\n" + "=" * 60)
print("Videos per word:")
print("=" * 60)

all_good = True
for label, count in df.groupby('label').size().sort_values(ascending=False).items():
    status = "✅" if count >= 28 else "❌"
    if count < 28:
        all_good = False
    print(f"{status} {label:12s}: {count:3d} videos")
    
    # Check for files 17-28
    label_df = df[df['label'] == label]
    files_17_28 = [p for p in label_df['path'].tolist() 
                   if any(str(i) in p for i in range(17, 29))]
    
    if count >= 28:
        print(f"   ✅ Has files 17-28: {len(files_17_28)} files")
        if len(files_17_28) < 12:
            print(f"   ⚠️  Warning: Expected 12 files (17-28), found {len(files_17_28)}")
    else:
        print(f"   ❌ Missing files 17-28!")

print("\n" + "=" * 60)
if all_good:
    print("✅ CSV is correct - all words have 28-29 videos!")
else:
    print("❌ CSV has issues - some words have less than 28 videos")
print("=" * 60)

