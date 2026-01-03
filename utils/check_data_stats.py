"""
Check statistics about the dataset
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_dataset_stats():
    """Check dataset statistics"""
    csv_path = Path('Data/Labels/dataset.csv')
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Total classes: {df['label'].nunique()}")
    
    print("\n" + "=" * 60)
    print("📹 Videos per word:")
    print("=" * 60)
    
    counts = df.groupby('label').size().sort_values(ascending=False)
    
    for label, count in counts.items():
        status = "✅" if count >= 20 else "⚠️" if count >= 10 else "❌"
        print(f"{status} {label:15s}: {count:3d} videos")
    
    print("\n" + "=" * 60)
    print("📈 Recommendations:")
    print("=" * 60)
    
    min_count = counts.min()
    max_count = counts.max()
    avg_count = counts.mean()
    
    print(f"Minimum: {min_count} videos per word")
    print(f"Maximum: {max_count} videos per word")
    print(f"Average: {avg_count:.1f} videos per word")
    
    print("\n💡 Recommendations:")
    if min_count < 10:
        print("  ❌ Some words have less than 10 videos - add more!")
    elif min_count < 20:
        print("  ⚠️  Some words have less than 20 videos - consider adding more")
    else:
        print("  ✅ All words have at least 20 videos - good!")
    
    if avg_count < 20:
        print(f"  💡 Aim for at least 20-30 videos per word for better accuracy")
    elif avg_count < 50:
        print(f"  💡 Consider adding more videos (50+) for production-quality model")
    else:
        print(f"  ✅ Excellent dataset size!")

if __name__ == '__main__':
    check_dataset_stats()

