"""
Script to analyze dataset distribution and identify issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter

def analyze_dataset(csv_path: str):
    """
    Analyze dataset and identify potential issues
    
    Args:
        csv_path: Path to dataset CSV file
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("📊 Dataset Analysis")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Number of classes: {df['label'].nunique()}")
    print(f"   Classes: {sorted(df['label'].unique())}")
    
    # Distribution by label
    print(f"\n📋 Distribution by Label:")
    label_counts = df.groupby('label').size().sort_values(ascending=False)
    
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        status = "⚠️" if count < 20 else "✅"
        if count == max_count:
            status = "🔴"  # Most common (might be the problem)
        print(f"   {status} {label:12s}: {count:3d} samples ({percentage:5.2f}%)")
    
    print(f"\n⚠️  Class Imbalance:")
    print(f"   Max: {max_count}, Min: {min_count}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2.0:
        print(f"   ⚠️  WARNING: Significant class imbalance detected!")
        print(f"      The model might favor the majority class.")
    
    # Distribution by split
    print(f"\n📊 Distribution by Split:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"   {split:6s}: {len(split_df):3d} samples")
        
        # Per label in split
        split_label_counts = split_df.groupby('label').size()
        for label, count in split_label_counts.items():
            print(f"      {label:12s}: {count:3d}")
    
    # Identify problematic classes
    print(f"\n🔍 Problem Analysis:")
    
    # Classes with too few samples
    few_samples = label_counts[label_counts < 20]
    if len(few_samples) > 0:
        print(f"   ⚠️  Classes with < 20 samples (need more data):")
        for label, count in few_samples.items():
            print(f"      - {label}: {count} samples")
    
    # Most common class
    most_common = label_counts.index[0]
    most_common_count = label_counts.iloc[0]
    print(f"\n   🔴 Most common class: {most_common} ({most_common_count} samples)")
    print(f"      If model always predicts '{most_common}', this is likely the cause!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if imbalance_ratio > 2.0:
        print(f"   1. ⚠️  Class imbalance detected!")
        print(f"      - Add more samples to minority classes")
        print(f"      - Use class weights during training")
        print(f"      - Consider data augmentation for minority classes")
    
    if len(few_samples) > 0:
        print(f"   2. ⚠️  Some classes have too few samples:")
        print(f"      - Aim for at least 30-50 samples per class")
        print(f"      - Add more videos for: {', '.join(few_samples.index)}")
    
    if most_common_count > len(df) * 0.4:
        print(f"   3. ⚠️  '{most_common}' is {most_common_count/len(df)*100:.1f}% of dataset!")
        print(f"      - Model will learn to predict '{most_common}' by default")
        print(f"      - Need to balance the dataset")
    
    # Check if all classes have samples in all splits
    print(f"\n🔍 Split Coverage:")
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        splits = label_df['split'].unique()
        missing_splits = set(['train', 'val', 'test']) - set(splits)
        if missing_splits:
            print(f"   ⚠️  {label}: Missing splits: {missing_splits}")
        else:
            train_count = len(label_df[label_df['split'] == 'train'])
            val_count = len(label_df[label_df['split'] == 'val'])
            test_count = len(label_df[label_df['split'] == 'test'])
            print(f"   ✅ {label}: train={train_count}, val={val_count}, test={test_count}")
    
    print("\n" + "=" * 60)
    
    return {
        'total_samples': len(df),
        'num_classes': df['label'].nunique(),
        'label_counts': label_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'most_common_class': most_common,
        'few_samples_classes': few_samples.to_dict() if len(few_samples) > 0 else {}
    }


if __name__ == "__main__":
    import sys
    
    csv_path = "Data/Labels/dataset.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    results = analyze_dataset(csv_path)
    
    # Save results to JSON
    output_json = Path(csv_path).parent / "dataset_analysis.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Analysis saved to: {output_json}")

