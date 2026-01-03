"""
Quick script to check if dataset is loaded correctly
Run this in Colab before training to verify data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_loader import SignLanguageDataLoader

def check_dataset():
    """Check if dataset is correct"""
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Check CSV
    print("\n1. Checking CSV file...")
    df = pd.read_csv(csv_path)
    print(f"   Total samples in CSV: {len(df)}")
    print(f"   Expected: 226 samples (with new videos)")
    
    if len(df) < 200:
        print(f"\n   ⚠️  PROBLEM: Only {len(df)} samples!")
        print(f"   CSV is NOT updated with new videos!")
        print(f"   Expected 226 samples, got {len(df)}")
        return False
    else:
        print(f"   ✅ CSV has {len(df)} samples - looks good!")
    
    # Check labels
    print(f"\n2. Checking labels...")
    labels = df['label'].unique()
    print(f"   Number of classes: {len(labels)}")
    print(f"   Labels: {list(labels)}")
    
    print(f"\n   Samples per label:")
    for label, count in df.groupby('label').size().sort_values(ascending=False).items():
        if count < 25:
            print(f"      {label:12s}: {count:3d} samples ⚠️  (should be 28-29)")
        else:
            print(f"      {label:12s}: {count:3d} samples ✅")
    
    # Check splits
    print(f"\n3. Checking splits...")
    for split, count in df.groupby('split').size().items():
        print(f"   {split:6s}: {count:3d} samples")
    
    # Try loading data
    print(f"\n4. Testing data loading...")
    try:
        loader = SignLanguageDataLoader(csv_path, keypoints_dir)
        splits = loader.get_all_splits()
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        print(f"   ✅ Data loaded successfully!")
        print(f"\n   Data shapes:")
        print(f"      Train: {X_train.shape[0]} samples")
        print(f"      Val:   {X_val.shape[0]} samples")
        print(f"      Test:  {X_test.shape[0]} samples")
        
        # Check label distribution
        print(f"\n   Label distribution in train:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = loader.label_encoder.inverse_transform([label_idx])[0]
            if count < 15:
                print(f"      {label_name:12s}: {count:3d} samples ⚠️  (imbalanced!)")
            else:
                print(f"      {label_name:12s}: {count:3d} samples ✅")
        
        # Check data quality
        print(f"\n5. Checking data quality...")
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print(f"   ⚠️  WARNING: NaN or Inf values in data!")
        else:
            print(f"   ✅ No NaN or Inf values")
        
        print(f"\n   Data statistics:")
        print(f"      Train X - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
        print(f"      Train X - Min: {np.min(X_train):.4f}, Max: {np.max(X_train):.4f}")
        
        # Check if data is normalized
        if abs(np.mean(X_train)) < 0.1 and abs(np.std(X_train) - 1.0) < 0.1:
            print(f"   ✅ Data appears normalized")
        else:
            print(f"   ⚠️  Data might not be normalized correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_dataset()
    if not success:
        print("\n" + "=" * 60)
        print("❌ Dataset has problems - fix before training!")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("✅ Dataset looks good - ready for training!")
        print("=" * 60)

