"""
Debug script to check if data is loaded correctly
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_loader import SignLanguageDataLoader

def debug_data_loading():
    """Debug data loading to find issues"""
    print("=" * 60)
    print("Debugging Data Loading")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nCSV loaded: {len(df)} samples")
    print(f"Labels: {df['label'].unique()}")
    print(f"Split distribution:")
    print(df.groupby('split').size())
    
    # Try to load data
    print("\n" + "=" * 60)
    print("Loading data with SignLanguageDataLoader...")
    print("=" * 60)
    
    try:
        loader = SignLanguageDataLoader(csv_path, keypoints_dir)
        splits = loader.get_all_splits()
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        print(f"\nData shapes:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
        
        print(f"\nLabel distribution in train:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = loader.label_encoder.inverse_transform([label_idx])[0]
            print(f"  {label_name}: {count} samples")
        
        print(f"\nData statistics:")
        print(f"  Train X - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
        print(f"  Train X - Min: {np.min(X_train):.4f}, Max: {np.max(X_train):.4f}")
        print(f"  Train X - NaN count: {np.isnan(X_train).sum()}")
        print(f"  Train X - Inf count: {np.isinf(X_train).sum()}")
        
        print(f"\n  Val X - Mean: {np.mean(X_val):.4f}, Std: {np.std(X_val):.4f}")
        print(f"  Val X - Min: {np.min(X_val):.4f}, Max: {np.max(X_val):.4f}")
        print(f"  Val X - NaN count: {np.isnan(X_val).sum()}")
        print(f"  Val X - Inf count: {np.isinf(X_val).sum()}")
        
        # Check if data is normalized
        if hasattr(loader, 'mean') and hasattr(loader, 'std'):
            print(f"\nNormalization stats:")
            print(f"  Mean shape: {loader.mean.shape}")
            print(f"  Std shape: {loader.std.shape}")
            print(f"  Mean range: [{np.min(loader.mean):.4f}, {np.max(loader.mean):.4f}]")
            print(f"  Std range: [{np.min(loader.std):.4f}, {np.max(loader.std):.4f}]")
        
        # Check for class imbalance
        print(f"\nClass balance check:")
        train_counts = np.bincount(y_train)
        val_counts = np.bincount(y_val)
        for i, (train_count, val_count) in enumerate(zip(train_counts, val_counts)):
            label_name = loader.label_encoder.inverse_transform([i])[0]
            print(f"  {label_name}: Train={train_count}, Val={val_count}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loading()

