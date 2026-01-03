"""
Check raw data (before normalization) to see if classes are different
"""

import sys
import os
from pathlib import Path
import numpy as np

# Fix path for Colab
try:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
except NameError:
    # __file__ not defined in Jupyter/Colab - use current working directory
    project_root = Path(os.getcwd())
    if project_root.name == 'scripts':
        project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data_loader import SignLanguageDataLoader

def check_raw_data():
    """Check raw data before normalization"""
    print("=" * 60)
    print("Checking Raw Data (Before Normalization)")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Keypoints are already normalized in extraction, so disable normalization here
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False)
    
    # Load raw data without normalization
    split_df = loader.df[loader.df['split'] == 'train'].copy()
    
    X_raw = []
    y_raw = []
    
    print("\nLoading raw data (no normalization)...")
    for idx, row in split_df.iterrows():
        try:
            keypoints = loader.load_keypoints(row['path'])
            X_raw.append(keypoints)
            label_encoded = loader.label_encoder.transform([row['label']])[0]
            y_raw.append(label_encoded)
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            continue
    
    y_raw = np.array(y_raw)
    
    # Pad sequences
    max_length = max(len(seq) for seq in X_raw)
    num_features = X_raw[0].shape[1] if len(X_raw) > 0 else 126
    X_padded = np.zeros((len(X_raw), max_length, num_features), dtype='float32')
    
    for i, seq in enumerate(X_raw):
        seq_length = min(len(seq), max_length)
        X_padded[i, :seq_length, :] = seq[:seq_length]
    
    print(f"Raw data shape: {X_padded.shape}")
    print(f"Raw data - Mean: {np.mean(X_padded):.6f}, Std: {np.std(X_padded):.6f}")
    print(f"Raw data - Min: {np.min(X_padded):.6f}, Max: {np.max(X_padded):.6f}")
    
    # Check mean values per class (raw data)
    print("\nRaw mean values per class (first 5 features):")
    for class_idx in range(loader.num_classes):
        class_mask = y_raw == class_idx
        if np.any(class_mask):
            class_name = loader.label_encoder.inverse_transform([class_idx])[0]
            class_data = X_padded[class_mask]
            # Calculate mean only for non-padded parts
            class_means = []
            for i, mask in enumerate(class_mask):
                if mask:
                    seq_len = len(X_raw[i])
                    class_means.append(np.mean(X_padded[i, :seq_len, :], axis=0))
            
            if class_means:
                overall_mean = np.mean(class_means, axis=0)
                print(f"  {class_name:12s}: {overall_mean[:5]}")
    
    # Check if classes are different in raw data
    print("\nChecking if classes are different in RAW data...")
    class_means_raw = []
    for class_idx in range(loader.num_classes):
        class_mask = y_raw == class_idx
        if np.any(class_mask):
            class_data = X_padded[class_mask]
            # Calculate mean only for non-padded parts
            means = []
            for i, mask in enumerate(class_mask):
                if mask:
                    seq_len = len(X_raw[i])
                    means.append(np.mean(X_padded[i, :seq_len, :], axis=0))
            
            if means:
                class_mean = np.mean(means, axis=0)
                class_means_raw.append(class_mean)
    
    if len(class_means_raw) > 1:
        differences = []
        for i in range(len(class_means_raw)):
            for j in range(i+1, len(class_means_raw)):
                diff = np.mean(np.abs(class_means_raw[i] - class_means_raw[j]))
                differences.append(diff)
        
        avg_diff = np.mean(differences)
        min_diff = np.min(differences)
        max_diff = np.max(differences)
        
        print(f"  Average difference: {avg_diff:.6f}")
        print(f"  Min difference: {min_diff:.6f}, Max difference: {max_diff:.6f}")
        
        if avg_diff < 0.1:
            print("  ❌ PROBLEM: Raw data classes are too similar!")
            print("  This means the keypoints themselves are not different enough.")
            print("  Solution: Check keypoint extraction - maybe all videos are being processed the same way?")
        elif avg_diff < 1.0:
            print("  ⚠️  WARNING: Raw data classes are somewhat similar")
        else:
            print("  ✅ Raw data classes have good differences")
    
    return True

if __name__ == "__main__":
    check_raw_data()

