"""
Try training without normalization in data_loader since keypoints are already normalized
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
    project_root = Path(os.getcwd())
    if project_root.name == 'scripts':
        project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data_loader import SignLanguageDataLoader

def check_without_normalization():
    """Check data without normalization in data_loader"""
    print("=" * 60)
    print("Checking Data WITHOUT Additional Normalization")
    print("(Keypoints are already normalized in extraction)")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Keypoints are already normalized in extraction, so disable normalization here
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False)
    
    # Manually load data without normalization
    split_df = loader.df[loader.df['split'] == 'train'].copy()
    
    X = []
    y = []
    
    print("\nLoading data (no additional normalization)...")
    for idx, row in split_df.iterrows():
        try:
            keypoints = loader.load_keypoints(row['path'])
            X.append(keypoints)
            label_encoded = loader.label_encoder.transform([row['label']])[0]
            y.append(label_encoded)
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            continue
    
    y = np.array(y)
    
    # Pad sequences
    max_length = max(len(seq) for seq in X)
    num_features = X[0].shape[1] if len(X) > 0 else 126
    X_padded = np.zeros((len(X), max_length, num_features), dtype='float32')
    
    seq_lengths = []
    for i, seq in enumerate(X):
        seq_length = min(len(seq), max_length)
        X_padded[i, :seq_length, :] = seq[:seq_length]
        seq_lengths.append(seq_length)
    
    print(f"Data shape: {X_padded.shape}")
    print(f"Data - Mean: {np.mean(X_padded):.6f}, Std: {np.std(X_padded):.6f}")
    print(f"Data - Min: {np.min(X_padded):.6f}, Max: {np.max(X_padded):.6f}")
    
    # Check mean values per class
    print("\nMean values per class (first 5 features) WITHOUT additional normalization:")
    for class_idx in range(loader.num_classes):
        class_mask = y == class_idx
        if np.any(class_mask):
            class_name = loader.label_encoder.inverse_transform([class_idx])[0]
            class_data = X_padded[class_mask]
            # Calculate mean only for non-padded parts
            class_means = []
            for i, mask in enumerate(class_mask):
                if mask:
                    seq_len = seq_lengths[i]
                    class_means.append(np.mean(X_padded[i, :seq_len, :], axis=0))
            
            if class_means:
                overall_mean = np.mean(class_means, axis=0)
                overall_std = np.std(class_means, axis=0)
                print(f"  {class_name:12s}: mean={overall_mean[:5]}, std={overall_std[:5]}")
    
    # Check if classes are different
    print("\nChecking if classes are different WITHOUT additional normalization...")
    class_means = []
    for class_idx in range(loader.num_classes):
        class_mask = y == class_idx
        if np.any(class_mask):
            class_data = X_padded[class_mask]
            means = []
            for i, mask in enumerate(class_mask):
                if mask:
                    seq_len = seq_lengths[i]
                    means.append(np.mean(X_padded[i, :seq_len, :], axis=0))
            
            if means:
                class_mean = np.mean(means, axis=0)
                class_means.append(class_mean)
    
    if len(class_means) > 1:
        differences = []
        for i in range(len(class_means)):
            for j in range(i+1, len(class_means)):
                diff = np.mean(np.abs(class_means[i] - class_means[j]))
                differences.append(diff)
        
        avg_diff = np.mean(differences)
        min_diff = np.min(differences)
        max_diff = np.max(differences)
        
        print(f"  Average difference: {avg_diff:.6f}")
        print(f"  Min difference: {min_diff:.6f}, Max difference: {max_diff:.6f}")
        
        if avg_diff < 0.01:
            print("  ❌ PROBLEM: Classes are too similar!")
        elif avg_diff < 0.1:
            print("  ⚠️  WARNING: Classes are somewhat similar")
        else:
            print("  ✅ Classes have good differences")
        
        print("\n💡 RECOMMENDATION:")
        if avg_diff > 0.1:
            print("  ✅ Data looks good! Try training WITHOUT normalization in data_loader")
            print("  (Keypoints are already normalized in extraction)")
        else:
            print("  ⚠️  Data classes are similar. May need:")
            print("     - More training data")
            print("     - Different normalization strategy")
            print("     - Feature engineering")
    
    return True

if __name__ == "__main__":
    check_without_normalization()

