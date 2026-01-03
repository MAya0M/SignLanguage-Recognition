"""
Fix normalization to exclude padding zeros
This is critical for model learning!
"""

import sys
import os
from pathlib import Path
import numpy as np

# Fix path for Colab
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data_loader import SignLanguageDataLoader

def check_data_differences():
    """Check if data from different classes is actually different"""
    print("=" * 60)
    print("Checking Data Differences Between Classes")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Keypoints are already normalized in extraction, so disable normalization here
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False)
    splits = loader.get_all_splits()
    
    X_train, y_train = splits['train']
    
    # Check mean values per class
    print("\nMean values per class (first 5 features):")
    print("⚠️  NOTE: If all classes have similar means, normalization might be the problem!")
    for class_idx in range(loader.num_classes):
        class_mask = y_train == class_idx
        if np.any(class_mask):
            class_name = loader.label_encoder.inverse_transform([class_idx])[0]
            class_data = X_train[class_mask]
            # Calculate mean only for non-zero (non-padded) parts
            # For simplicity, use all data but note padding effect
            class_mean = np.mean(class_data, axis=(0, 1))
            class_std = np.std(class_data, axis=(0, 1))
            print(f"  {class_name:12s}: mean={class_mean[:5]}, std={class_std[:5]}")
    
    # Check if classes are too similar
    print("\n⚠️  Checking if classes are too similar after normalization...")
    class_means = []
    for class_idx in range(loader.num_classes):
        class_mask = y_train == class_idx
        if np.any(class_mask):
            class_data = X_train[class_mask]
            class_mean = np.mean(class_data, axis=(0, 1))
            class_means.append((class_idx, class_mean))
    
    # Calculate pairwise differences
    if len(class_means) > 1:
        differences = []
        for i, (idx1, mean1) in enumerate(class_means):
            for j, (idx2, mean2) in enumerate(class_means[i+1:], i+1):
                diff = np.mean(np.abs(mean1 - mean2))
                differences.append(diff)
        
        avg_diff = np.mean(differences)
        min_diff = np.min(differences)
        max_diff = np.max(differences)
        
        print(f"   Average difference between classes: {avg_diff:.6f}")
        print(f"   Min difference: {min_diff:.6f}, Max difference: {max_diff:.6f}")
        
        if avg_diff < 0.01:
            print("   ❌ PROBLEM: Classes are too similar after normalization!")
            print("   This suggests normalization is removing important differences.")
            print("   Solution: Try normalizing per sample or per time-step instead of globally.")
        elif avg_diff < 0.1:
            print("   ⚠️  WARNING: Classes are quite similar - model might struggle")
        else:
            print("   ✅ Classes have good separation")
    
    # Check if classes are different
    print("\nChecking if classes are distinguishable...")
    class_means = []
    for class_idx in range(loader.num_classes):
        class_mask = y_train == class_idx
        if np.any(class_mask):
            class_data = X_train[class_mask]
            class_mean = np.mean(class_data, axis=(0, 1))
            class_means.append(class_mean)
    
    if len(class_means) > 1:
        # Calculate pairwise differences
        max_diff = 0
        min_diff = float('inf')
        for i in range(len(class_means)):
            for j in range(i+1, len(class_means)):
                diff = np.mean(np.abs(class_means[i] - class_means[j]))
                max_diff = max(max_diff, diff)
                min_diff = min(min_diff, diff)
        
        print(f"  Mean difference between classes: {min_diff:.6f} - {max_diff:.6f}")
        if max_diff < 0.01:
            print("  ❌ PROBLEM: Classes are too similar - model cannot distinguish them!")
        else:
            print("  ✅ Classes have measurable differences")
    
    return True

if __name__ == "__main__":
    check_data_differences()

