"""
Check if training data is correctly loaded
Simulates what happens in Colab
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_data_loading():
    """Check if data can be loaded correctly"""
    print("=" * 60)
    print("Checking Training Data Loading")
    print("=" * 60)
    
    csv_path = Path("Data/Labels/dataset.csv")
    keypoints_dir = Path("Data/Keypoints/rawVideos")
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    
    if not keypoints_dir.exists():
        print(f"❌ Keypoints directory not found: {keypoints_dir}")
        return
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\n✅ CSV loaded: {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    
    # Check splits
    print(f"\n📊 Split distribution:")
    for split in ['train', 'val', 'test']:
        count = len(df[df['split'] == split])
        print(f"   {split:6s}: {count:3d} samples")
    
    # Check labels
    labels = df['label'].unique()
    print(f"\n🏷️  Labels: {len(labels)} classes")
    for label in sorted(labels):
        count = len(df[df['label'] == label])
        print(f"   {label:12s}: {count:3d} samples")
    
    # Try to load some files
    print(f"\n🔍 Testing file loading...")
    errors = []
    loaded = 0
    
    for idx, row in df.head(20).iterrows():  # Test first 20 files
        path_in_csv = row['path']
        
        # Simulate what data_loader does
        if path_in_csv.startswith("keypoints/"):
            relative_path = path_in_csv.replace("keypoints/", "", 1)
        else:
            relative_path = path_in_csv
        
        actual_path = keypoints_dir / relative_path
        
        if actual_path.exists():
            try:
                keypoints = np.load(actual_path)
                loaded += 1
                if loaded <= 3:  # Show first 3 successful loads
                    print(f"   ✅ {path_in_csv}")
                    print(f"      → {actual_path}")
                    print(f"      Shape: {keypoints.shape}")
            except Exception as e:
                errors.append((path_in_csv, str(e)))
                print(f"   ❌ {path_in_csv}: {e}")
        else:
            errors.append((path_in_csv, "File not found"))
            print(f"   ❌ {path_in_csv}")
            print(f"      → {actual_path} (NOT FOUND)")
    
    print(f"\n📈 Summary:")
    print(f"   Successfully loaded: {loaded}/20 test files")
    print(f"   Errors: {len(errors)}")
    
    if errors:
        print(f"\n⚠️  Sample errors:")
        for path, error in errors[:5]:
            print(f"   {path}: {error}")
    
    # Check if all files exist
    print(f"\n🔍 Checking all files...")
    missing = []
    for idx, row in df.iterrows():
        path_in_csv = row['path']
        
        if path_in_csv.startswith("keypoints/"):
            relative_path = path_in_csv.replace("keypoints/", "", 1)
        else:
            relative_path = path_in_csv
        
        actual_path = keypoints_dir / relative_path
        
        if not actual_path.exists():
            missing.append(path_in_csv)
    
    if missing:
        print(f"   ❌ {len(missing)} files missing!")
        print(f"   Sample missing files:")
        for path in missing[:10]:
            print(f"      {path}")
    else:
        print(f"   ✅ All {len(df)} files exist!")

if __name__ == "__main__":
    check_data_loading()

