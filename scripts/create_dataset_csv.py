import os
import csv
import re
from pathlib import Path
import random


def get_all_npy_files(keypoints_dir):
    """
    Collect all .npy files from the keypoints directory, organized by label
    
    Args:
        keypoints_dir: Path to the Keypoints directory (e.g., Data/Keypoints/rawVideos)
    
    Returns:
        Dictionary mapping label names to lists of .npy file paths
    """
    keypoints_path = Path(keypoints_dir)
    labels_dict = {}
    
    if not keypoints_path.exists():
        print(f"Warning: Directory {keypoints_path} does not exist")
        return labels_dict
    
    # Iterate through all subdirectories (each represents a label)
    for label_dir in sorted(keypoints_path.iterdir()):
        if label_dir.is_dir():
            label_name = label_dir.name
            npy_files = sorted([f for f in label_dir.glob("*.npy")])
            
            if npy_files:
                labels_dict[label_name] = npy_files
                print(f"Found {len(npy_files)} files in {label_name}")
    
    return labels_dict


def split_files(files, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split files into train, validation, and test sets
    
    Args:
        files: List of file paths
        train_ratio: Ratio for training set (default: 0.6)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Shuffle files with seed for reproducibility
    files = list(files)
    random.seed(seed)
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    total = len(shuffled_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = shuffled_files[:train_end]
    val_files = shuffled_files[train_end:val_end]
    test_files = shuffled_files[val_end:]
    
    return train_files, val_files, test_files


def create_csv_dataset(keypoints_dir, output_csv, base_path="keypoints", 
                       train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create a CSV file with dataset information (path, label, split)
    
    Args:
        keypoints_dir: Path to the Keypoints directory (e.g., Data/Keypoints/rawVideos)
        output_csv: Path to output CSV file
        base_path: Base path prefix for files in CSV (default: "keypoints")
        train_ratio: Ratio for training set (default: 0.6)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.2)
        seed: Random seed for reproducibility
    """
    keypoints_path = Path(keypoints_dir)
    
    # Get all files organized by label
    labels_dict = get_all_npy_files(keypoints_dir)
    
    if not labels_dict:
        print("No files found. Exiting.")
        return
    
    # Prepare CSV data
    csv_rows = []
    
    for label_name, files in sorted(labels_dict.items()):
        # Normalize label name to uppercase
        label_upper = label_name.upper()
        
        # Split files into train/val/test
        train_files, val_files, test_files = split_files(
            files, train_ratio, val_ratio, test_ratio, seed
        )
        
        # Helper function to create relative path using actual filename
        def create_relative_path(file_path):
            # Get the relative path from keypoints_dir and convert to string with forward slashes
            relative_to_keypoints = file_path.relative_to(keypoints_path)
            # Use actual filename as-is (not normalized) to match real file names
            relative_str = str(relative_to_keypoints).replace('\\', '/')
            # Add base_path prefix if specified
            if base_path:
                return f"{base_path}/{relative_str}"
            return relative_str
        
        # Add train files
        for file_path in train_files:
            relative_path = create_relative_path(file_path)
            csv_rows.append({
                'path': relative_path,
                'label': label_upper,
                'split': 'train'
            })
        
        # Add val files
        for file_path in val_files:
            relative_path = create_relative_path(file_path)
            csv_rows.append({
                'path': relative_path,
                'label': label_upper,
                'split': 'val'
            })
        
        # Add test files
        for file_path in test_files:
            relative_path = create_relative_path(file_path)
            csv_rows.append({
                'path': relative_path,
                'label': label_upper,
                'split': 'test'
            })
        
        print(f"Label {label_upper}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Sort CSV rows: first by label, then by split (train, val, test), then by path
    split_order = {'train': 0, 'val': 1, 'test': 2}
    csv_rows_sorted = sorted(csv_rows, key=lambda x: (x['label'], split_order[x['split']], x['path']))
    
    # Write CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['path', 'label', 'split']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_rows_sorted:
            writer.writerow(row)
    
    print(f"\nCSV file created: {output_csv}")
    print(f"Total entries: {len(csv_rows_sorted)}")
    
    # Print statistics
    train_count = sum(1 for row in csv_rows_sorted if row['split'] == 'train')
    val_count = sum(1 for row in csv_rows_sorted if row['split'] == 'val')
    test_count = sum(1 for row in csv_rows_sorted if row['split'] == 'test')
    
    print(f"\nDataset split:")
    print(f"  Train: {train_count} ({train_count/len(csv_rows_sorted)*100:.1f}%)")
    print(f"  Val:   {val_count} ({val_count/len(csv_rows_sorted)*100:.1f}%)")
    print(f"  Test:  {test_count} ({test_count/len(csv_rows_sorted)*100:.1f}%)")


if __name__ == "__main__":
    import sys
    
    # Set up paths
    base_dir = Path("Data")
    keypoints_dir = base_dir / "Keypoints" / "rawVideos"
    labels_dir = base_dir / "Labels"
    
    # Create Labels directory if it doesn't exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Default output filename
    output_filename = "dataset.csv"
    
    # Check for command line arguments (optional filename)
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        # Ensure it has .csv extension
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
    
    # Output CSV will always be in Data/Labels directory
    output_csv = labels_dir / output_filename
    
    # Check if keypoints directory exists
    if not keypoints_dir.exists():
        print(f"Error: Keypoints directory not found: {keypoints_dir}")
        print("Please run extract_keypoints.py first to generate keypoint files.")
        sys.exit(1)
    
    print("=" * 60)
    print("Creating dataset CSV file")
    print("=" * 60)
    print(f"Keypoints directory: {keypoints_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Split ratios: 60% train, 20% val, 20% test")
    print("=" * 60 + "\n")
    
    # Create CSV
    create_csv_dataset(keypoints_dir, output_csv, base_path="keypoints",
                      train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

