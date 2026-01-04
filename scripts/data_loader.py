"""
Data loader for sign language keypoints dataset
Loads data from CSV file and prepares it for GRU model training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, List, Dict
import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_keypoints import smart_frame_sampling


class SignLanguageDataLoader:
    """Data loader for sign language keypoints dataset"""
    
    def __init__(self, csv_path: str, keypoints_base_dir: str = "Data/Keypoints/rawVideos", 
                 normalize: bool = False, use_smart_sampling: bool = True, target_frames: int = 96):
        """
        Initialize data loader
        
        Args:
            csv_path: Path to CSV file with columns: path, label, split
            keypoints_base_dir: Base directory where keypoint files are stored
            normalize: Whether to apply z-score normalization (default: False, since keypoints are already normalized in extraction)
            use_smart_sampling: Whether to use smart frame sampling (skip start, focus on relevant part)
            target_frames: Target number of frames for smart sampling (default: 96)
        """
        self.csv_path = Path(csv_path)
        self.keypoints_base_dir = Path(keypoints_base_dir)
        self.label_encoder = LabelEncoder()
        self.max_length = None
        self.num_classes = None
        self.normalize = normalize  # Store normalization flag
        self.use_smart_sampling = use_smart_sampling
        self.target_frames = target_frames
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Encode labels
        all_labels = self.df['label'].unique()
        self.label_encoder.fit(all_labels)
        self.num_classes = len(all_labels)
        
        print(f"Found {len(self.df)} samples")
        print(f"Labels: {list(all_labels)}")
        print(f"Number of classes: {self.num_classes}")
        if not normalize:
            print("⚠️  Normalization DISABLED (keypoints already normalized in extraction)")
        else:
            print("✅ Normalization ENABLED")
    
    def load_keypoints(self, relative_path: str) -> np.ndarray:
        """
        Load keypoints from .npy file
        
        Args:
            relative_path: Relative path from CSV (e.g., "keypoints/HELLO/hello_01.npy")
            
        Returns:
            numpy array with shape (num_frames, features)
            Features are flattened from (num_frames, 2, 21, 3) -> (num_frames, 126)
        """
        # Extract actual path - handle both formats:
        # 1. "keypoints/LABEL/file.npy" (old format)
        # 2. "LABEL/file.npy" (new format)
        if relative_path.startswith("keypoints/"):
            # Remove "keypoints/" prefix
            relative_path = relative_path.replace("keypoints/", "", 1)
        actual_path = self.keypoints_base_dir / relative_path
        
        # Load keypoints array (shape: num_frames, 2, 21, 3)
        keypoints = np.load(actual_path)
        
        # Apply smart frame sampling if enabled
        # This skips the similar start frames and focuses on the actual gesture
        if self.use_smart_sampling:
            keypoints = smart_frame_sampling(keypoints, target_frames=self.target_frames, skip_start_ratio=0.2)
        
        # Flatten to (num_frames, features) where features = 2 * 21 * 3 = 126
        # Reshape from (num_frames, 2, 21, 3) to (num_frames, 126)
        num_frames = keypoints.shape[0]
        keypoints_flat = keypoints.reshape(num_frames, -1)
        
        return keypoints_flat
    
    def get_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific split (train, val, or test)
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Tuple of (X, y) where:
            - X: padded sequences of keypoints (num_samples, max_length, features)
            - y: encoded labels (num_samples,)
        """
        split_df = self.df[self.df['split'] == split].copy()
        
        X = []
        y = []
        
        print(f"\nLoading {split} data...")
        for idx, row in split_df.iterrows():
            try:
                keypoints = self.load_keypoints(row['path'])
                X.append(keypoints)
                
                # Encode label
                label_encoded = self.label_encoder.transform([row['label']])[0]
                y.append(label_encoded)
            except Exception as e:
                print(f"Error loading {row['path']}: {e}")
                continue
        
        # Convert y to numpy array
        y = np.array(y)
        
        # Pad sequences to same length
        if self.max_length is None:
            # Use the maximum length in training set
            if split == 'train':
                self.max_length = max(len(seq) for seq in X)
                print(f"Max sequence length (from {split}): {self.max_length}")
            else:
                # Should be set from training
                if hasattr(self, '_max_length'):
                    self.max_length = self._max_length
                else:
                    raise ValueError("max_length not set. Load training data first.")
        else:
            # Store for use in validation/test
            if split == 'train':
                self._max_length = self.max_length
        
        # Pad sequences using numpy (pad_sequences doesn't work well with object arrays)
        num_samples = len(X)
        num_features = X[0].shape[1] if len(X) > 0 else 126
        X_padded = np.zeros((num_samples, self.max_length, num_features), dtype='float32')
        
        # Store actual sequence lengths for normalization
        seq_lengths = []
        for i, seq in enumerate(X):
            seq_length = min(len(seq), self.max_length)
            X_padded[i, :seq_length, :] = seq[:seq_length]
            seq_lengths.append(seq_length)
        
        # Normalize data (zero mean, unit variance) for better training
        # CRITICAL: Exclude padding zeros from normalization statistics!
        # NOTE: Keypoints are already normalized in extraction, so normalization here is optional
        if self.normalize and split == 'train':
            # Calculate mean and std only from non-zero (non-padded) data
            # Create a mask: True for actual data, False for padding
            # We'll use a simple heuristic: if a feature is all zeros across time, it's likely padding
            # But better: track actual sequence lengths and only normalize those parts
            
            # Method: Calculate stats only from non-zero regions
            # For each sample, find where actual data ends (where all features become zero)
            # Actually, simpler: use variance to detect padding - padding has zero variance
            # But even simpler: normalize per feature, ignoring zeros
            
            # Calculate mean and std per feature, but only from non-zero values
            # This is more complex, so let's use a simpler approach:
            # Normalize per feature across all non-padded time steps
            
            # Use stored sequence lengths
            max_actual_length = max(seq_lengths) if seq_lengths else self.max_length
            
            # Calculate stats only from actual data (not padding)
            # For each sample, only use data up to its actual length
            non_padded_data = []
            for i, seq_len in enumerate(seq_lengths):
                non_padded_data.append(X_padded[i, :seq_len, :])
            
            if non_padded_data:
                # Concatenate all non-padded data
                all_non_padded = np.concatenate(non_padded_data, axis=0)  # Shape: (total_frames, features)
                
                # Calculate mean and std per feature
                self.mean = np.mean(all_non_padded, axis=0, keepdims=True)  # Shape: (1, num_features)
                self.std = np.std(all_non_padded, axis=0, keepdims=True) + 1e-8  # Shape: (1, num_features)
                
                # Reshape for broadcasting: (1, 1, num_features)
                self.mean = self.mean.reshape(1, 1, -1)
                self.std = self.std.reshape(1, 1, -1)
                
                # Normalize only non-padded parts (keep padding as zeros)
                # This is critical - padding should stay as zeros, not be normalized
                for i, seq_len in enumerate(seq_lengths):
                    X_padded[i, :seq_len, :] = (X_padded[i, :seq_len, :] - self.mean[0, 0, :]) / self.std[0, 0, :]
                # Padding remains as zeros (already zero, so no change needed)
                
                # Store normalization stats for val/test
                self._normalization_mean = self.mean
                self._normalization_std = self.std
            else:
                # Fallback to old method if no data
                self.mean = np.mean(X_padded, axis=(0, 1), keepdims=True)
                self.std = np.std(X_padded, axis=(0, 1), keepdims=True) + 1e-8
                X_padded = (X_padded - self.mean) / self.std
                self._normalization_mean = self.mean
                self._normalization_std = self.std
        elif self.normalize:
            # Use training statistics for validation/test
            if hasattr(self, '_normalization_mean') and hasattr(self, '_normalization_std'):
                # Normalize only non-padded parts (keep padding as zeros)
                for i, seq_len in enumerate(seq_lengths):
                    X_padded[i, :seq_len, :] = (X_padded[i, :seq_len, :] - self._normalization_mean[0, 0, :]) / self._normalization_std[0, 0, :]
                # Padding remains as zeros
            else:
                # Fallback: normalize with current data stats (shouldn't happen)
                print("⚠️  WARNING: Using fallback normalization for validation/test")
                # Calculate stats from non-padded data
                non_padded_data = []
                for i, seq_len in enumerate(seq_lengths):
                    non_padded_data.append(X_padded[i, :seq_len, :])
                if non_padded_data:
                    all_non_padded = np.concatenate(non_padded_data, axis=0)
                    mean = np.mean(all_non_padded, axis=0, keepdims=True).reshape(1, 1, -1)
                    std = np.std(all_non_padded, axis=0, keepdims=True).reshape(1, 1, -1) + 1e-8
                    for i, seq_len in enumerate(seq_lengths):
                        X_padded[i, :seq_len, :] = (X_padded[i, :seq_len, :] - mean[0, 0, :]) / std[0, 0, :]
        
        print(f"Loaded {len(X_padded)} samples")
        print(f"X shape: {X_padded.shape}")
        print(f"y shape: {y.shape}")
        
        return X_padded, y
    
    def get_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all splits (train, val, test)
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' and values (X, y) tuples
        """
        splits = {}
        
        # Load train first to set max_length
        splits['train'] = self.get_split_data('train')
        self._max_length = self.max_length  # Store for validation/test
        
        # Load validation and test
        splits['val'] = self.get_split_data('val')
        splits['test'] = self.get_split_data('test')
        
        return splits
    
    def get_label_names(self) -> List[str]:
        """Get list of label names in order"""
        return list(self.label_encoder.classes_)
    
    def decode_label(self, encoded_label: int) -> str:
        """Decode encoded label back to string"""
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def encode_label(self, label: str) -> int:
        """Encode label string to integer"""
        return self.label_encoder.transform([label])[0]


if __name__ == "__main__":
    # Test data loader
    loader = SignLanguageDataLoader("Data/Labels/dataset.csv")
    
    print("\n" + "="*60)
    print("Testing data loader...")
    print("="*60)
    
    splits = loader.get_all_splits()
    
    print("\nSummary:")
    for split_name, (X, y) in splits.items():
        print(f"{split_name}: X.shape={X.shape}, y.shape={y.shape}")
    
    print(f"\nLabel names: {loader.get_label_names()}")
    print(f"Number of classes: {loader.num_classes}")

