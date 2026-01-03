"""
Debug script to understand why model is not learning
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Fix path for Colab - handle both local and Colab environments
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tensorflow import keras
    from scripts.data_loader import SignLanguageDataLoader
    from scripts.model_gru import build_gru_model, compile_model
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Script directory: {script_dir}")
    print(f"   Project root: {project_root}")
    print(f"   Make sure you're running from the project root directory")
    sys.exit(1)

def debug_training():
    """Debug why model is not learning"""
    print("=" * 60)
    print("Debugging Model Training")
    print("=" * 60)
    
    csv_path = "Data/Labels/dataset.csv"
    keypoints_dir = "Data/Keypoints/rawVideos"
    
    # Check if files exist
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    if not Path(keypoints_dir).exists():
        print(f"❌ Keypoints directory not found: {keypoints_dir}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    # Load data
    print("\n1. Loading data...")
    try:
        loader = SignLanguageDataLoader(csv_path, keypoints_dir)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    splits = loader.get_all_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"   Train: {X_train.shape}, {y_train.shape}")
    print(f"   Val:   {X_val.shape}, {y_val.shape}")
    print(f"   Test:  {X_test.shape}, {y_test.shape}")
    
    # Check label distribution
    print(f"\n2. Label distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    print(f"   Train labels:")
    for label_idx, count in zip(unique_train, counts_train):
        label_name = loader.label_encoder.inverse_transform([label_idx])[0]
        print(f"      {label_name:12s}: {count:3d} samples")
    
    print(f"   Val labels:")
    for label_idx, count in zip(unique_val, counts_val):
        label_name = loader.label_encoder.inverse_transform([label_idx])[0]
        print(f"      {label_name:12s}: {count:3d} samples")
    
    # Check if labels are balanced
    print(f"\n3. Label balance check:")
    if len(set(counts_train)) == 1:
        print(f"   ✅ Labels are perfectly balanced")
    else:
        print(f"   ⚠️  Labels are imbalanced:")
        print(f"      Min: {min(counts_train)}, Max: {max(counts_train)}")
    
    # Check data statistics
    print(f"\n4. Data statistics:")
    print(f"   Train X - Mean: {np.mean(X_train):.6f}, Std: {np.std(X_train):.6f}")
    print(f"   Train X - Min: {np.min(X_train):.6f}, Max: {np.max(X_train):.6f}")
    print(f"   Train X - NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    
    # Check if data is all zeros or constant
    print(f"\n5. Data variation check:")
    sample_variances = np.var(X_train, axis=(1, 2))
    print(f"   Sample variance - Min: {np.min(sample_variances):.6f}, Max: {np.max(sample_variances):.6f}, Mean: {np.mean(sample_variances):.6f}")
    
    if np.max(sample_variances) < 1e-6:
        print(f"   ❌ PROBLEM: Data has no variation - all samples are the same!")
        return False
    
    # Check feature variation
    feature_variance = np.var(X_train, axis=(0, 1))
    zero_variance_features = np.sum(feature_variance < 1e-6)
    print(f"   Features with zero variance: {zero_variance_features}/{len(feature_variance)}")
    
    if zero_variance_features > len(feature_variance) * 0.5:
        print(f"   ⚠️  WARNING: More than 50% of features have no variation!")
    
    # Try a simple test - can model learn to distinguish at least 2 classes?
    print(f"\n6. Testing if model can learn simple task...")
    
    # Build a simple model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = loader.num_classes
    
    model = build_gru_model(
        input_shape=input_shape,
        num_classes=num_classes,
        gru_units=64,  # Smaller for testing
        dropout_rate=0.2,
        num_gru_layers=1
    )
    model = compile_model(model, learning_rate=0.001)
    
    # Train for just 5 epochs to see if loss decreases
    print(f"   Training for 5 epochs to check if loss decreases...")
    history = model.fit(
        X_train[:50], y_train[:50],  # Use small subset
        validation_data=(X_val[:20], y_val[:20]),
        epochs=5,
        verbose=0
    )
    
    initial_loss = history.history['loss'][0]
    final_loss = history.history['loss'][-1]
    loss_change = initial_loss - final_loss
    
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Loss change: {loss_change:.4f}")
    
    if loss_change < 0.01:
        print(f"   ❌ PROBLEM: Loss is not decreasing - model cannot learn!")
        print(f"   This suggests the data or model architecture has a fundamental issue.")
        return False
    else:
        print(f"   ✅ Loss is decreasing - model can learn")
    
    # Check predictions
    print(f"\n7. Checking initial predictions...")
    try:
        num_samples_to_check = min(10, len(y_val))
        y_pred_proba = model.predict(X_val[:num_samples_to_check], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print(f"   Predictions on {num_samples_to_check} val samples:")
        for i in range(num_samples_to_check):
            true_label = loader.label_encoder.inverse_transform([y_val[i]])[0]
            pred_label = loader.label_encoder.inverse_transform([y_pred[i]])[0]
            confidence = y_pred_proba[i][y_pred[i]]
            match = "✅" if y_pred[i] == y_val[i] else "❌"
            print(f"      {match} True: {true_label:12s}, Pred: {pred_label:12s}, Conf: {confidence:.3f}")
    except Exception as e:
        print(f"   ⚠️  Could not check predictions: {e}")
    
    return True

if __name__ == "__main__":
    success = debug_training()
    if not success:
        print("\n" + "=" * 60)
        print("❌ Found problems - model cannot learn!")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("✅ Basic checks passed")
        print("=" * 60)

