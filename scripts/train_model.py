"""
Training script for CNN + LSTM Sign Language Recognition Model
"""

import argparse
import os
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_loader import SignLanguageDataLoader
from scripts.model_cnn_lstm import build_cnn_lstm_model, compile_model


def train_model(
    csv_path: str,
    keypoints_dir: str,
    output_dir: str = "models",
    batch_size: int = 8,  # Smaller batch for better gradient updates
    epochs: int = 200,
    cnn_filters: int = 64,  # CNN filters
    lstm_units: int = 128,  # LSTM units
    num_cnn_layers: int = 2,  # Number of CNN layers
    dropout_rate: float = 0.3,  # Dropout rate
    learning_rate: float = 0.001,  # Learning rate
    patience: int = 10,
    validation_split: float = 0.0  # Not used, we have explicit val set
):
    """
    Train the CNN + LSTM model
    
    Args:
        csv_path: Path to CSV file with dataset info
        keypoints_dir: Directory containing keypoint .npy files
        output_dir: Directory to save model and training artifacts
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        cnn_filters: Number of filters in CNN layers
        lstm_units: Number of units in LSTM layer
        num_cnn_layers: Number of CNN layers
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        patience: Early stopping patience
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Sign Language Recognition - CNN + LSTM Model Training")
    print("="*60)
    print(f"CSV path: {csv_path}")
    print(f"Keypoints directory: {keypoints_dir}")
    print(f"Output directory: {run_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"CNN filters: {cnn_filters}")
    print(f"LSTM units: {lstm_units}")
    print(f"CNN layers: {num_cnn_layers}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Load data
    # NOTE: Keypoints are already normalized in extraction, so we disable normalization here
    # This prevents double normalization which can blur class differences
    # Smart frame sampling is enabled by default to skip similar start frames and focus on the gesture
    print("Loading data...")
    print("⚠️  Using data WITHOUT additional normalization (keypoints already normalized in extraction)")
    print("✅ Using smart frame sampling (skips similar start, focuses on gesture)")
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False, use_smart_sampling=True)
    splits = loader.get_all_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"\nData shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Check data quality
    print(f"\nData quality checks:")
    print(f"  Train samples: {len(X_train)} (expected: ~130)")
    print(f"  Val samples: {len(X_val)} (expected: ~40)")
    print(f"  Test samples: {len(X_test)} (expected: ~56)")
    
    # Check for NaN or Inf
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print(f"  ⚠️  WARNING: NaN or Inf values in training data!")
    else:
        print(f"  ✅ No NaN or Inf values in training data")
    
    # Check label distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\n  Label distribution in train:")
    for label_idx, count in zip(unique_train, counts_train):
        label_name = loader.label_encoder.inverse_transform([label_idx])[0]
        print(f"    {label_name}: {count} samples")
    
    # Calculate class weights to handle imbalance
    # This gives more weight to minority classes during training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\n  Class weights (to handle imbalance):")
    for label_idx, weight in class_weight_dict.items():
        label_name = loader.label_encoder.inverse_transform([label_idx])[0]
        print(f"    {label_name}: {weight:.3f}")
    
    # Check if imbalance is severe
    max_count = counts_train.max()
    min_count = counts_train.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    if imbalance_ratio > 2.0:
        print(f"\n  ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.2f}x)")
        print(f"     Using class weights to balance training")
    
    # Check data statistics
    print(f"\n  Data statistics:")
    print(f"    Train X - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
    print(f"    Train X - Min: {np.min(X_train):.4f}, Max: {np.max(X_train):.4f}")
    print(f"    Val X - Mean: {np.mean(X_val):.4f}, Std: {np.std(X_val):.4f}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = loader.num_classes
    
    print(f"\nBuilding model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    model = build_cnn_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        num_cnn_layers=num_cnn_layers
    )
    
    model = compile_model(model, learning_rate=learning_rate)
    
    print("\nModel architecture:")
    model.summary()
    
    # Save model architecture
    model_json = model.to_json()
    with open(run_dir / "model_architecture.json", "w") as f:
        f.write(model_json)
    
    # Save label mapping
    label_mapping = {
        'classes': loader.get_label_names(),
        'num_classes': loader.num_classes,
        'class_to_idx': {cls: idx for idx, cls in enumerate(loader.get_label_names())}
    }
    with open(run_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Callbacks - improved for better training
    callbacks = [
        ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        # Disable early stopping - let model train for all epochs
        # EarlyStopping(
        #     monitor='val_accuracy',
        #     patience=patience * 3,
        #     restore_best_weights=True,
        #     verbose=1,
        #     min_delta=0.0001,
        #     mode='max'
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Less aggressive reduction
            patience=10,  # More patience before reducing LR
            min_lr=1e-6,  # Higher minimum learning rate
            verbose=1,
            cooldown=2  # Shorter cooldown
        )
    ]
    
    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Shuffle data for better training
    # Create indices and shuffle
    train_indices = np.arange(len(X_train))
    np.random.seed(42)
    np.random.shuffle(train_indices)
    X_train_shuffled = X_train[train_indices]
    y_train_shuffled = y_train[train_indices]
    
    # Train model with progress tracking
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Training samples: {len(X_train_shuffled)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("="*60 + "\n")
    
    try:
        history = model.fit(
            X_train_shuffled, y_train_shuffled,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,  # Shuffle batches
            class_weight=class_weight_dict  # Use class weights to handle imbalance
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving current model state...")
        model.save(run_dir / "interrupted_model.keras")
        raise
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Saving current model state...")
        model.save(run_dir / "error_model.keras")
        raise
    
    # Save final model
    model.save(run_dir / "final_model.keras")
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy)
    }
    with open(run_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save training parameters
    training_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'cnn_filters': cnn_filters,
        'lstm_units': lstm_units,
        'num_cnn_layers': num_cnn_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'patience': patience,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'max_sequence_length': X_train.shape[1],
        'num_features': X_train.shape[2]
    }
    with open(run_dir / "training_params.json", "w") as f:
        json.dump(training_params, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model saved to: {run_dir / 'best_model.keras'}")
    print(f"All artifacts saved to: {run_dir}")
    print(f"{'='*60}")
    
    return model, history, loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN + LSTM model for sign language recognition")
    parser.add_argument("--csv", type=str, default="Data/Labels/dataset.csv",
                       help="Path to CSV file with dataset info")
    parser.add_argument("--keypoints-dir", type=str, default="Data/Keypoints/rawVideos",
                       help="Directory containing keypoint .npy files")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save model and training artifacts")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Maximum number of epochs")
    parser.add_argument("--cnn-filters", type=int, default=64,
                       help="Number of filters in CNN layers")
    parser.add_argument("--lstm-units", type=int, default=128,
                       help="Number of units in LSTM layer")
    parser.add_argument("--num-cnn-layers", type=int, default=2,
                       help="Number of CNN layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv,
        keypoints_dir=args.keypoints_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        cnn_filters=args.cnn_filters,
        lstm_units=args.lstm_units,
        num_cnn_layers=args.num_cnn_layers,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        patience=args.patience
    )

