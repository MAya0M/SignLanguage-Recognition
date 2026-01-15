"""
Visualize Model Results - Create graphs similar to Actual vs Predicted style
Shows training history, confusion matrix, and per-class accuracy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_loader import SignLanguageDataLoader


def load_training_history(run_dir):
    """Load training history from JSON"""
    history_path = Path(run_dir) / "training_history.json"
    if not history_path.exists():
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def load_test_results(run_dir):
    """Load test results from JSON"""
    results_path = Path(run_dir) / "test_results.json"
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def load_label_mapping(run_dir):
    """Load label mapping from JSON"""
    mapping_path = Path(run_dir) / "label_mapping.json"
    if not mapping_path.exists():
        return None
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping


def plot_training_history(run_dir, output_path=None):
    """Plot training history (accuracy and loss over epochs)"""
    history = load_training_history(run_dir)
    if not history:
        print("No training history found!")
        return
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax1.plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax1.plot(epochs, history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy - Train vs Validation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add best validation accuracy
    if 'val_accuracy' in history:
        best_val_acc = max(history['val_accuracy'])
        best_epoch = history['val_accuracy'].index(best_val_acc) + 1
        ax1.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(best_epoch, best_val_acc, f'Best: {best_val_acc:.3f}', 
                fontsize=9, ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Loss
    ax2.plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax2.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss - Train vs Validation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add best validation loss
    if 'val_loss' in history:
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        ax2.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax2.text(best_epoch, best_val_loss, f'Best: {best_val_loss:.3f}', 
                fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_test_predictions(run_dir, csv_path, keypoints_dir, output_path=None):
    """Plot actual vs predicted for test set (similar to the example image)"""
    # Load model
    model_path = Path(run_dir) / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        # Try loading with custom_objects to handle initializer issues
        model = keras.models.load_model(str(model_path), compile=False)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not load model with standard method: {e}")
        print("Trying alternative loading method with custom objects...")
        try:
            from tensorflow.keras.initializers import Orthogonal
            custom_objects = {'Orthogonal': Orthogonal}
            model = keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
        except Exception as e2:
            print(f"Error loading model: {e2}")
            print("Note: This might be a Keras version compatibility issue.")
            print("The model was likely saved with a different Keras version.")
            print("Training history plot was created successfully!")
            print("Skipping test predictions plot...")
            return
    
    # Load data
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False, use_smart_sampling=True)
    X_test, y_test = loader.get_split_data('test')
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get label names
    label_mapping = load_label_mapping(run_dir)
    if label_mapping:
        label_names = label_mapping['classes']
    else:
        label_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual vs predicted
    x_positions = np.arange(len(y_test))
    
    # Plot actual (blue line)
    ax.plot(x_positions, y_test, 'b-', label='Actual', linewidth=2, alpha=0.7)
    
    # Plot predicted (orange dashed line)
    ax.plot(x_positions, y_pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    
    # Calculate metrics
    accuracy = np.mean(y_test == y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Add metrics to plot
    metrics_text = f'Test Accuracy: {accuracy:.3f}\nTest MAE: {mae:.3f}\nTest RMSE: {rmse:.3f}'
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Class Label', fontsize=12)
    ax.set_title('Sign Language Recognition - Test: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show class names
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    ax.set_yticks(unique_labels)
    ax.set_yticklabels([label_names[i] if i < len(label_names) else f"Class {i}" for i in unique_labels])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved test predictions plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(run_dir, csv_path, keypoints_dir, output_path=None):
    """Plot confusion matrix"""
    # Load model
    model_path = Path(run_dir) / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        model = keras.models.load_model(str(model_path), compile=False)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not load model: {e}")
        print("Skipping confusion matrix plot...")
        return
    
    # Load data
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False, use_smart_sampling=True)
    X_test, y_test = loader.get_split_data('test')
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get label names
    label_mapping = load_label_mapping(run_dir)
    if label_mapping:
        label_names = label_mapping['classes']
    else:
        label_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Percentage'})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_accuracy(run_dir, csv_path, keypoints_dir, output_path=None):
    """Plot accuracy per class"""
    # Load model
    model_path = Path(run_dir) / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        model = keras.models.load_model(str(model_path), compile=False)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not load model: {e}")
        print("Skipping per-class accuracy plot...")
        return
    
    # Load data
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False, use_smart_sampling=True)
    X_test, y_test = loader.get_split_data('test')
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get label names
    label_mapping = load_label_mapping(run_dir)
    if label_mapping:
        label_names = label_mapping['classes']
    else:
        label_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Calculate per-class accuracy
    unique_labels = np.unique(y_test)
    per_class_acc = []
    per_class_names = []
    
    for label in unique_labels:
        mask = y_test == label
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_test[mask])
            per_class_acc.append(acc)
            per_class_names.append(label_names[label] if label < len(label_names) else f"Class {label}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bar chart
    bars = ax.bar(per_class_names, per_class_acc, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add overall accuracy line
    overall_acc = np.mean(y_test == y_pred)
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Accuracy: {overall_acc:.2%}')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class accuracy plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(run_dir, csv_path, keypoints_dir):
    """Print detailed classification report"""
    # Load model
    model_path = Path(run_dir) / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        model = keras.models.load_model(str(model_path), compile=False)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not load model: {e}")
        print("Skipping classification report...")
        return
    
    # Load data
    loader = SignLanguageDataLoader(csv_path, keypoints_dir, normalize=False, use_smart_sampling=True)
    X_test, y_test = loader.get_split_data('test')
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get label names
    label_mapping = load_label_mapping(run_dir)
    if label_mapping:
        label_names = label_mapping['classes']
    else:
        label_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Print classification report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument("--run-dir", type=str, required=True,
                       help="Path to model run directory (e.g., models/run_20260104_103453)")
    parser.add_argument("--csv", type=str, default="Data/Labels/dataset.csv",
                       help="Path to CSV file")
    parser.add_argument("--keypoints-dir", type=str, default="Data/Keypoints/rawVideos",
                       help="Path to keypoints directory")
    parser.add_argument("--output-dir", type=str, default="output/plots",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Visualizing Model Results")
    print("="*60)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    # Plot 1: Training History
    print("1. Plotting training history...")
    plot_training_history(run_dir, output_dir / "training_history.png")
    
    # Plot 2: Test Predictions (Actual vs Predicted)
    print("\n2. Plotting test predictions (Actual vs Predicted)...")
    plot_test_predictions(run_dir, args.csv, args.keypoints_dir, 
                         output_dir / "test_predictions.png")
    
    # Plot 3: Confusion Matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(run_dir, args.csv, args.keypoints_dir,
                         output_dir / "confusion_matrix.png")
    
    # Plot 4: Per-Class Accuracy
    print("\n4. Plotting per-class accuracy...")
    plot_per_class_accuracy(run_dir, args.csv, args.keypoints_dir,
                            output_dir / "per_class_accuracy.png")
    
    # Print classification report
    print("\n5. Classification Report:")
    print_classification_report(run_dir, args.csv, args.keypoints_dir)
    
    print("\n" + "="*60)
    print("All plots saved to:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()

