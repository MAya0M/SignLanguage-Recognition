"""
Evaluate trained model and show prediction quality
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_loader import SignLanguageDataLoader
from tensorflow import keras

def evaluate_model(model_path: str, csv_path: str, keypoints_dir: str):
    """Evaluate model and show detailed metrics"""
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load label mapping
    model_dir = Path(model_path).parent
    label_mapping_path = model_dir / "label_mapping.json"
    
    if label_mapping_path.exists():
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        label_names = label_mapping['classes']
    else:
        print("⚠️  Label mapping not found, using default")
        label_names = ['GOODBYE', 'HELLO', 'ILOVEYOU', 'NO', 'PLEASE', 'SORRY', 'THANKS', 'YES']
    
    # Load data
    print(f"\nLoading test data...")
    loader = SignLanguageDataLoader(csv_path, keypoints_dir)
    splits = loader.get_all_splits()
    
    X_test, y_test = splits['test']
    
    # Make predictions
    print(f"\nMaking predictions on {len(X_test)} test samples...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    
    print("\n" + "=" * 60)
    print("Prediction Quality")
    print("=" * 60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare to random chance
    num_classes = len(label_names)
    random_chance = 1.0 / num_classes
    print(f"Random Chance: {random_chance:.4f} ({random_chance*100:.2f}%)")
    
    if accuracy <= random_chance * 1.1:  # Within 10% of random
        print(f"\n❌ Model is NOT learning - accuracy is at random chance level!")
        print(f"   This means the model is just guessing.")
    elif accuracy < 0.5:
        print(f"\n⚠️  Model accuracy is low - needs improvement")
    elif accuracy < 0.7:
        print(f"\n✅ Model is learning but could be better")
    else:
        print(f"\n✅ Model is performing well!")
    
    # Classification report
    print(f"\n" + "=" * 60)
    print("Per-Class Performance")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Rows = True labels, Columns = Predicted labels")
    print("\n" + " " * 12, end="")
    for name in label_names:
        print(f"{name[:8]:>8}", end="")
    print()
    for i, name in enumerate(label_names):
        print(f"{name[:12]:12}", end="")
        for j in range(len(label_names)):
            print(f"{cm[i, j]:8}", end="")
        print()
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i, name in enumerate(label_names):
        class_mask = y_test == i
        if np.any(class_mask):
            class_accuracy = np.mean(y_pred[class_mask] == y_test[class_mask])
            print(f"  {name:12s}: {class_accuracy:.2%}")
    
    # Most confused pairs
    print(f"\nMost Confused Pairs:")
    max_confusions = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                max_confusions.append((cm[i, j], label_names[i], label_names[j]))
    max_confusions.sort(reverse=True)
    for count, true_label, pred_label in max_confusions[:5]:
        print(f"  {true_label} → {pred_label}: {count} times")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--csv", type=str, default="Data/Labels/dataset.csv", help="Path to CSV")
    parser.add_argument("--keypoints-dir", type=str, default="Data/Keypoints/rawVideos", help="Keypoints directory")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.csv, args.keypoints_dir)

