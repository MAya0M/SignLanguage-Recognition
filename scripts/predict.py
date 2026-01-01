"""
Inference script for Sign Language Recognition
Predicts sign language words from video files or keypoint arrays
"""

import argparse
import numpy as np
from pathlib import Path
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_keypoints import extract_hand_keypoints_from_video, normalize_keypoints


class SignLanguagePredictor:
    """Predictor for sign language recognition"""
    
    def __init__(self, model_path: str, label_mapping_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved .keras model file
            label_mapping_path: Path to label_mapping.json (if None, tries to find in same directory)
        """
        self.model_path = Path(model_path)
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(str(self.model_path))
        print("Model loaded successfully!")
        
        # Load label mapping
        if label_mapping_path is None:
            label_mapping_path = self.model_path.parent / "label_mapping.json"
        
        label_mapping_path = Path(label_mapping_path)
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
            self.label_names = self.label_mapping['classes']
            print(f"Labels: {self.label_names}")
        else:
            print("Warning: label_mapping.json not found. Using numeric labels.")
            self.label_names = None
        
        # Get expected input shape from model
        self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
        print(f"Expected input shape: {self.input_shape}")
    
    def preprocess_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Preprocess keypoints for prediction
        
        Args:
            keypoints: Keypoints array with shape (num_frames, 2, 21, 3)
            
        Returns:
            Preprocessed array with shape (1, max_length, features)
        """
        # Normalize keypoints (same as training)
        keypoints = normalize_keypoints(keypoints)
        
        # Flatten to (num_frames, features)
        num_frames = keypoints.shape[0]
        keypoints_flat = keypoints.reshape(num_frames, -1)
        
        # Pad to expected length
        max_length = self.input_shape[0]
        keypoints_padded = pad_sequences(
            [keypoints_flat],
            maxlen=max_length,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        
        return keypoints_padded
    
    def predict_from_keypoints(self, keypoints: np.ndarray) -> dict:
        """
        Predict from keypoints array
        
        Args:
            keypoints: Keypoints array with shape (num_frames, 2, 21, 3)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        X = self.preprocess_keypoints(keypoints)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        
        # Get label name
        if self.label_names:
            label = self.label_names[top_idx]
        else:
            label = str(top_idx)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            if self.label_names:
                label_name = self.label_names[idx]
            else:
                label_name = str(idx)
            top_3.append({
                'label': label_name,
                'confidence': float(predictions[0][idx])
            })
        
        return {
            'prediction': label,
            'confidence': confidence,
            'top_3': top_3,
            'all_predictions': {
                (self.label_names[i] if self.label_names else str(i)): float(predictions[0][i])
                for i in range(len(predictions[0]))
            }
        }
    
    def predict_from_video(self, video_path: str, max_hands: int = 2) -> dict:
        """
        Predict from video file
        
        Args:
            video_path: Path to video file
            max_hands: Maximum number of hands to detect
            
        Returns:
            Dictionary with prediction results
        """
        print(f"Extracting keypoints from video: {video_path}...")
        
        # Extract keypoints
        keypoints = extract_hand_keypoints_from_video(video_path, max_hands=max_hands)
        
        if keypoints is None or len(keypoints) == 0:
            raise ValueError(f"Failed to extract keypoints from {video_path}")
        
        print(f"Extracted {len(keypoints)} frames")
        
        # Predict
        return self.predict_from_keypoints(keypoints)
    
    def predict_from_npy(self, npy_path: str) -> dict:
        """
        Predict from .npy file
        
        Args:
            npy_path: Path to .npy file with keypoints
            
        Returns:
            Dictionary with prediction results
        """
        print(f"Loading keypoints from {npy_path}...")
        keypoints = np.load(npy_path)
        
        return self.predict_from_keypoints(keypoints)


def main():
    parser = argparse.ArgumentParser(description="Predict sign language from video or keypoints")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.keras file)")
    parser.add_argument("--label-mapping", type=str, default=None,
                       help="Path to label_mapping.json (optional, tries to find in model directory)")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file")
    parser.add_argument("--keypoints", type=str, default=None,
                       help="Path to .npy keypoints file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SignLanguagePredictor(args.model, args.label_mapping)
    
    # Predict
    if args.video:
        results = predictor.predict_from_video(args.video)
    elif args.keypoints:
        results = predictor.predict_from_npy(args.keypoints)
    else:
        parser.error("Must provide either --video or --keypoints")
    
    # Print results
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    print(f"Predicted: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    print("\nTop 3 predictions:")
    for i, pred in enumerate(results['top_3'], 1):
        print(f"  {i}. {pred['label']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    print("="*60)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()

