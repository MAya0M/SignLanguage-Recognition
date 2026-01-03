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
    
    def detect_word_boundaries(self, keypoints: np.ndarray, min_frames_per_word: int = 10, 
                               movement_threshold: float = 0.02) -> list:
        """
        Detect boundaries between words by analyzing hand movement
        
        Args:
            keypoints: Keypoints array with shape (num_frames, 2, 21, 3)
            min_frames_per_word: Minimum frames for a valid word segment
            movement_threshold: Threshold for detecting stillness (end of word)
            
        Returns:
            List of (start_frame, end_frame) tuples for each word segment
        """
        num_frames = keypoints.shape[0]
        if num_frames < min_frames_per_word:
            return [(0, num_frames)]
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, num_frames):
            # Calculate distance between frames (using wrist position as reference)
            frame_diff = np.abs(keypoints[i] - keypoints[i-1])
            movement = np.mean(frame_diff)
            movements.append(movement)
        
        if len(movements) == 0:
            return [(0, num_frames)]
        
        movements = np.array(movements)
        
        # Find frames with low movement (potential word boundaries)
        # Use a sliding window to smooth movement detection
        window_size = min(5, len(movements) // 4)
        if window_size < 1:
            window_size = 1
        
        smoothed_movements = []
        for i in range(len(movements)):
            start = max(0, i - window_size // 2)
            end = min(len(movements), i + window_size // 2 + 1)
            smoothed_movements.append(np.mean(movements[start:end]))
        
        smoothed_movements = np.array(smoothed_movements)
        threshold = np.percentile(smoothed_movements, 30)  # Bottom 30% = stillness
        
        # Find boundaries (low movement areas)
        boundaries = [0]
        in_stillness = False
        stillness_start = None
        
        for i, movement in enumerate(smoothed_movements):
            if movement < threshold:
                if not in_stillness:
                    in_stillness = True
                    stillness_start = i + 1  # +1 because movements start from frame 1
            else:
                if in_stillness and stillness_start is not None:
                    # End of stillness - potential boundary
                    boundary = (stillness_start + i + 1) // 2
                    if boundary - boundaries[-1] >= min_frames_per_word:
                        boundaries.append(boundary)
                    in_stillness = False
                    stillness_start = None
        
        boundaries.append(num_frames)
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start >= min_frames_per_word:
                segments.append((start, end))
        
        # If no segments found, return the whole video
        if len(segments) == 0:
            segments = [(0, num_frames)]
        
        return segments
    
    def predict_multiple_words(self, keypoints: np.ndarray, min_confidence: float = 0.1,
                               segment_method: str = 'auto') -> list:
        """
        Predict multiple words from a video by segmenting it
        
        Args:
            keypoints: Keypoints array with shape (num_frames, 2, 21, 3)
            min_confidence: Minimum confidence to include a prediction
            segment_method: 'auto' (detect boundaries) or 'sliding' (sliding window)
            
        Returns:
            List of dictionaries with predictions for each word segment
        """
        num_frames = keypoints.shape[0]
        max_length = self.input_shape[0]
        
        if segment_method == 'auto':
            # Detect word boundaries
            segments = self.detect_word_boundaries(keypoints)
        else:
            # Sliding window approach
            window_size = max_length
            step_size = window_size // 2
            segments = []
            for start in range(0, num_frames, step_size):
                end = min(start + window_size, num_frames)
                if end - start >= max_length // 2:  # Minimum segment size
                    segments.append((start, end))
        
        predictions = []
        
        for seg_idx, (start, end) in enumerate(segments):
            segment_keypoints = keypoints[start:end]
            
            # Predict for this segment
            try:
                result = self.predict_from_keypoints(segment_keypoints)
                
                # Only include if confidence is high enough
                if result['confidence'] >= min_confidence:
                    predictions.append({
                        'word': result['prediction'],
                        'confidence': result['confidence'],
                        'start_frame': int(start),
                        'end_frame': int(end),
                        'segment_index': seg_idx,
                        'all_predictions': result.get('all_predictions', {})
                    })
            except Exception as e:
                print(f"Warning: Failed to predict segment {seg_idx}: {e}")
                continue
        
        return predictions
    
    def predict_from_video(self, video_path: str, max_hands: int = 2, 
                         detect_multiple_words: bool = True) -> dict:
        """
        Predict from video file
        
        Args:
            video_path: Path to video file
            max_hands: Maximum number of hands to detect
            detect_multiple_words: If True, detect multiple words in the video
            
        Returns:
            Dictionary with prediction results
        """
        print(f"Extracting keypoints from video: {video_path}...")
        
        # Extract keypoints
        keypoints = extract_hand_keypoints_from_video(video_path, max_hands=max_hands)
        
        if keypoints is None or len(keypoints) == 0:
            raise ValueError(f"Failed to extract keypoints from {video_path}")
        
        print(f"Extracted {len(keypoints)} frames")
        
        if detect_multiple_words:
            # For short videos (like live chunks), treat as single word
            # For longer videos, try to detect multiple words
            num_frames = len(keypoints)
            max_length = self.input_shape[0]
            
            # If video is short (less than 1.5x max_length), treat as single word
            if num_frames < max_length * 1.5:
                # Short video - single prediction
                result = self.predict_from_keypoints(keypoints)
                return {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'all_predictions': result.get('all_predictions', {}),
                    'words': [{
                        'word': result['prediction'],
                        'confidence': result['confidence'],
                        'start_frame': 0,
                        'end_frame': num_frames
                    }],
                    'multiple_words_detected': False,
                    'word_count': 1
                }
            else:
                # Long video - detect multiple words
                words = self.predict_multiple_words(keypoints, min_confidence=0.1)
                
                if len(words) == 0:
                    # Fallback to single prediction
                    result = self.predict_from_keypoints(keypoints)
                    return {
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'all_predictions': result.get('all_predictions', {}),
                        'words': [{
                            'word': result['prediction'],
                            'confidence': result['confidence'],
                            'start_frame': 0,
                            'end_frame': num_frames
                        }],
                        'multiple_words_detected': False,
                        'word_count': 1
                    }
                
                # Return all words
                return {
                    'prediction': words[0]['word'],  # First word for backward compatibility
                    'confidence': words[0]['confidence'],
                    'all_predictions': words[0].get('all_predictions', {}),
                    'words': words,
                    'multiple_words_detected': True,
                    'word_count': len(words)
                }
        else:
            # Single prediction (original behavior)
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

