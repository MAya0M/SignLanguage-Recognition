import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import urllib.request

try:
    # Try new API (MediaPipe 0.10+)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image, ImageFormat
    USE_NEW_API = True
except ImportError:
    # Fall back to old API
    import mediapipe as mp
    USE_NEW_API = False

# Model URL for Hand Landmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

def download_model_if_needed():
    """Download the Hand Landmarker model if it doesn't exist"""
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)
    
    if not MODEL_PATH.exists():
        print(f"Downloading Hand Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")
    return str(MODEL_PATH)

def normalize_keypoints(keypoints_array):
    """
    Advanced normalization for hand keypoints to make recognition invariant to:
    1. Hand position in frame (translation)
    2. Hand size (scale)
    3. Left/Right hand (mirror to consistent orientation)
    4. Hand rotation in XY plane (align to consistent direction)
    
    Args:
        keypoints_array: numpy array with shape (num_frames, num_hands, 21, 3)
    
    Returns:
        Normalized numpy array with same shape
    """
    # Create a copy to avoid modifying original
    normalized = keypoints_array.copy()
    
    # MediaPipe landmark indices
    WRIST_IDX = 0
    MIDDLE_FINGER_MCP_IDX = 9
    INDEX_FINGER_MCP_IDX = 5
    PINKY_MCP_IDX = 17
    
    num_frames, num_hands, num_keypoints, num_coords = normalized.shape
    
    for frame_idx in range(num_frames):
        for hand_idx in range(num_hands):
            hand_keypoints = normalized[frame_idx, hand_idx, :, :].copy()  # Shape: (21, 3)
            
            # Check if hand was detected (wrist not at origin)
            wrist = hand_keypoints[WRIST_IDX, :]
            
            # Skip normalization if no hand detected (all zeros)
            if np.allclose(wrist, 0.0):
                continue
            
            # Step 1: Translate so wrist is at (0, 0, 0)
            hand_keypoints = hand_keypoints - wrist
            
            # Step 2: Determine if left or right hand and flip if needed
            # Use the direction from pinky to index finger MCP to determine hand orientation
            index_mcp = hand_keypoints[INDEX_FINGER_MCP_IDX, :]
            pinky_mcp = hand_keypoints[PINKY_MCP_IDX, :]
            
            # Vector from pinky to index (in camera view)
            # For right hand: index is to the right (positive X), pinky is to the left (negative X)
            # For left hand: index is to the left (negative X), pinky is to the right (positive X)
            hand_direction = index_mcp - pinky_mcp
            
            # If the x-component is negative, it's likely a left hand, flip it
            if hand_direction[0] < 0:
                # Mirror the hand across the YZ plane (flip X coordinates)
                hand_keypoints[:, 0] = -hand_keypoints[:, 0]
            
            # Step 3: Rotate hand to align with a consistent direction
            # Align the hand so the middle finger MCP is in a consistent direction
            middle_mcp = hand_keypoints[MIDDLE_FINGER_MCP_IDX, :]
            
            # Project to XY plane for rotation alignment
            middle_mcp_xy = middle_mcp[:2]
            middle_mcp_xy_norm = np.linalg.norm(middle_mcp_xy)
            
            if middle_mcp_xy_norm > 1e-6:
                # Calculate angle to align middle finger MCP to positive Y direction
                target_direction = np.array([0.0, 1.0])
                current_direction = middle_mcp_xy / middle_mcp_xy_norm
                
                # Calculate rotation angle
                # For 2D vectors, cross product is: a.x * b.y - a.y * b.x
                cos_angle = np.dot(current_direction, target_direction)
                sin_angle = current_direction[0] * target_direction[1] - current_direction[1] * target_direction[0]
                angle = np.arctan2(sin_angle, cos_angle)
                
                # Apply rotation to XY plane
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                          [sin_a, cos_a, 0],
                                          [0, 0, 1]])
                
                # Rotate all keypoints
                hand_keypoints = hand_keypoints @ rotation_matrix.T
            
            # Step 4: Scale by hand size (distance from wrist to middle finger MCP)
            middle_mcp = hand_keypoints[MIDDLE_FINGER_MCP_IDX, :]
            hand_size = np.linalg.norm(middle_mcp)
            
            # Avoid division by zero
            if hand_size > 1e-6:  # Very small threshold
                hand_keypoints = hand_keypoints / hand_size
            
            normalized[frame_idx, hand_idx, :, :] = hand_keypoints
    
    return normalized

def extract_hand_keypoints_from_video(video_path, max_hands=2):
    """
    Extracts hand keypoints from a video using MediaPipe Hand Landmarker
    
    Args:
        video_path: Path to the video file
        max_hands: Maximum number of hands to detect (1 or 2)
    
    Returns:
        numpy array with shape (num_frames, num_hands, 21, 3) 
        where each hand contains 21 keypoints with coordinates (x, y, z)
    """
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open file {video_path}")
        return None
    
    if USE_NEW_API:
        # Use new API (MediaPipe 0.10+)
        # Download model if needed
        model_path = download_model_if_needed()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
        all_keypoints = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame
            detection_result = detector.detect(mp_image)
            
            # Prepare array for keypoints of current frame
            frame_keypoints = np.zeros((max_hands, 21, 3))
            
            if detection_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    if idx >= max_hands:
                        break
                    
                    # Extract 21 keypoints
                    for i, landmark in enumerate(hand_landmarks):
                        frame_keypoints[idx, i, 0] = landmark.x
                        frame_keypoints[idx, i, 1] = landmark.y
                        frame_keypoints[idx, i, 2] = landmark.z
            
            all_keypoints.append(frame_keypoints)
        
        cap.release()
        detector.close()
        
    else:
        # Use old API (MediaPipe < 0.10)
        mp_hands = mp.solutions.hands
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            all_keypoints = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB (MediaPipe expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = hands.process(rgb_frame)
                
                # Prepare array for keypoints of current frame
                frame_keypoints = np.zeros((max_hands, 21, 3))
                
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if idx >= max_hands:
                            break
                        
                        # Extract 21 keypoints
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            frame_keypoints[idx, i, 0] = landmark.x
                            frame_keypoints[idx, i, 1] = landmark.y
                            frame_keypoints[idx, i, 2] = landmark.z
                
                all_keypoints.append(frame_keypoints)
            
            cap.release()
    
    # Convert to numpy array: (num_frames, num_hands, 21, 3)
    keypoints_array = np.array(all_keypoints)
    
    # Normalize keypoints (translate wrist to origin, scale by hand size)
    keypoints_array = normalize_keypoints(keypoints_array)
    
    return keypoints_array


def process_all_videos(input_dir, output_dir, skip_existing=False, overwrite=True):
    """
    Iterates through all videos in the directory and extracts keypoints from them
    
    Args:
        input_dir: Input directory with videos
        output_dir: Output directory for saving .npy files
        skip_existing: If True, skip files that already exist (default: False)
        overwrite: If True, overwrite existing files (default: True)
                   Note: overwrite takes precedence over skip_existing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    skipped_count = 0
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            # Create output path preserving directory structure
            relative_path = video_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.npy')
            
            # Check if file exists and should be skipped
            file_existed = output_file.exists()
            if file_existed and skip_existing and not overwrite:
                skipped_count += 1
                continue  # Skip existing file
            
            # Extract keypoints
            keypoints = extract_hand_keypoints_from_video(video_file, max_hands=2)
            
            if keypoints is not None and len(keypoints) > 0:
                # Create necessary directories
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save as .npy
                np.save(output_file, keypoints)
                
                action = "Overwritten" if file_existed else "Saved"
                print(f"{action}: {output_file} (shape: {keypoints.shape})")
            else:
                print(f"Warning: No keypoints found in {video_file}")
                
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} existing file(s). Use --skip-existing to skip files.")


if __name__ == "__main__":
    import sys
    
    # Set up paths
    base_dir = Path("Data")
    raw_videos_dir = base_dir / "rawVideos"
    sessions_dir = base_dir / "Sessions"
    keypoints_dir = base_dir / "Keypoints"
    
    # Check for command line arguments
    skip_mode = "--skip-existing" in sys.argv or "-s" in sys.argv
    overwrite = not skip_mode  # By default, overwrite existing files
    skip_existing = skip_mode
    
    print("=" * 60)
    print("Extracting hand keypoints from all videos")
    if overwrite:
        print("Mode: OVERWRITE (will reprocess all existing files)")
    else:
        print("Mode: SKIP EXISTING (use --skip-existing to skip existing files)")
    print("=" * 60)
    
    # Process videos from rawVideos
    if raw_videos_dir.exists():
        print("\n[1/2] Processing videos from rawVideos...")
        process_all_videos(raw_videos_dir, keypoints_dir / "rawVideos", 
                          skip_existing=skip_existing, overwrite=overwrite)
    else:
        print(f"⚠ Directory not found: {raw_videos_dir}")
    
    # Process videos from Sessions
    if sessions_dir.exists():
        print("\n[2/2] Processing videos from Sessions...")
        process_all_videos(sessions_dir, keypoints_dir / "Sessions",
                          skip_existing=skip_existing, overwrite=overwrite)
    else:
        print(f"⚠ Directory not found: {sessions_dir}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

