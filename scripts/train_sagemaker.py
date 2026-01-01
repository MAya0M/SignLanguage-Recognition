"""
SageMaker Training Script
This script is designed to run in SageMaker environment
"""

import os
import subprocess
import sys
import json

if __name__ == "__main__":
    # SageMaker sets environment variables
    # Training data is in /opt/ml/input/data/training
    # Model output goes to /opt/ml/model
    
    # Parse hyperparameters from environment (set by SageMaker)
    hyperparameters = {}
    if os.path.exists('/opt/ml/input/config/hyperparameters.json'):
        with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
            hyperparameters = json.load(f)
    
    # Default values
    csv_path = hyperparameters.get('csv-path', '/opt/ml/input/data/training/Data/Labels/dataset.csv')
    keypoints_dir = hyperparameters.get('keypoints-dir', '/opt/ml/input/data/training/Data/Keypoints/rawVideos')
    output_dir = '/opt/ml/model'
    batch_size = int(hyperparameters.get('batch-size', '32'))
    epochs = int(hyperparameters.get('epochs', '100'))
    gru_units = int(hyperparameters.get('gru-units', '128'))
    num_gru_layers = int(hyperparameters.get('num-gru-layers', '2'))
    dropout = float(hyperparameters.get('dropout', '0.3'))
    learning_rate = float(hyperparameters.get('learning-rate', '0.001'))
    patience = int(hyperparameters.get('patience', '10'))
    
    # Build command
    cmd = [
        sys.executable, "train_model.py",
        "--csv", csv_path,
        "--keypoints-dir", keypoints_dir,
        "--output-dir", output_dir,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--gru-units", str(gru_units),
        "--num-gru-layers", str(num_gru_layers),
        "--dropout", str(dropout),
        "--learning-rate", str(learning_rate),
        "--patience", str(patience)
    ]
    
    print("="*60)
    print("SageMaker Training")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    # Run training
    subprocess.check_call(cmd)

