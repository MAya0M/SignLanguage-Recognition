"""
Script to check if trained model exists and is valid
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_model():
    """Check if trained model exists"""
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print(f"   Expected: {models_dir.absolute()}")
        return False
    
    # Find all run directories
    run_dirs = sorted(models_dir.glob('run_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        print("❌ No trained models found!")
        print(f"   Expected directory: {models_dir / 'run_XXXXX'}")
        print("\n   To add a model:")
        print("   1. Train model in Google Colab")
        print("   2. Download the run_XXXXX folder")
        print("   3. Copy it to the models/ directory")
        print("   4. See docs/HOW_TO_ADD_MODEL.md for details")
        return False
    
    print(f"✅ Found {len(run_dirs)} model run(s):\n")
    
    all_valid = True
    for run_dir in run_dirs:
        print(f"📁 {run_dir.name}")
        
        # Check for required files
        model_file = run_dir / 'best_model.keras'
        label_file = run_dir / 'label_mapping.json'
        
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ best_model.keras ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ best_model.keras - MISSING!")
            all_valid = False
        
        if label_file.exists():
            print(f"   ✅ label_mapping.json")
        else:
            print(f"   ⚠️  label_mapping.json - MISSING (optional but recommended)")
        
        print()
    
    if all_valid:
        latest = run_dirs[0]
        print(f"✅ Latest model: {latest.name}")
        print(f"   Path: {latest.absolute()}")
        return True
    else:
        print("⚠️  Some models are missing required files!")
        return False

if __name__ == '__main__':
    success = check_model()
    sys.exit(0 if success else 1)

