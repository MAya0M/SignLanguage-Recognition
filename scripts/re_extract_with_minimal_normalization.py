"""
Re-extract all keypoints with MINIMAL normalization (only translate, no rotate/scale)
This will preserve size and rotation differences which help distinguish classes
"""

import sys
import os
from pathlib import Path

# Fix path for Colab
try:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
except NameError:
    project_root = Path(os.getcwd())
    if project_root.name == 'scripts':
        project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("Re-extracting Keypoints with MINIMAL Normalization")
print("=" * 60)
print("\n⚠️  IMPORTANT:")
print("   This will OVERWRITE all existing keypoint files!")
print("   Normalization changed to MINIMAL (only translate)")
print("   This preserves size/rotation differences between classes")
print("\n" + "=" * 60)

# Step 1: Re-extract keypoints with minimal normalization
print("\n[Step 1/3] Re-extracting keypoints from videos...")
print("   (This may take a while - processing all videos)")
os.system('python scripts/extract_keypoints.py')

# Step 2: Regenerate CSV
print("\n[Step 2/3] Regenerating dataset.csv...")
os.system('python scripts/create_dataset_csv.py')

# Step 3: Verify
print("\n[Step 3/3] Verifying dataset...")
os.system('python scripts/check_dataset.py')

print("\n" + "=" * 60)
print("✅ Done! Keypoints re-extracted with minimal normalization")
print("=" * 60)
print("\n📝 Next steps:")
print("   1. Check the dataset statistics above")
print("   2. Run training again - model should learn better now!")
print("=" * 60)

