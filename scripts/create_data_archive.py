"""
Create/Update sign_language_data.tar.gz archive with all data
"""

import tarfile
import sys
from pathlib import Path
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_archive():
    """Create tar.gz archive with all data"""
    print("=" * 60)
    print("Creating sign_language_data.tar.gz archive")
    print("=" * 60)
    
    data_dir = Path("Data")
    archive_name = "sign_language_data.tar.gz"
    
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} directory not found!")
        return False
    
    # Check what we're archiving
    print(f"\n📁 Data directory: {data_dir.absolute()}")
    
    # Count files
    video_files = list(data_dir.rglob("*.mp4"))
    npy_files = list(data_dir.rglob("*.npy"))
    csv_files = list(data_dir.rglob("*.csv"))
    
    print(f"   Video files: {len(video_files)}")
    print(f"   Keypoint files (.npy): {len(npy_files)}")
    print(f"   CSV files: {len(csv_files)}")
    
    # Create archive
    print(f"\n📦 Creating archive: {archive_name}")
    print("   This may take a few minutes...")
    
    try:
        file_count = 0
        with tarfile.open(archive_name, "w:gz") as tar:
            # Add all files in Data directory
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    # Get relative path from Data directory
                    arcname = file_path.relative_to(data_dir.parent)
                    tar.add(file_path, arcname=arcname)
                    file_count += 1
                    if file_count % 50 == 0:
                        print(f"   Added {file_count} files...")
        
        # Get archive size
        archive_size_mb = Path(archive_name).stat().st_size / (1024 * 1024)
        
        print(f"\n✅ Archive created successfully!")
        print(f"   Archive: {archive_name}")
        print(f"   Size: {archive_size_mb:.1f} MB")
        print(f"   Total files: {file_count}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating archive: {e}")
        return False

if __name__ == "__main__":
    success = create_archive()
    sys.exit(0 if success else 1)

