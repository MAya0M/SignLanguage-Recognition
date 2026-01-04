"""
Update the tar.gz archive with new data
"""

import tarfile
import os
from pathlib import Path

def update_tar_archive():
    """Update sign_language_data.tar.gz with current Data directory"""
    data_dir = Path('Data')
    tar_path = Path('sign_language_data.tar.gz')
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return
    
    print("=" * 60)
    print("Updating tar.gz archive")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output: {tar_path}")
    print("=" * 60)
    
    # Remove old archive if exists
    if tar_path.exists():
        print(f"Removing old archive...")
        tar_path.unlink()
    
    # Create new archive
    print(f"Creating new archive...")
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add('Data', arcname='Data')
    
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"\n[OK] Archive created: {tar_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    update_tar_archive()

