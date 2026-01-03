"""
Script to prepare everything for training:
1. Extract keypoints from new videos
2. Update CSV file with all videos
3. Verify everything is ready
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("🚀 Preparing Dataset for Training")
    print("=" * 60)
    
    # Step 1: Extract keypoints from videos
    print("\n[Step 1/3] Extracting keypoints from videos...")
    print("-" * 60)
    
    try:
        from scripts.extract_keypoints import process_all_videos
        
        base_dir = Path("Data")
        raw_videos_dir = base_dir / "rawVideos"
        keypoints_dir = base_dir / "Keypoints" / "rawVideos"
        
        if not raw_videos_dir.exists():
            print(f"❌ Error: {raw_videos_dir} not found!")
            return False
        
        # Process videos (will skip existing, process new ones)
        print(f"Processing videos from: {raw_videos_dir}")
        print(f"Output directory: {keypoints_dir}")
        print("\nThis may take a while...")
        
        process_all_videos(
            raw_videos_dir, 
            keypoints_dir, 
            skip_existing=True,  # Skip existing keypoints
            overwrite=False      # Don't overwrite existing
        )
        
        print("✅ Keypoints extraction complete!")
        
    except Exception as e:
        print(f"❌ Error extracting keypoints: {e}")
        return False
    
    # Step 2: Create/Update CSV
    print("\n[Step 2/3] Creating/Updating CSV file...")
    print("-" * 60)
    
    try:
        from scripts.create_dataset_csv import create_csv_dataset
        
        keypoints_dir = Path("Data/Keypoints/rawVideos")
        output_csv = Path("Data/Labels/dataset.csv")
        
        # Create Labels directory if needed
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Keypoints directory: {keypoints_dir}")
        print(f"Output CSV: {output_csv}")
        
        create_csv_dataset(
            keypoints_dir, 
            output_csv, 
            base_path="keypoints",
            train_ratio=0.6, 
            val_ratio=0.2, 
            test_ratio=0.2, 
            seed=42
        )
        
        print("✅ CSV file created/updated!")
        
    except Exception as e:
        print(f"❌ Error creating CSV: {e}")
        return False
    
    # Step 3: Verify everything
    print("\n[Step 3/3] Verifying dataset...")
    print("-" * 60)
    
    try:
        import pandas as pd
        
        csv_path = Path("Data/Labels/dataset.csv")
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        
        print(f"\n✅ Dataset verification:")
        print(f"   Total samples: {len(df)}")
        print(f"   Total classes: {df['label'].nunique()}")
        
        print(f"\n📹 Videos per word:")
        counts = df.groupby('label').size().sort_values(ascending=False)
        for label, count in counts.items():
            status = "✅" if count >= 20 else "⚠️" if count >= 10 else "❌"
            print(f"   {status} {label:15s}: {count:3d} videos")
        
        print(f"\n📊 Split distribution:")
        split_counts = df.groupby('split').size()
        for split, count in split_counts.items():
            print(f"   {split:5s}: {count:3d} ({count/len(df)*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("✅ Everything is ready for training!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Push to Git: git add . && git commit -m 'Add new videos' && git push")
        print("2. Train model in Google Colab (run the notebook)")
        print("3. Download the trained model")
        print("4. Add model to models/ directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

