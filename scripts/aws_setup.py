"""
AWS Setup Script for Training Sign Language Recognition Model
This script helps prepare and upload data to AWS for training
"""

import os
import boto3
from pathlib import Path
import tarfile
import argparse
from botocore.exceptions import ClientError


def create_data_archive(data_dir: str = "Data", output_file: str = "sign_language_data.tar.gz"):
    """
    Create a tar.gz archive of the data directory
    
    Args:
        data_dir: Directory containing Keypoints and Labels
        output_file: Output archive filename
    """
    print(f"Creating archive of {data_dir}...")
    
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(data_dir, arcname=os.path.basename(data_dir))
    
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
    print(f"Archive created: {output_file} ({file_size:.2f} MB)")
    
    return output_file


def upload_to_s3(
    file_path: str,
    bucket_name: str,
    s3_key: str = None,
    aws_profile: str = None
):
    """
    Upload file to S3
    
    Args:
        file_path: Local file path to upload
        bucket_name: S3 bucket name
        s3_key: S3 key (object name). If None, uses filename
        aws_profile: AWS profile name (optional)
    """
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3_client = session.client('s3')
    
    if s3_key is None:
        s3_key = Path(file_path).name
    
    try:
        print(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}...")
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"Upload completed!")
        return f"s3://{bucket_name}/{s3_key}"
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise


def create_training_script_aws():
    """Create a training script optimized for AWS"""
    script_content = """#!/bin/bash
# AWS Training Script for Sign Language Recognition

# Install dependencies
pip install -r requirements.txt

# Download data from S3 if needed
# aws s3 cp s3://your-bucket/sign_language_data.tar.gz ./
# tar -xzf sign_language_data.tar.gz

# Run training
python train_model.py \\
    --csv Data/Labels/dataset.csv \\
    --keypoints-dir Data/Keypoints/rawVideos \\
    --output-dir models \\
    --batch-size 32 \\
    --epochs 100 \\
    --gru-units 128 \\
    --num-gru-layers 2 \\
    --dropout 0.3 \\
    --learning-rate 0.001 \\
    --patience 10

# Upload model to S3
# aws s3 sync models/ s3://your-bucket/models/ --exclude "*" --include "*.keras"
"""
    
    with open("train_aws.sh", "w") as f:
        f.write(script_content)
    
    # Make executable (Unix-like)
    os.chmod("train_aws.sh", 0o755)
    print("Created train_aws.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS setup for sign language training")
    parser.add_argument("--create-archive", action="store_true",
                       help="Create data archive")
    parser.add_argument("--upload", type=str, metavar="BUCKET",
                       help="Upload data archive to S3 bucket")
    parser.add_argument("--s3-key", type=str,
                       help="S3 key (object name) for upload")
    parser.add_argument("--aws-profile", type=str,
                       help="AWS profile name")
    parser.add_argument("--create-script", action="store_true",
                       help="Create AWS training script")
    
    args = parser.parse_args()
    
    if args.create_archive:
        archive_file = create_data_archive()
        print(f"\nArchive ready: {archive_file}")
        print(f"You can upload it with: python aws_setup.py --upload YOUR_BUCKET")
    
    if args.upload:
        archive_file = "sign_language_data.tar.gz"
        if not Path(archive_file).exists():
            print(f"Archive not found. Creating it first...")
            archive_file = create_data_archive()
        
        s3_path = upload_to_s3(
            archive_file,
            args.upload,
            s3_key=args.s3_key,
            aws_profile=args.aws_profile
        )
        print(f"\nData uploaded to: {s3_path}")
    
    if args.create_script:
        create_training_script_aws()
        print("\nAWS training script created: train_aws.sh")
    
    if not any([args.create_archive, args.upload, args.create_script]):
        parser.print_help()
        print("\nExamples:")
        print("  # Create data archive")
        print("  python aws_setup.py --create-archive")
        print("\n  # Upload to S3")
        print("  python aws_setup.py --upload my-bucket --s3-key data/sign_language_data.tar.gz")
        print("\n  # Create training script")
        print("  python aws_setup.py --create-script")

