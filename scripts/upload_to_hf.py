#!/usr/bin/env python3
"""Upload trained models to HuggingFace Hub."""

import os
import glob
from huggingface_hub import HfApi, create_repo

# Set your HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "mahmoudomarus/nsca-world-model"  # Change to your username/repo

def upload_models():
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    # Find all checkpoint files
    checkpoint_files = glob.glob("checkpoints/*.pth")
    
    if not checkpoint_files:
        print("No checkpoint files found in checkpoints/")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    for local_path in checkpoint_files:
        filename = os.path.basename(local_path)
        print(f"Uploading {local_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                token=HF_TOKEN
            )
            print(f"  ✓ Uploaded {filename}")
        except Exception as e:
            print(f"  ✗ Failed to upload {filename}: {e}")
    
    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable first")
        print("  export HF_TOKEN='your_token_here'")
    else:
        upload_models()
