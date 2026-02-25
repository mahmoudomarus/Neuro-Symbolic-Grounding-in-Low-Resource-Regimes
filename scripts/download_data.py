#!/usr/bin/env python3
"""
Data Download Script for NSCA Training.

⚠️  RUN THIS ON YOUR CLOUD INSTANCE, NOT LOCALLY!
    Large datasets (50-450GB) should be downloaded where training happens.

Downloads and prepares datasets from HuggingFace:
- Something-Something v2 (action recognition) - 20GB
- Kinetics-400 (temporal dynamics) - 450GB full, use --subset
- Greatest Hits (impact sounds - for audio v2.1)

Usage:
    # On cloud instance - download training data
    python scripts/download_data.py --all --subset 0.1
    
    # Download specific dataset
    python scripts/download_data.py --dataset somethingsomething
    
    # Local testing only (tiny CIFAR subset)
    python scripts/download_data.py --local-test
    
Note: Pre-validation tests (noisy_tv, forgetting, balloon, slot_discovery)
      use SYNTHETIC data and don't require any downloads!
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Check for required packages
try:
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import login
    # HfFolder is deprecated in newer versions
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        # Create a simple replacement
        class HfFolder:
            @staticmethod
            def get_token():
                import os
                return os.environ.get('HF_TOKEN', None)
except ImportError:
    print("Installing required packages...")
    os.system("pip install datasets huggingface_hub")
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import login
    class HfFolder:
        @staticmethod
        def get_token():
            import os
            return os.environ.get('HF_TOKEN', None)


def get_cache_dir() -> Path:
    """Get the data cache directory."""
    cache_dir = Path("./data/huggingface_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def check_hf_auth() -> bool:
    """Check if HuggingFace is authenticated."""
    token = HfFolder.get_token()
    if token:
        print("✓ HuggingFace authentication found")
        return True
    else:
        print("⚠ No HuggingFace token found")
        print("  Some datasets require authentication.")
        print("  Run: huggingface-cli login")
        return False


def download_something_something(subset: float = 1.0, cache_dir: Optional[Path] = None):
    """
    Download Something-Something v2 dataset.
    
    220K video clips of humans performing actions with objects.
    Excellent for learning action-object relationships.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: Something-Something v2")
    print("="*60)
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    try:
        # Get dataset info first
        builder = load_dataset_builder("HuggingFaceM4/something_something_v2")
        print(f"Dataset info: {builder.info.description[:200]}...")
        print(f"Features: {list(builder.info.features.keys())}")
        
        # Calculate split
        if subset < 1.0:
            split = f"train[:{int(subset*100)}%]"
            print(f"Downloading {subset*100:.0f}% of training data...")
        else:
            split = "train"
            print("Downloading full training data...")
        
        dataset = load_dataset(
            "HuggingFaceM4/something_something_v2",
            split=split,
            cache_dir=str(cache_dir),
        )
        
        print(f"\n✓ Downloaded {len(dataset)} samples")
        print(f"Sample keys: {dataset[0].keys()}")
        
        # Save info
        info_path = cache_dir / "something_something_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Samples: {len(dataset)}\n")
            f.write(f"Features: {list(dataset.features.keys())}\n")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to download Something-Something v2: {e}")
        print("\nTrying alternative: UCF101...")
        
        try:
            dataset = load_dataset(
                "uoft-cs/cifar10",  # Fallback to simpler dataset
                split=f"train[:{int(subset*100)}%]" if subset < 1.0 else "train",
                cache_dir=str(cache_dir),
            )
            print(f"✓ Downloaded fallback dataset: {len(dataset)} samples")
            return dataset
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return None


def download_kinetics(subset: float = 0.1, cache_dir: Optional[Path] = None):
    """
    Download Kinetics-400 dataset (subset recommended due to size).
    
    300K video clips of human actions.
    Warning: Full dataset is ~450GB, use subset=0.1 for testing.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: Kinetics-400")
    print("="*60)
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    print(f"⚠ Kinetics-400 is very large (~450GB full)")
    print(f"  Downloading {subset*100:.0f}% subset...")
    
    try:
        # Kinetics can be tricky - try different sources
        try:
            dataset = load_dataset(
                "AlexFierworker/kinetics400",
                split=f"train[:{int(subset*100)}%]",
                cache_dir=str(cache_dir),
            )
        except:
            # Alternative source
            dataset = load_dataset(
                "nateraw/kinetics",
                split=f"train[:{int(subset*100)}%]",
                cache_dir=str(cache_dir),
            )
        
        print(f"\n✓ Downloaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to download Kinetics: {e}")
        print("\nKinetics often requires manual download due to size.")
        print("Alternative: Use Something-Something v2 instead (smaller, still good).")
        return None


def download_greatest_hits(cache_dir: Optional[Path] = None):
    """
    Download audio data for training.
    Prefers torchaudio SpeechCommands (no HuggingFace scripts).
    Greatest Hits requires manual download from Cornell.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: Speech Commands (audio)")
    print("="*60)
    
    # Use torchaudio SpeechCommands - works without HuggingFace script deprecation
    try:
        import torchaudio
        root = Path("./data/speech_commands")
        root.mkdir(parents=True, exist_ok=True)
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(root),
            url="speech_commands_v0.02",
            download=True,
            subset=None,
        )
        n = len(dataset)
        print(f"\n✓ Downloaded Speech Commands: {n} samples")
        return dataset
    except Exception as e:
        print(f"❌ Speech Commands failed: {e}")
        print("  Install torchaudio: pip install torchaudio")
        return None


def download_cifar_for_testing(cache_dir: Optional[Path] = None):
    """Download CIFAR-100 for quick testing."""
    print("\n" + "="*60)
    print("DOWNLOADING: CIFAR-100 (for quick testing)")
    print("="*60)
    
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    try:
        dataset = load_dataset(
            "cifar100",
            split="train",
            cache_dir=str(cache_dir),
        )
        
        print(f"\n✓ Downloaded {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def verify_downloads(cache_dir: Optional[Path] = None):
    """Verify downloaded datasets."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    print("\n" + "="*60)
    print("VERIFYING DOWNLOADS")
    print("="*60)
    
    # Check cache size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    print(f"Total cache size: {total_size / 1e9:.2f} GB")
    
    # List datasets
    print("\nCached datasets:")
    for subdir in cache_dir.iterdir():
        if subdir.is_dir():
            size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            print(f"  {subdir.name}: {size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download NSCA training data")
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--dataset', type=str, choices=[
        'somethingsomething', 'kinetics', 'greathits', 'cifar', 'audio'
    ], help='Specific dataset to download')
    parser.add_argument('--subset', type=float, default=0.1,
                       help='Fraction of data to download (default: 0.1 = 10%)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory for downloads')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing downloads')
    parser.add_argument('--local-test', action='store_true',
                       help='Download tiny dataset for local testing only')
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir()
    
    print("="*60)
    print("NSCA DATA DOWNLOAD SCRIPT")
    print("="*60)
    print(f"Cache directory: {cache_dir}")
    print(f"Subset: {args.subset*100:.0f}%")
    
    # Check authentication
    check_hf_auth()
    
    if args.verify:
        verify_downloads(cache_dir)
        return
    
    if args.local_test:
        print("\n⚠️  LOCAL/RTX 3050 MODE: Downloading CIFAR-100 + SpeechCommands")
        print("   These datasets support full training on 6GB VRAM.\n")
        download_cifar_for_testing(cache_dir)
        download_greatest_hits(cache_dir)  # Downloads SpeechCommands as substitute
        verify_downloads(cache_dir)
        return
    
    if args.all:
        download_cifar_for_testing(cache_dir)
        download_something_something(args.subset, cache_dir)
        download_kinetics(args.subset, cache_dir)
        download_greatest_hits(cache_dir)
    elif args.dataset:
        if args.dataset == 'somethingsomething':
            download_something_something(args.subset, cache_dir)
        elif args.dataset == 'kinetics':
            download_kinetics(args.subset, cache_dir)
        elif args.dataset in ['greathits', 'audio']:
            download_greatest_hits(cache_dir)
        elif args.dataset == 'cifar':
            download_cifar_for_testing(cache_dir)
    else:
        # Default: download small test datasets
        print("\nNo dataset specified. Downloading CIFAR for testing...")
        download_cifar_for_testing(cache_dir)
    
    verify_downloads(cache_dir)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify data: python scripts/download_data.py --verify")
    print("2. Start training: python scripts/train_world_model.py")


if __name__ == "__main__":
    main()
