#!/usr/bin/env python3
"""
Convenience launcher for NSCA training. Defaults to RTX 3050 config.

Usage:
    python train.py
    python train.py --config configs/training_config.yaml
    python train.py --phase vision --data-dir /path/to/data
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "train_world_model.py"
    local_config = root / "configs" / "training_config_local.yaml"
    
    args = list(sys.argv[1:])
    if "--config" not in args:
        args = ["--config", str(local_config)] + args
    
    cmd = [sys.executable, str(script)] + args
    sys.exit(subprocess.call(cmd))
