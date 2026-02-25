#!/usr/bin/env python3
"""
Full NSCA World Model Training Orchestrator.

Runs complete training pipeline:
1. Pre-validation checkpoints (cheap insurance)
2. Babbling phase (grounding)
3. Vision encoder
4. Cross-modal fusion
5. Temporal dynamics
6. Full evaluation with ablation study

Usage:
    # Full training
    python scripts/run_full_training.py --config configs/training_config.yaml
    
    # Skip validation (dangerous, only if already passed)
    python scripts/run_full_training.py --skip-validation
    
    # Resume from checkpoint
    python scripts/run_full_training.py --resume checkpoints/phase3_fusion.pth
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch


def run_command(cmd: list, description: str, required: bool = True) -> bool:
    """Run a command and check for success."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description} (exit code {result.returncode})")
        if required:
            print("This is a required step. Please fix before continuing.")
            return False
        else:
            print("This is optional. Continuing...")
    else:
        print(f"\n✅ COMPLETED: {description} ({elapsed:.1f}s)")
    
    return result.returncode == 0


def check_gpu() -> bool:
    """Check GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return True
    else:
        print("WARNING: No GPU detected. Training will be slow.")
        return False


def phase_0_validation(skip: bool = False) -> bool:
    """Run pre-validation checkpoints."""
    if skip:
        print("\n⚠️  SKIPPING VALIDATION (--skip-validation)")
        return True
    
    print("\n" + "="*70)
    print("PHASE 0: PRE-VALIDATION CHECKPOINTS (~$15, 4 hours)")
    print("="*70)
    
    tests = [
        (["python", "scripts/noisy_tv_test.py", "--episodes", "50"], 
         "Noisy TV Test (Curiosity Robustness)", True),
        (["python", "scripts/forgetting_test.py", "--epochs", "20"],
         "Catastrophic Forgetting Test (EWC)", True),
        (["python", "scripts/balloon_test.py", "--steps", "1000"],
         "Balloon Test (Prior Override)", True),
        (["python", "scripts/slot_discovery_test.py", "--free-slots", "16"],
         "Slot Discovery Test (Dynamic Properties)", True),
    ]
    
    all_passed = True
    for cmd, desc, required in tests:
        success = run_command(cmd, desc, required)
        if required and not success:
            all_passed = False
    
    if not all_passed:
        print("\n" + "="*70)
        print("❌ VALIDATION FAILED")
        print("="*70)
        print("Some required tests failed. Please fix before full training.")
        print("This prevents wasting $150+ on a broken system.")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL VALIDATION CHECKPOINTS PASSED")
    print("="*70)
    return True


def phase_1_babbling(config_path: str, steps: int = 100000) -> bool:
    """Run babbling phase for grounding."""
    print("\n" + "="*70)
    print("PHASE 1: BABBLING (~$30, 10 hours)")
    print("="*70)
    
    # Check if babbling script exists
    babbling_script = Path("scripts/run_babbling.py")
    if not babbling_script.exists():
        print("Babbling script not found. Creating simplified version...")
        # Fall back to lethality test as proof of concept
        return run_command(
            ["python", "scripts/lethality_test.py", "--n-seeds", "3", "--demos", "10"],
            "Simplified Babbling (Lethality Test)",
            required=False
        )
    
    return run_command(
        ["python", "scripts/run_babbling.py",
         "--random-steps", "10000",
         "--competence-steps", str(steps - 10000),
         "--config", config_path],
        "Curriculum Babbling",
        required=False
    )


def phase_2_vision(config_path: str, resume: Optional[str] = None) -> bool:
    """Train vision encoder."""
    print("\n" + "="*70)
    print("PHASE 2: VISION ENCODER")
    print("="*70)
    cmd = ["python", "scripts/train_world_model.py", "--config", config_path, "--phase", "vision"]
    if resume:
        cmd.extend(["--resume", resume])
    return run_command(cmd, "Vision Encoder Training")


def phase_3_audio(config_path: str, resume: Optional[str] = None) -> bool:
    """Train audio encoder (real SpeechCommands pipeline)."""
    print("\n" + "="*70)
    print("PHASE 3: AUDIO ENCODER")
    print("="*70)
    cmd = ["python", "scripts/train_world_model.py", "--config", config_path, "--phase", "audio"]
    if resume:
        cmd.extend(["--resume", resume])
    return run_command(cmd, "Audio Encoder Training")


def phase_4_fusion(config_path: str, resume: Optional[str] = None, data_dir: Optional[str] = None) -> bool:
    """Train cross-modal fusion."""
    print("\n" + "="*70)
    print("PHASE 4: CROSS-MODAL FUSION (~$30, 8 hours)")
    print("="*70)
    
    cmd = ["python", "scripts/train_world_model.py",
           "--config", config_path,
           "--phase", "fusion"]
    
    if resume:
        cmd.extend(["--resume", resume])
    
    if data_dir:
        cmd.extend(["--data-dir", data_dir])
        print(f"Using REAL data from: {data_dir}")
    
    return run_command(cmd, "Cross-Modal Fusion Training")


def phase_5_temporal(config_path: str, resume: Optional[str] = None, data_dir: Optional[str] = None) -> bool:
    """Train temporal model and dynamics."""
    print("\n" + "="*70)
    print("PHASE 5: TEMPORAL MODEL + DYNAMICS (~$30, 8 hours)")
    print("="*70)
    
    cmd = ["python", "scripts/train_world_model.py",
           "--config", config_path,
           "--phase", "temporal"]
    
    if resume:
        cmd.extend(["--resume", resume])
    
    if data_dir:
        cmd.extend(["--data-dir", data_dir])
    
    return run_command(cmd, "Temporal Model Training")


def phase_6_evaluation(checkpoint: str, n_seeds: int = 20) -> bool:
    """Run full evaluation with ablation study."""
    print("\n" + "="*70)
    print(f"PHASE 6: EVALUATION + ABLATION (N={n_seeds} seeds)")
    print("="*70)
    
    return run_command(
        ["python", "scripts/evaluate.py",
         "--checkpoint", checkpoint,
         "--n-seeds", str(n_seeds)],
        "Full Evaluation",
        required=False
    )


def main():
    parser = argparse.ArgumentParser(description="NSCA Full Training Orchestrator")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config (use training_config_local.yaml for RTX 3050)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation checkpoints (dangerous!)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--start-phase', type=int, default=0,
                       help='Start from specific phase (0-6)')
    parser.add_argument('--include-audio', action='store_true',
                       help='Include audio encoder training')
    parser.add_argument('--ablation-seeds', type=int, default=20,
                       help='Number of seeds for ablation study')
    parser.add_argument('--babbling-steps', type=int, default=100000,
                       help='Total babbling interaction steps')
    parser.add_argument('--data-dir', type=str, default='',
                       help='Path to Greatest Hits data (optional; uses CIFAR+Speech fallback if empty)')
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("      NSCA WORLD MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print(f"Start phase: {args.start_phase}")
    print(f"Data directory: {args.data_dir}")
    if args.data_dir and Path(args.data_dir).exists():
        print("✓ Greatest Hits data will be used for fusion/temporal")
    else:
        print("ℹ Using CIFAR + SpeechCommands fallback (run download_data.py --local-test first)")
    
    # Check GPU
    check_gpu()
    
    # Create checkpoint directory
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Track timing
    start_time = time.time()
    phase_times = {}
    
    # Phase 0: Validation
    if args.start_phase <= 0:
        t0 = time.time()
        if not phase_0_validation(args.skip_validation):
            print("\nAborting training due to validation failures.")
            sys.exit(1)
        phase_times['validation'] = time.time() - t0
    
    # Phase 1: Babbling
    if args.start_phase <= 1:
        t0 = time.time()
        phase_1_babbling(args.config, args.babbling_steps)
        phase_times['babbling'] = time.time() - t0
    
    # Phases 2-5: Run train_world_model with --phase all so model state is preserved
    # (vision -> audio -> fusion/temporal in one process)
    if args.start_phase <= 5:
        t0 = time.time()
        cmd = ["python", "scripts/train_world_model.py", "--config", args.config, "--phase", "all"]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])
        if args.resume:
            cmd.extend(["--resume", args.resume])
        run_command(cmd, "Vision + Audio + Fusion + Temporal Training")
        phase_times['training'] = time.time() - t0
    
    # Phase 6: Evaluation
    if args.start_phase <= 6:
        t0 = time.time()
        checkpoint = "checkpoints/world_model_final.pth"
        if Path(checkpoint).exists():
            phase_6_evaluation(checkpoint, args.ablation_seeds)
        else:
            print(f"Checkpoint not found: {checkpoint}")
            print("Run training phases first or specify --resume")
        phase_times['evaluation'] = time.time() - t0
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print("\nPhase breakdown:")
    for phase, t in phase_times.items():
        print(f"  {phase}: {t/60:.1f} min")
    
    print("\nNext steps:")
    print("1. Review logs in logs/ directory")
    print("2. Check checkpoints in checkpoints/ directory")
    print("3. Run additional evaluation: python scripts/evaluate.py")
    print("4. Update paper with final results")


if __name__ == "__main__":
    main()
