# NSCA Cloud Deployment Guide

Complete guide for training NSCA on cloud GPU instances.

## TL;DR Quick Start

```bash
# On cloud instance:
git clone https://github.com/your-username/NSCA.git
cd NSCA
pip install -r requirements.txt
pip install metaworld mujoco gymnasium wandb

# Login to WandB
wandb login

# Run full training
python scripts/run_full_training.py --config configs/training_config.yaml
```

---

## 1. Choosing a Cloud Provider

### Recommended: Vast.ai (Best Value)

**Why**: Cheapest GPUs, community marketplace, good for research

| GPU | Price/Hour | VRAM | CPU Cores | Notes |
|-----|------------|------|-----------|-------|
| RTX 4090 | $0.35-0.50 | 24GB | 8-16 | Best value |
| RTX 3090 | $0.25-0.35 | 24GB | 8-16 | Budget option |
| A100 40GB | $1.50-2.00 | 40GB | 16-32 | Faster training |

**Critical**: Filter for machines with **16+ CPU cores** (MuJoCo is CPU-bound)

### Setup on Vast.ai

1. Create account at https://vast.ai
2. Add payment method
3. Go to "Search" → Filter:
   - GPU: RTX 4090 or A100
   - **CPU: ≥16 cores** (important!)
   - RAM: ≥64GB
   - Storage: ≥200GB
   - Disk Speed: ≥500 MB/s (SSD)

4. Select instance → "Rent"
5. Choose "SSH" or "Jupyter" access

### Alternative: RunPod

**Why**: More reliable, easier UI, good support

| GPU | Price/Hour | Notes |
|-----|------------|-------|
| RTX 4090 | $0.44 | Reliable |
| A100 40GB | $1.99 | Fast |

Setup at https://runpod.io

### Alternative: Lambda Labs

**Why**: ML-focused, pre-configured environments

| GPU | Price/Hour | Notes |
|-----|------------|-------|
| A10 | $1.10 | Good balance |
| A100 | $2.00 | Fastest |

Setup at https://lambdalabs.com/cloud

---

## 2. Instance Setup Script

Save this as `setup_instance.sh` and run on your cloud instance:

```bash
#!/bin/bash
# NSCA Cloud Instance Setup
# Run: bash setup_instance.sh

set -e  # Exit on error

echo "=============================================="
echo "NSCA Cloud Instance Setup"
echo "=============================================="

# System updates
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq git htop tmux

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Clone repository
echo "Cloning NSCA..."
if [ ! -d "NSCA" ]; then
    git clone https://github.com/your-username/NSCA.git
fi
cd NSCA

# Create virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install NSCA dependencies
echo "Installing NSCA dependencies..."
pip install -r requirements.txt

# Install simulation dependencies
echo "Installing MuJoCo and MetaWorld..."
pip install mujoco gymnasium metaworld

# Install monitoring tools
echo "Installing WandB..."
pip install wandb

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python verify_world_model.py

echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. wandb login"
echo "2. python scripts/run_full_training.py"
```

---

## 3. Training Commands

### Quick Test (5 minutes)

```bash
# Verify everything works
python scripts/run_full_training.py --start-phase 0 --skip-validation
```

### Full Training (~60 hours)

```bash
# Run in tmux for persistence
tmux new -s nsca

# Start training with WandB logging
python scripts/run_full_training.py \
    --config configs/training_config.yaml \
    --babbling-steps 100000 \
    --ablation-seeds 20

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t nsca
```

### Phase-by-Phase (for debugging)

```bash
# Phase 0: Validation only
python scripts/run_full_training.py --start-phase 0

# Phase 1: Babbling
python scripts/run_full_training.py --start-phase 1

# Phase 2: Vision encoder
python scripts/run_full_training.py --start-phase 2

# Phase 4-5: Fusion + Temporal
python scripts/run_full_training.py --start-phase 4

# Phase 6: Evaluation
python scripts/run_full_training.py --start-phase 6
```

---

## 4. Monitoring Training

### WandB Dashboard

1. Go to https://wandb.ai
2. Find your project: `nsca-world-model`
3. Monitor:
   - Training loss curves
   - Prior weight adaptation
   - Slot activations
   - Memory usage

### Terminal Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# CPU usage (important for MuJoCo)
htop

# Disk space
df -h

# Training logs
tail -f logs/training.log
```

### Check Progress

```bash
# List checkpoints
ls -la checkpoints/

# Check latest metrics
python -c "
import torch
ckpt = torch.load('checkpoints/latest.pth')
print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'Loss: {ckpt.get(\"loss\", \"N/A\")}')
"
```

---

## 5. Saving and Downloading Results

### Create Results Archive

```bash
# Create archive of important files
tar -czvf nsca_results.tar.gz \
    checkpoints/*.pth \
    logs/ \
    wandb/ \
    --exclude="*.tmp"
```

### Download via SCP

```bash
# From your local machine:
scp -r user@instance-ip:/root/NSCA/checkpoints ./local_checkpoints/
scp user@instance-ip:/root/NSCA/nsca_results.tar.gz ./
```

### Upload to HuggingFace Hub

```bash
# On instance:
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/',
    repo_id='your-username/nsca-checkpoints',
    repo_type='model'
)
"
```

---

## 6. Cost Optimization Tips

### 1. Use Spot Instances (Vast.ai)

- Enable "Interruptible" for ~50% discount
- Your training saves checkpoints, so interruption just means resume

### 2. Start Small

```bash
# Test with 10% data first
python scripts/download_data.py --subset 0.1
python scripts/train_world_model.py --data-fraction 0.1 --epochs 10
```

### 3. Monitor and Stop Early

```bash
# If validation loss plateaus, stop training
# Use WandB alerts for automatic notification
```

### 4. Use Preemptible A100s

On Lambda/GCP, preemptible A100s are much cheaper but can be terminated.
NSCA saves checkpoints every epoch, so this is safe.

---

## 7. Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_world_model.py --batch-size 16

# Enable gradient checkpointing
# Edit configs/training_config.yaml:
# training.general.gradient_checkpointing: true
```

### MuJoCo Slow (CPU-bound)

```bash
# Check CPU usage
htop

# If CPU is at 100%, babbling is bottlenecked
# Solution: Rent instance with more CPU cores

# Reduce babbling parallelism
python scripts/run_full_training.py --babbling-workers 4
```

### WandB Connection Issues

```bash
# Use offline mode
wandb offline

# Sync later
wandb sync ./wandb/offline-run-*
```

### Instance Terminated (Spot)

```bash
# Resume from checkpoint
python scripts/run_full_training.py \
    --resume checkpoints/latest.pth \
    --start-phase 2
```

---

## 8. Recommended Training Timeline

| Day | Phase | Hours | Cost |
|-----|-------|-------|------|
| 1 | Setup + Validation | 4h | $2 |
| 1 | Babbling | 10h | $5 |
| 2 | Vision Encoder | 12h | $6 |
| 3 | Fusion + Temporal | 12h | $6 |
| 4 | Ablation Study | 24h | $12 |
| **Total** | | **62h** | **$31** (RTX 4090) |

**With A100**: ~40h total, ~$80

---

## 9. Post-Training Checklist

- [ ] Download all checkpoints
- [ ] Export WandB runs
- [ ] Save babbling logs for audit
- [ ] Run final evaluation on test set
- [ ] Generate training report

```bash
# Generate final report
python scripts/generate_report.py \
    --checkpoint checkpoints/world_model_final.pth \
    --wandb-run your-run-id \
    --output paper/supplementary/training_report.pdf
```

---

## 10. Quick Reference

### SSH to Instance

```bash
# Vast.ai
ssh -p PORT root@INSTANCE_IP -i ~/.ssh/id_rsa

# RunPod
ssh root@POD_IP -i ~/.ssh/id_rsa
```

### Jupyter Access

```bash
# Forward port
ssh -L 8888:localhost:8888 root@INSTANCE_IP

# Then open: http://localhost:8888
```

### File Transfer

```bash
# Upload
scp local_file.py root@INSTANCE_IP:/root/NSCA/

# Download
scp root@INSTANCE_IP:/root/NSCA/checkpoints/best.pth ./
```

### Emergency Stop

```bash
# Kill training
pkill -f train_world_model

# Or from WandB dashboard: "Stop run"
```

---

## Support

- **Issues**: Open GitHub issue with logs
- **WandB**: Check run logs for errors
- **Cloud provider**: Use their support chat

Good luck with training! 🚀
