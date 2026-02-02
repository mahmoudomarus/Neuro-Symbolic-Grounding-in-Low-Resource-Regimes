#!/bin/bash
# =============================================================================
# NSCA Enhanced Training Script
# =============================================================================
# This script runs the enhanced multi-modal training with:
# - Refined data augmentation (physics-aware)
# - Hierarchical cross-modal fusion
# - Automatic HuggingFace upload
#
# Prerequisites:
# 1. Set HF_TOKEN environment variable
# 2. Download training data to data/ directory
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "NSCA Enhanced Training v2.0"
echo "=============================================="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set. Auto-upload will be disabled.${NC}"
    echo "To enable auto-upload, run: export HF_TOKEN=your_token_here"
    AUTO_UPLOAD=""
else
    echo -e "${GREEN}HF_TOKEN found. Auto-upload enabled.${NC}"
    AUTO_UPLOAD="--auto-upload"
fi

# Default parameters
DATA_DIR="${DATA_DIR:-/workspace/vis-data}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
HF_REPO="${HF_REPO:-omartabius/NSCA}"
CONFIG="${CONFIG:-./configs/training_config.yaml}"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: Data directory not found: $DATA_DIR${NC}"
    echo ""
    echo "Please download the training data first:"
    echo "  1. Greatest Hits dataset"
    echo "  2. Or specify a different data directory with DATA_DIR=/path/to/data"
    echo ""
    exit 1
fi

# Create save directory
mkdir -p "$SAVE_DIR"

echo ""
echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Device: $DEVICE"
echo "  Save Directory: $SAVE_DIR"
echo "  HuggingFace Repo: $HF_REPO"
echo "  Config: $CONFIG"
echo ""

# Run training
echo "Starting training..."
echo ""

# Check if HF upload should be disabled
if [ "$NO_HF_UPLOAD" = "1" ]; then
    echo -e "${YELLOW}HuggingFace upload disabled (NO_HF_UPLOAD=1)${NC}"
    python scripts/train_multimodal.py \
        --data-dir "$DATA_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --device "$DEVICE" \
        --save-dir "$SAVE_DIR" \
        --config "$CONFIG" \
        --enhanced-aug \
        --physics-aware
else
    python scripts/train_multimodal.py \
        --data-dir "$DATA_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --device "$DEVICE" \
        --save-dir "$SAVE_DIR" \
        --config "$CONFIG" \
        --enhanced-aug \
        --physics-aware \
        --hf-repo "$HF_REPO" \
        $AUTO_UPLOAD \
        --upload-hf
fi

echo ""
echo -e "${GREEN}Training complete!${NC}"
echo ""
echo "Checkpoints saved to: $SAVE_DIR"
echo "Best model: $SAVE_DIR/multimodal_best.pth"
echo "Final model: $SAVE_DIR/multimodal_final.pth"

if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "Models uploaded to: https://huggingface.co/$HF_REPO"
fi
