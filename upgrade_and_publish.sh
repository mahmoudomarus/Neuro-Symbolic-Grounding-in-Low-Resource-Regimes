#!/bin/bash
# ============================================================================
# upgrade_and_publish.sh — Master script for Phase 10-12
# ============================================================================
#
# This script:
# 1. Creates dummy custom data if ./data/my_dataset doesn't exist
# 2. Retrains the encoder on custom RGB data
# 3. Retrains the binder for language grounding
# 4. Generates architecture documentation
# 5. Generates academic manuscript
# 6. Launches the Streamlit dashboard
#
# Run from project root: ./upgrade_and_publish.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  NeuroSymbolic-JEPA-Core — Upgrade & Publish"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
python -c "import torch, torchvision, streamlit" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# ============================================================================
# Step 1: Check/create custom dataset
# ============================================================================
CUSTOM_DATA_DIR="./data/my_dataset"

if [ -d "$CUSTOM_DATA_DIR" ]; then
    echo "[Step 1] Custom dataset found at $CUSTOM_DATA_DIR"
    # Count classes
    NUM_CLASSES=$(find "$CUSTOM_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    echo "         Found $NUM_CLASSES class folders"
else
    echo "[Step 1] Custom dataset not found. Creating dummy data..."
    
    # Create class directories
    mkdir -p "$CUSTOM_DATA_DIR/class_A"
    mkdir -p "$CUSTOM_DATA_DIR/class_B"
    
    # Download placeholder images from Lorem Picsum
    echo "         Downloading placeholder images..."
    
    for i in 1 2 3 4 5; do
        # Class A - different seed for variety
        curl -sL "https://picsum.photos/seed/classA${i}/100/100" -o "$CUSTOM_DATA_DIR/class_A/img_${i}.jpg" 2>/dev/null || {
            # Fallback: create simple colored images with Python
            python -c "
from PIL import Image
import random
img = Image.new('RGB', (100, 100), (random.randint(200, 255), random.randint(0, 100), random.randint(0, 100)))
img.save('$CUSTOM_DATA_DIR/class_A/img_${i}.jpg')
"
        }
        
        # Class B - different colors
        curl -sL "https://picsum.photos/seed/classB${i}/100/100" -o "$CUSTOM_DATA_DIR/class_B/img_${i}.jpg" 2>/dev/null || {
            python -c "
from PIL import Image
import random
img = Image.new('RGB', (100, 100), (random.randint(0, 100), random.randint(0, 100), random.randint(200, 255)))
img.save('$CUSTOM_DATA_DIR/class_B/img_${i}.jpg')
"
        }
    done
    
    echo "         Created dummy dataset with 2 classes, 5 images each"
fi

# ============================================================================
# Step 2: Train encoder on custom data
# ============================================================================
echo ""
echo "[Step 2] Training Vision System (Encoder)..."
echo "         This may take a few minutes..."
python src/world_model/train_encoder.py

# ============================================================================
# Step 3: Train binder for language grounding
# ============================================================================
echo ""
echo "[Step 3] Training Language Grounding (Binder)..."
python src/language/train_grounding.py

# ============================================================================
# Step 4: Generate architecture documentation
# ============================================================================
echo ""
echo "[Step 4] Generating architecture documentation..."
python scripts/generate_architecture_doc.py

# ============================================================================
# Step 5: Generate academic manuscript
# ============================================================================
echo ""
echo "[Step 5] Generating academic manuscript..."
python scripts/generate_manuscript.py

# ============================================================================
# Step 6: Launch dashboard
# ============================================================================
echo ""
echo "=============================================="
echo "  All training and generation complete!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - checkpoints/encoder_v1.pth (trained encoder)"
echo "  - checkpoints/binder_v1.pth (trained binder)"
echo "  - checkpoints/dataset_config.json (dataset configuration)"
echo "  - docs/ARCHITECTURE.md (auto-generated documentation)"
echo "  - paper/manuscript.md (academic manuscript)"
echo ""
echo "[Step 6] Launching Streamlit dashboard..."
echo "         Open http://localhost:8501 in your browser"
echo ""

streamlit run src/dashboard.py
