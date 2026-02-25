# How to Run NSCA

A simple guide to train and run the Neuro-Symbolic Cognitive Architecture.

---

## Prerequisites

- **Python 3.9+**
- **PyTorch 2.0+** (with CUDA if using GPU)
- **GPU**: RTX 3050 (6 GB) or better recommended. CPU works but is slow.

---

## 1. Clone and Install

```bash
git clone https://github.com/mahmoudomarus/Neuro-Symbolic-Grounding-in-Low-Resource-Regimes.git
cd Neuro-Symbolic-Grounding-in-Low-Resource-Regimes

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Verify Installation

```bash
python verify_world_model.py
```

You should see `✓` for each component. If any fail, check that PyTorch and torchaudio are installed correctly.

---

## 3. Training

### Option A: RTX 3050 / 6 GB GPU (recommended)

Uses `configs/training_config_local.yaml` with smaller batch sizes.

```bash
python train.py
```

This runs the full pipeline: **Vision → Audio → Fusion → Temporal** in one process. Datasets (CIFAR-100, SpeechCommands) are downloaded automatically on first run.

### Option B: Cloud GPU (A100, etc.)

```bash
python scripts/train_world_model.py --config configs/training_config.yaml --phase all
```

### Option C: Single Phase

To train only one phase:

```bash
# Vision encoder only
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase vision

# Audio encoder only
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase audio

# Fusion + Temporal (requires vision and audio checkpoints from previous phases)
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase fusion --resume checkpoints/audio_encoder_epoch50.pth
```

---

## 4. Data

| Phase   | Data Source                | When Downloaded                    |
|---------|----------------------------|------------------------------------|
| Vision  | CIFAR-100                  | First vision training run          |
| Audio   | SpeechCommands (torchaudio)| First audio training run           |
| Fusion  | CIFAR + SpeechCommands     | First fusion run (fallback)        |
| Fusion* | Greatest Hits              | Only if you use `--data-dir /path/to/greatest-hits` |

\* Greatest Hits is an aligned video+audio dataset. Without it, the fallback uses CIFAR images as “video” paired with SpeechCommands. Results are better with real Greatest Hits.

### Pre-download data (optional)

```bash
python scripts/download_data.py --local-test
```

Downloads CIFAR-100 + SpeechCommands so training doesn’t wait on downloads. Training will also download them if needed.

---

## 5. Outputs

- **Checkpoints**: `checkpoints/`
  - `vision_encoder_epoch{N}.pth` — Vision phase
  - `audio_encoder_epoch{N}.pth` — Audio phase
  - `fusion_best.pth` — Best fusion model
  - `world_model_final.pth` — Final full model

- **Logs**: `logs/`

---

## 6. Demo (after training)

Once you have a trained checkpoint:

```bash
python scripts/demo_pipeline.py --checkpoint checkpoints/world_model_final.pth --data-dir /path/to/vis-data
```

If you don’t have Greatest Hits, you can still run the demo with any directory containing `*_denoised.mp4` and `*_denoised.wav` video–audio pairs.

---

## 7. Full Pipeline (validation + babbling + training)

To run the complete pipeline including validation and babbling:

```bash
python scripts/run_full_training.py --config configs/training_config_local.yaml
```

To skip validation (only if you’ve passed it before):

```bash
python scripts/run_full_training.py --config configs/training_config_local.yaml --skip-validation
```

---

## 8. Resume Training

If training stops, resume with:

```bash
python train.py --resume checkpoints/vision_encoder_epoch50.pth
```

Or, for a specific phase:

```bash
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase all --resume checkpoints/vision_encoder_epoch100.pth
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory (OOM)** | Use `configs/training_config_local.yaml`. Reduce `batch_size` further in the config if needed. |
| **SpeechCommands download fails** | Ensure `torchaudio` is installed: `pip install torchaudio` |
| **CIFAR download fails** | Run `pip install datasets` and try again. |
| **`verify_world_model.py` fails** | Ensure all dependencies are installed. Check for CUDA/GPU compatibility if using GPU. |
| **Fusion phase: “Greatest Hits failed”** | Expected if you don’t have Greatest Hits. The CIFAR+SpeechCommands fallback will be used automatically. |

---

## Summary

```bash
# Minimal run on RTX 3050
python train.py
```

That’s it. Datasets download automatically, and training runs all phases sequentially.
