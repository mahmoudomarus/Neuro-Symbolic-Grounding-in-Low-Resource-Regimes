# Code Change Log

> Generated: 2026-02-23  
> Purpose: Full record of all modifications made by the AI assistant, for author review.

---

## 1. New Files Created

### `configs/training_config_local.yaml`
- **Type**: Brand new file — does not affect the original configuration
- **Contents**: Training configuration tuned for RTX 3050 (6 GB VRAM)
  - Uses small public datasets: CIFAR-100 (vision), SpeechCommands (audio)
  - Weights & Biases logging disabled by default (`use_wandb: false`)
  - Batch sizes and gradient accumulation settings are identical to the modified `training_config.yaml` described below

---

### `scripts/run_babbling.py`
- **Type**: Brand new file
- **Contents**: Curriculum babbling training script
  - Implements a random exploration phase followed by a competence-driven interaction phase
  - Saves exploration results to `logs/babbling_results.json`
  - Does **not** modify any model weights; serves purely as an evaluation / data-collection tool

---

### `train.py`
- **Type**: Brand new file (top-level launcher wrapper)
- **Contents**: Simplified entry point that auto-activates the virtual environment and assembles the command
  - Equivalent to running `scripts/train_world_model.py` manually; no additional logic

---

## 2. Modified Existing Files

### `configs/training_config.yaml`
| Parameter | Original Value | New Value | Reason |
|-----------|---------------|-----------|--------|
| `training.vision.batch_size` | 256 | **32** | 6 GB VRAM is insufficient for 256 × 224×224 images |
| `training.audio.batch_size` | 128 | **16** | Same reason |
| `training.fusion.batch_size` | 64 | **8** | Same reason |
| `training.temporal.batch_size` | 32 | **4** | Transformer layers require significantly more memory |
| `training.general.gradient_accumulation_steps` | 4 | **8** | Compensates for smaller batch size; effective batch size is preserved |
| `training.general.num_workers` | 8 | **4** | Prevents DataLoader workers from saturating the CPU |

**Model architecture, loss functions, hyperparameters (lr, weight_decay, etc.), and dataset paths were not changed.**

---

### `verify_world_model.py`
- **Location**: Line 159
- **Original code**:
  ```python
  is_supported, motion = physics.gravity(state_before)
  ```
- **Modified code**:
  ```python
  is_supported, motion, *_ = physics.gravity(state_before)
  ```
- **Reason**: `physics.gravity()` returns 3 values; the original code only unpacked 2, causing `ValueError: too many values to unpack`. The `*_` idiom discards the extra return value without changing any logic.

---

### `src/priors/audio_prior.py`
- **Location**: Inside `AuditoryPrior.forward()`
- **Change**: Wrapped the STFT and mel-filterbank computation in:
  ```python
  with torch.amp.autocast('cuda', enabled=False):
      # STFT / mel computation forced to float32 here
  ```
  Added a clamp before the log operation:
  ```python
  mel_spec = mel_spec.clamp(min=1e-9)
  log_mel  = torch.log(mel_spec)
  ```
- **Reason**: Under mixed precision (float16), `torch.matmul` can produce numerical underflow, causing `log(0) = -inf` which propagates as NaN loss. Forcing float32 in this region eliminates the NaN without changing the computation logic or output semantics.

---

### `scripts/train_world_model.py`
This file received the most changes, grouped by category:

#### 2a. PyTorch 2.x Compatibility Fixes
| Original API | Updated API |
|--------------|-------------|
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler('cuda')` |
| `torch.cuda.amp.autocast()` | `torch.amp.autocast('cuda', ...)` |

The original APIs are deprecated in PyTorch 2.x and emit warnings. The new APIs are fully equivalent.

#### 2b. Audio Encoder Training — Full Rewrite (original was a placeholder)
The original code used **random noise** as training data and had an incorrect loss computation. Changes:
- Added `AudioAugment` class: generates two augmented views of the same audio clip (time stretch, pitch shift, additive noise)
- Added `_build_audio_dataset()`: loads data in priority order:
  1. `torchaudio.datasets.SPEECHCOMMANDS` (reads locally cached WAV files via `soundfile`, no FFmpeg required)
  2. HuggingFace `common_voice` as fallback
  3. Synthetic sine-wave dataset as last resort
- Training objective changed to the correct **NT-Xent contrastive loss**
- Removed the original `min(..., 5)` epoch cap (which limited training to only 5 epochs)
- Added `if not torch.isfinite(loss): continue` to skip batches with NaN/Inf loss

#### 2c. Fusion / Temporal Synthetic Data Improvement
The original synthetic data consisted of purely random vectors with a meaningless `tensor.pow(2).mean()` loss. Changes:
- Synthetic data now generates paired vision/audio features from a shared latent code (structured pairing)
- Loss changed to **VICReg** (Variance-Invariance-Covariance Regularization) to prevent representation collapse

#### 2d. Interrupt-and-Resume System (added this session)
- Added global `_STOP_REQUESTED` flag with signal handlers for `SIGINT` and `SIGTERM`
- Added `save_full_checkpoint()`: saves the complete training state (model weights + optimizer state + scheduler state + current epoch + phase name)
- Added `load_full_checkpoint()`: restores the complete training state
- All three training functions (`train_vision_encoder` / `train_audio_encoder` / `train_fusion_and_temporal`) now accept:
  - `start_epoch` parameter (resume from a specific epoch)
  - `resume_ckpt` parameter (load the corresponding checkpoint file)
  - Auto-save to `*_last.pth` and `*_best.pth` every N epochs or on interrupt
- `main()` now auto-detects the latest checkpoint file and resumes automatically (no need to pass `--resume` manually)

---

## 3. Files That Were NOT Modified

The following core files were **left completely untouched**:
- `src/world_model/unified_world_model.py` — main model architecture
- `src/encoders/vision_encoder.py` — vision encoder definition
- `src/encoders/audio_encoder.py` — audio encoder definition
- `src/world_model/temporal_world_model.py` — temporal model
- `src/cognitive_agent.py` — cognitive agent
- `src/priors/visual_prior.py` — visual priors
- `requirements.txt` — dependency list
- All test scripts (`noisy_tv_test.py`, `forgetting_test.py`, `balloon_test.py`, etc.)

---

## 4. Verification Commands

To verify any individual change:
```bash
# Check audio_prior NaN fix
grep -n "autocast\|enabled=False\|clamp" src/priors/audio_prior.py

# Check verify_world_model unpack fix
grep -n "gravity" verify_world_model.py

# Check config batch sizes
grep -n "batch_size\|gradient_accumulation\|num_workers" configs/training_config.yaml

# Check train_world_model signal handling and checkpoint functions
grep -n "_STOP_REQUESTED\|save_full_checkpoint\|load_full_checkpoint" scripts/train_world_model.py
```
