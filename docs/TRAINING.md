# Training and Evaluation Guide

## Overview

Training the NSCA cognitive architecture involves multiple phases, each targeting specific components of the layered system.

**v2.0 Update**: Training now includes a mandatory **babbling phase** for grounded concept learning and an **ablation study** protocol for rigorous evaluation.

---

## Hardware Requirements

| Configuration | GPU Memory | Training Time (approx) |
|--------------|------------|------------------------|
| Minimum | 8GB VRAM | 24-48 hours |
| Recommended | 16GB VRAM | 12-24 hours |
| Optimal | 40GB+ VRAM (A100) | 4-8 hours |

---

## Critical Changes in v2.0

### 1. Babbling Phase (REQUIRED)

Before evaluation, the agent must complete a **babbling phase** to ground concepts:

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --babbling-steps 10000
```

**Why**: We removed all hard-coded concept groundings. The agent must learn "rock = hard" through striking interactions, not manual dictionaries.

### 2. Adaptive Physics Training

When training on physics exceptions (balloons, magnets), monitor `prior_weight`:

```python
# Expected behavior during balloon training
# Step 0: prior_weight ≈ 0.9 (trusts gravity prior)
# Step 500: prior_weight ≈ 0.5 (uncertain)
# Step 1000: prior_weight ≈ 0.35 (correction dominates, but floor respected)
```

### 3. EWC for Continual Learning

When training on multiple tasks sequentially:

```python
from src.learning.ewc import create_ewc_for_nsca

ewc = create_ewc_for_nsca(model, importance_weight=1000, semantic_multiplier=10.0)

# After Task A
ewc.consolidate(task_a_dataloader)

# During Task B
loss = task_b_loss + ewc.penalty()
```

---

## Training Phases

### Phase 1: Vision Encoder Pre-training

Train the vision encoder using contrastive learning (SimCLR-style).

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase vision
```

**Objective**: Learn visual features that cluster similar concepts together.

**Loss Function**:
```
L_vision = NT-Xent(z_i, z_j) + λ_var * Var_loss
```

Where:
- `z_i, z_j` are embeddings of augmented views of the same image
- `Var_loss` prevents representation collapse

**Data Augmentations**:
- Random resized crop
- Random horizontal flip
- Color jitter
- Random grayscale

### Phase 2: Audio Encoder Pre-training

Train the audio encoder using contrastive learning on audio data.

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase audio
```

**Objective**: Learn audio features that cluster similar sounds.

**Data Augmentations**:
- Time shifting
- Pitch shifting
- Time stretching
- Background noise addition

### Phase 3: Multi-Modal Fusion Training

Train the cross-modal fusion and temporal model.

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase fusion
```

**Objective**: Learn to integrate vision, audio, and proprioception into coherent world states.

**Loss Function**:
```
L_fusion = L_prediction + λ_consistency * L_consistency + λ_regularization * L_reg
```

### Phase 4: Dynamics Prediction

Train the imagination engine to predict future states.

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase dynamics
```

**Objective**: Accurate multi-step future state prediction.

**Loss Function**:
```
L_dynamics = MSE(z_predicted, z_actual) + λ_uncertainty * Uncertainty_calibration
```

### Phase 5: End-to-End Fine-tuning

Fine-tune all components jointly.

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase full
```

### Phase 6: Babbling Phase (v2.0 - REQUIRED)

Ground concepts through sensorimotor interaction.

```bash
python scripts/train_world_model.py \
    --config configs/training_config.yaml \
    --phase babbling \
    --babbling-steps 10000
```

**Objective**: Populate the grounding table through interaction, not manual supervision.

**Protocol**:
```
Phase 6.1 (Steps 1-1000): Random Exploration
  - Randomly select actions from available set
  - Build initial affordance coverage
  - Discover action-effect contingencies

Phase 6.2 (Steps 1001-10000): Competence-Driven
  - Retry actions that showed high learnability
  - Skip actions that are unlearnable (noisy TV defense)
  - Focus on interactions that reduce prediction error
```

**Grounding Actions**:
| Action | Property Learned | Signal |
|--------|-----------------|--------|
| strike | hardness | audio_frequency |
| lift | weight | force_required, acceleration |
| push | weight | resistance, displacement |
| squeeze | hardness, rigidity | deformation |
| drop | hardness | impact_sound |

**Verification**:
```bash
# Verify babbling produced valid groundings
python -c "
from src.language.llm_integration import LearnedGrounding
grounder = LearnedGrounding()
# Load from checkpoint...
print(f'Grounded concepts: {len(grounder.get_grounded_concepts())}')
print(f'Stats: {grounder.get_grounding_statistics()}')
"
```

---

## Configuration

### `configs/training_config.yaml`

```yaml
# Model Configuration
model:
  latent_dim: 256
  state_dim: 128
  action_dim: 16
  
  vision:
    input_height: 64
    input_width: 64
    base_channels: 32
    
  audio:
    sample_rate: 16000
    n_mels: 80
    
  fusion:
    num_heads: 8
    num_layers: 4
    
  temporal:
    num_heads: 8
    num_layers: 4
    max_seq_len: 32

# Training Configuration
training:
  batch_size: 64
  learning_rate: 3e-4
  weight_decay: 1e-4
  warmup_steps: 1000
  max_steps: 100000
  
  # Phase-specific
  vision_epochs: 50
  audio_epochs: 30
  fusion_epochs: 50
  dynamics_epochs: 30

# Data Configuration
data:
  vision_dataset: "imagenet-1k"
  audio_dataset: "audioset"
  num_workers: 8
  prefetch_factor: 2

# Logging
logging:
  use_wandb: true
  project: "nsca-training"
  log_interval: 100
  save_interval: 1000
```

---

## Datasets

### Recommended Datasets

| Dataset | Modality | Size | Use |
|---------|----------|------|-----|
| ImageNet-1K | Vision | 1.2M images | Visual encoder |
| AudioSet | Audio | 2M clips | Audio encoder |
| Kinetics-400 | Video | 400K clips | Multi-modal |
| Something-Something | Video | 220K clips | Action understanding |
| Epic-Kitchens | Multi-modal | 100K clips | Embodied learning |

### Using HuggingFace Datasets

```python
from datasets import load_dataset

# Vision
vision_data = load_dataset("imagenet-1k", split="train")

# Audio
audio_data = load_dataset("google/audioset", split="train")

# Video
video_data = load_dataset("kinetics400", split="train")
```

### Custom Dataset Format

```
data/
├── train/
│   ├── class_0/
│   │   ├── image_001.jpg
│   │   ├── audio_001.wav
│   │   └── proprio_001.npy
│   ├── class_1/
│   └── ...
└── val/
    └── ...
```

---

## Monitoring Training

### Weights & Biases Integration

```bash
# Login
wandb login

# Start training with logging
python scripts/train_world_model.py --config configs/training_config.yaml
```

### Key Metrics to Monitor

| Phase | Metrics |
|-------|---------|
| Vision | Contrastive loss, representation variance, cluster separation |
| Audio | Contrastive loss, mel reconstruction error |
| Fusion | Cross-modal alignment, attention entropy |
| Dynamics | Prediction MSE, uncertainty calibration |
| Full | All above + downstream task performance |

### Tensorboard Alternative

```bash
tensorboard --logdir logs/
```

---

## Evaluation

### Running Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/model_best.pth
```

### Evaluation Metrics

#### 1. Zero-Shot Classification

Evaluate visual features without task-specific training:

```python
from scripts.evaluate import evaluate_zero_shot

accuracy = evaluate_zero_shot(
    model,
    test_loader,
    concept_names=['rock', 'water', 'animal', ...]
)
```

#### 2. Few-Shot Learning

Test rapid adaptation with limited examples:

```python
from scripts.evaluate import evaluate_few_shot

# 5-way 5-shot classification
accuracy = evaluate_few_shot(
    model,
    test_loader,
    n_way=5,
    n_shot=5,
)
```

#### 3. Cross-Modal Retrieval

Test vision-audio alignment:

```python
from scripts.evaluate import evaluate_cross_modal

# Image → Audio retrieval
recall_at_k = evaluate_cross_modal(
    model,
    vision_queries,
    audio_targets,
    k=[1, 5, 10]
)
```

#### 4. Dynamics Prediction

Test future state prediction accuracy:

```python
from scripts.evaluate import evaluate_dynamics

mse, uncertainty_calibration = evaluate_dynamics(
    model,
    test_sequences,
    prediction_horizon=10,
)
```

#### 5. Property Extraction

Test semantic property accuracy:

```python
from scripts.evaluate import evaluate_properties

property_correlation = evaluate_properties(
    model,
    test_data,
    ground_truth_properties,
)
```

#### 6. Causal Reasoning

Test causal understanding:

```python
from scripts.evaluate import evaluate_causal

causal_accuracy = evaluate_causal(
    model,
    intervention_data,
)
```

---

## Checkpointing

### Saving

```python
# Automatic checkpointing during training
# Saves to checkpoints/model_step_{step}.pth

# Manual save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': step,
    'config': config,
}, 'checkpoints/model_manual.pth')
```

### Loading

```python
checkpoint = torch.load('checkpoints/model_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Distributed Training

### Multi-GPU (DataParallel)

```python
model = nn.DataParallel(model)
```

### Multi-Node (DistributedDataParallel)

```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="master_ip" --master_port=12355 \
    scripts/train_world_model.py --config configs/training_config.yaml

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr="master_ip" --master_port=12355 \
    scripts/train_world_model.py --config configs/training_config.yaml
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Learning rate | 1e-5 to 1e-3 | Higher = faster but less stable |
| Batch size | 32 to 256 | Larger = better gradient estimates |
| Temperature (contrastive) | 0.1 to 1.0 | Lower = harder negatives |
| Latent dimension | 128 to 512 | Higher = more capacity |
| Number of layers | 2 to 8 | More = more capacity |

### Recommended Starting Points

```yaml
# Conservative (stable)
learning_rate: 1e-4
batch_size: 64
temperature: 0.5
latent_dim: 256

# Aggressive (faster)
learning_rate: 3e-4
batch_size: 128
temperature: 0.3
latent_dim: 512
```

---

## Troubleshooting

### Common Issues

#### 1. Representation Collapse

**Symptoms**: All embeddings become identical, loss goes to zero.

**Solutions**:
- Increase variance regularization weight
- Use larger batch sizes
- Check data augmentation pipeline

#### 2. Training Divergence

**Symptoms**: Loss explodes, NaN values.

**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Check for data issues

#### 3. Out of Memory

**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Use mixed precision training
- Reduce model size

#### 4. Slow Training

**Solutions**:
- Increase num_workers for data loading
- Use pin_memory=True
- Profile with torch.profiler
- Use compiled model (PyTorch 2.0+)

---

## Best Practices

1. **Start small**: Test with smaller model/data first
2. **Monitor early**: Check metrics in first 1000 steps
3. **Save often**: Checkpoint every epoch minimum
4. **Validate regularly**: Run validation every few epochs
5. **Track experiments**: Use wandb/tensorboard consistently
6. **Document changes**: Log hyperparameter changes
7. **Version checkpoints**: Keep best N checkpoints

---

## Ablation Study (v2.0)

### Running the Ablation Study

The ablation study compares NSCA (with adaptive priors) vs Random Initialization.

```bash
# Quick test (5 seeds, subset of tasks)
python -c "
from src.evaluation import run_ablation_study, EvaluationConfig
config = EvaluationConfig(num_seeds=5, tasks=['pick-place-v2'])
run_ablation_study(config)
"

# Full study (20 seeds, all tasks) - REQUIRED for publication
python -c "
from src.evaluation import run_ablation_study
from pathlib import Path
summary = run_ablation_study(output_path=Path('results/ablation.json'))
"
```

### Expected Results

| Demos | NSCA (priors) | Random Init | Cohen's d |
|-------|---------------|-------------|-----------|
| 5 | ~65% ± 8% | ~22% ± 6% | ~5.4 (large) |
| 10 | ~78% ± 6% | ~45% ± 9% | ~3.8 (large) |
| 100 | ~95% ± 2% | ~93% ± 3% | ~0.7 (medium) |

**Key insight**: Curves converge at high data - priors buy efficiency, not oracle performance.

### Statistical Requirements

- **N=20 seeds minimum** (required by reviewers)
- **95% confidence intervals** on all metrics
- **Cohen's d effect size** for condition comparison
- **Learning curves** (not just final numbers)

### Verification Checklist

Before claiming results:

```bash
# 1. No hard-coded groundings
grep -r "CONCEPT_GROUNDINGS" src/  # Should return nothing

# 2. Babbling log audit
# Sample 100 interactions, verify no human labels

# 3. Prior override check
# Train on balloons, verify prior_weight drops to ~0.35

# 4. Statistical rigor
# Verify N=20 seeds, 95% CI, Cohen's d reported
```
