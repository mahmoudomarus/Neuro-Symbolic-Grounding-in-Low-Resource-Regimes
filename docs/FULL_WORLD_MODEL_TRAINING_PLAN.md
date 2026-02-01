# NSCA Full World Model Training Plan

## Executive Summary

The physics prior experiment has **validated the core hypothesis**:
- **+7.2% accuracy** at N=20 samples
- **+5.5% accuracy** at N=50 samples
- **Convergence at high data** (as expected - priors help sample efficiency, not asymptotic performance)
- **Adaptive prior weight** (0.49 → 0.35-0.41) shows correct learning behavior

Now we proceed to train the **complete world model** covering all architectural components:

| Component | Status | Validated |
|-----------|--------|-----------|
| Physics Priors (AdaptivePhysicsPrior) | ✅ Implemented | ✅ Yes (+7.2% sample efficiency) |
| Multi-Sensory Fusion | ✅ Implemented | ⏳ Pending |
| Dynamic Property Discovery | ✅ Implemented | ⏳ Pending |
| Babbling Phase (Grounding) | 🔶 Partial | ⏳ Pending |
| Dual Memory System | ✅ Implemented | ⏳ Pending |
| Continual Learning (EWC) | ✅ Implemented | ⏳ Pending |
| Robust Curiosity | ✅ Implemented | ⏳ Pending |
| Adversarial Robustness | ✅ Implemented | ⏳ Pending |

---

## Phase 0: Pre-Training Validation Checkpoints

Before investing in full training ($150+), run these cheap validation tests (~$15 total):

### Checkpoint 0.1: Noisy TV Test ($2, 30 min)
```bash
python scripts/noisy_tv_test.py --episodes 50 --seed 42
```

**Pass Criteria:**
- TV visits in second half < 20% of first half
- Learnability score < 0.1 for TV states

### Checkpoint 0.2: Forgetting Test ($5, 1 hour)
```bash
# Train on Task A, then Task B, test Task A
python scripts/forgetting_test.py --ewc-weight 1000
```

**Pass Criteria:**
- Without EWC: Task A drops to ~30%
- With EWC: Task A stays at ~70%

### Checkpoint 0.3: Prior Override Test ($3, 45 min)
```bash
python scripts/balloon_test.py --anti-gravity --steps 1000
```

**Pass Criteria:**
- prior_weight drops from 0.9 → ~0.35
- Model predicts upward motion after training

### Checkpoint 0.4: Slot Discovery Test ($5, 1 hour)
```bash
python scripts/slot_discovery_test.py --free-slots 16 --interactions 5000
```

**Pass Criteria:**
- At least 3 free slots activate (activation > 0.5)
- Discovered slots correlate with distinct properties (r > 0.6)

---

## Phase 1: Babbling Phase Training (Core Innovation)

This is where NSCA fundamentally differs from standard models. Instead of supervised labels, the agent **discovers concepts through interaction**.

### 1.1 Environment Setup

```python
# Install dependencies
pip install metaworld mujoco gymnasium

# Verify MuJoCo works
python -c "import mujoco; print('MuJoCo version:', mujoco.mj_version())"
```

### 1.2 Babbling Protocol

| Phase | Steps | Mode | Objects |
|-------|-------|------|---------|
| Random | 10,000 | Constrained exploration | Set A (20 objects) |
| Competence-Driven | 90,000 | Retry learnable interactions | Set A |
| Evaluation | 1,000 | Zero-shot test | Set B (10 novel objects) |

**Key: Forced Interaction Initialization**
```python
SPAWN_MODES = {
    'drop': 'gripper_holding_object',      # Already grasping
    'strike': 'gripper_inches_from_object', # Close enough to hit
    'push': 'gripper_touching_object',      # Contact established
}
```

### 1.3 What Gets Learned

| Property | Sensory Channel | How Learned |
|----------|-----------------|-------------|
| Hardness | Audio (impact spectrum) | Strike objects, analyze frequency |
| Weight | Visual (motion resistance) | Push/lift, observe acceleration |
| Size | Visual (bounding box) | View from multiple angles |
| Fragility | Audio + Visual | Strike and observe outcomes |
| Elasticity | Visual (bounce dynamics) | Drop objects |

### 1.4 Grounding Protocol

**Empty initialization** - No pre-defined concept vectors:
```python
# WRONG (old approach)
CONCEPT_GROUNDINGS = {"rock": [0.9, 0.7, ...]}  # DELETED

# RIGHT (new approach)
grounding_table = {}  # Populated ONLY through interaction
```

**Online Grounding** (during babbling):
```python
if utterance_available:
    slot_loss = contrastive_align(active_slot, utterance_embedding)
```

**Post-hoc Grounding** (after training):
```python
examples = get_high_activation_examples(slot_idx=9, k=10)
slot_name = llm.describe(examples)  # "sticky/tacky objects"
```

---

## Phase 2: Vision Encoder Training

### 2.1 Dataset Selection

| Dataset | Size | Purpose | Download |
|---------|------|---------|----------|
| Something-Something v2 | 220K | Action recognition | HuggingFace |
| Kinetics-400 | 300K | Temporal dynamics | HuggingFace |

**Skip**: ImageNet (static), Epic-Kitchens (messy audio alignment)

### 2.2 Training Configuration

```yaml
training:
  vision:
    epochs: 100
    batch_size: 256
    learning_rate: 0.0003
    priors:
      gabor: true       # 8 orientations × 4 scales
      color_opponency: true  # RG, BY channels
      depth_cues: true  # Monocular depth estimation
```

### 2.3 Loss Functions

1. **Contrastive (NT-Xent)**: Learn invariant representations
2. **VICReg**: Prevent collapse without negatives
3. **Temporal Consistency**: Adjacent frames should be similar

---

## Phase 3: Audio Encoder Training (Defer to v2.1)

Audio adds complexity but is **not required for core validation**. Proceed with vision + proprioception first.

**When ready:**
| Dataset | Size | Purpose |
|---------|------|---------|
| Greatest Hits | 46K | Impact sounds (striking) |
| Synthetic | Generated | MuJoCo collision impulses |

---

## Phase 4: Cross-Modal Fusion

### 4.1 Architecture

```
Vision → [256D] ─┐
                 ├─→ Cross-Attention → Unified State [512D]
Proprio → [256D] ─┘
```

### 4.2 Training

- Freeze encoders
- Train attention weights only
- Use video data with joint positions (Something-Something v2)

---

## Phase 5: Temporal World Model + Dynamics

### 5.1 Next-State Prediction

```python
# Given: states[t-N:t], action[t]
# Predict: state[t+1]

dynamics_loss = MSE(predicted_state, actual_state)
uncertainty_loss = -log(predicted_uncertainty) where error > threshold
```

### 5.2 Adaptive Physics Integration

```python
# The prior provides a base prediction
prior_prediction = GravityPrior(state)  # Downward bias

# Learned correction handles exceptions
correction = dynamics_net(state)

# Blend with learnable weight (min 0.3)
effective_weight = 0.3 + F.softplus(prior_weight - 0.3)
prediction = effective_weight * prior_prediction + (1 - effective_weight) * correction
```

---

## Phase 6: Memory Consolidation

### 6.1 Dual Memory System

| Memory Type | Purpose | Update Frequency |
|-------------|---------|------------------|
| Episodic | Recent experiences | Every step |
| Semantic | Consolidated concepts | Every 1000 steps |

### 6.2 EWC Protection

```python
# Compute Fisher Information Matrix
fisher = compute_fisher(model, validation_data)

# Higher protection for semantic (10x)
ewc_loss = sum(fisher[param] * (param - old_param)**2)
```

---

## Phase 7: Full Evaluation

### 7.1 Benchmarks

| Benchmark | Tests | Target | Baseline |
|-----------|-------|--------|----------|
| Meta-World ML10 | Few-shot manipulation | 65% @ 5 demos | 22% random |
| Physion | Intuitive physics | 75-85% | Human 89% |
| Procgen | Generalization | Competitive with IMPALA | - |

### 7.2 Ablation Study Protocol

```bash
python scripts/run_ablation.py \
  --conditions "with_priors,without_priors,without_ewc,without_curiosity" \
  --n-seeds 20 \
  --tasks "pick-place-v2,push-v2,reach-v2"
```

**Statistical Requirements:**
- N = 20 seeds per condition
- Report: Mean ± SE, Cohen's d effect size
- Learning curves (not just final performance)

---

## Compute Requirements

### Minimum (Development)
| Component | Spec |
|-----------|------|
| GPU | RTX 3090/4090 (24GB) |
| CPU | 16 cores (EPYC/Ryzen) |
| RAM | 64GB |
| Storage | 500GB SSD |

### Recommended (Full Training)
| Component | Spec |
|-----------|------|
| GPU | A100 40GB |
| CPU | 32+ cores |
| RAM | 128GB |
| Storage | 1TB NVMe |

**Critical: CPU matters for babbling (MuJoCo is CPU-bound)**

---

## Cost Estimate

| Phase | Time | Cost (Vast.ai) |
|-------|------|----------------|
| Pre-validation (0.1-0.4) | 4h | $15 |
| Babbling | 10h | $30 |
| Vision Training | 12h | $50 |
| Fusion + Temporal | 8h | $30 |
| Ablation Study (N=20) | 24h | $75 |
| **Total** | **~60h** | **$200** |

---

## Quick Start Script

```bash
#!/bin/bash
# full_training.sh

# 0. Validation checkpoints
echo "Running pre-validation..."
python scripts/noisy_tv_test.py --episodes 50
python scripts/forgetting_test.py --ewc-weight 1000
python scripts/balloon_test.py --steps 1000
python scripts/slot_discovery_test.py --free-slots 16

# 1. Babbling phase
echo "Starting babbling phase..."
python scripts/run_babbling.py \
  --random-steps 10000 \
  --competence-steps 90000 \
  --object-set-a configs/objects_set_a.yaml \
  --output checkpoints/babbling_complete.pth

# 2. Vision encoder
echo "Training vision encoder..."
python scripts/train_world_model.py \
  --phase vision \
  --config configs/training_config.yaml \
  --wandb-project nsca-training

# 3. Fusion + temporal
echo "Training fusion and temporal..."
python scripts/train_world_model.py \
  --phase fusion \
  --resume checkpoints/vision_encoder_final.pth \
  --config configs/training_config.yaml

# 4. Evaluation
echo "Running evaluation..."
python scripts/evaluate.py \
  --checkpoint checkpoints/world_model_final.pth \
  --benchmarks metaworld,physion \
  --n-seeds 20

echo "Training complete!"
```

---

## Expected Results

### Sample Efficiency Curve (Target)

```
Test Accuracy vs Training Samples

        │
   100% │                    ●─────────────● NSCA
        │                 ●─────────────────● Baseline
    80% │           ●─────────────────
        │        ●
    60% │     ●
        │  ●    ← Prior advantage here (low-data regime)
    40% │
        └─────┬─────┬─────┬─────┬─────┬────→
              5    10    50   100  500   Demos
```

### Prior Weight Adaptation (Target)

```
prior_weight vs Training Steps

  1.0 │●
      │ ●
  0.8 │  ●
      │   ●●
  0.6 │     ●●
      │       ●●
  0.4 │         ●●──────────●  (floor at 0.3)
  0.3 │─────────────────────── (critical period protection)
      └────┬────┬────┬────┬────→
           1K   5K  10K  50K   Steps
```

---

## Failure Modes and Mitigations

| Failure | Detection | Fix |
|---------|-----------|-----|
| Slot collapse | All slots identical | Increase orthogonal init, add diversity loss |
| Curiosity stalling | No new states visited | Lower learnability threshold |
| EWC too strong | New tasks won't learn | Reduce Fisher weight |
| Prior won't override | prior_weight stays high | Check softplus gradient flow |

---

## What Success Looks Like

After full training:

1. **Few-shot manipulation**: 65% at 5 demos (vs 22% baseline)
2. **Physics prediction**: 75-85% on Physion
3. **Novel object generalization**: Set B performance within 10% of Set A
4. **Memory retention**: <10% forgetting on Task A after Task B
5. **Property discovery**: 5+ free slots with meaningful activations

---

## Next Steps

1. **Immediate**: Run pre-validation checkpoints (Phase 0)
2. **Day 1-2**: Complete babbling phase
3. **Day 3-4**: Vision encoder training
4. **Day 5**: Fusion + temporal model
5. **Day 6-7**: Full evaluation + ablation study

**Total timeline: ~1 week with dedicated GPU access**
