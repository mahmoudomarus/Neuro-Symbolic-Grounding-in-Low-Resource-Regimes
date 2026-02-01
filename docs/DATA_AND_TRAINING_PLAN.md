# NSCA Data and Training Plan

## Overview

NSCA is a ~16M parameter multi-modal model. This is **intentionally small** compared to modern LLMs—the hypothesis is that innate priors reduce data requirements, not that we need massive scale.

**CRITICAL LEARNING**: Do NOT start with ImageNet. Start with Meta-World. Validate the core hypothesis ($5) before investing in the full pipeline ($150).

---

## 0. The Lethality Test (DO THIS FIRST)

Before any data downloading or full training, run the **Minimal Viable Experiment**:

```bash
python scripts/lethality_test.py \
  --task pick-place-v2 \
  --n-seeds 5 \
  --demos 5
```

**Success Criteria**:
| Result | Interpretation | Action |
|--------|----------------|--------|
| prior=0.9 beats prior=0.5 by >20% | Core hypothesis validated | Proceed to full training |
| prior=0.9 ≈ prior=0.5 | Prior mechanism too weak | Debug AdaptivePhysicsPrior |
| prior=0.1 beats prior=0.9 | Priors are HURTING | Redesign architecture |

**Cost**: ~$5 on Vast.ai (2-3 hours)

---

## 1. Data Requirements by Training Phase

### Phase 1: Meta-World Validation (START HERE)

| Resource | Purpose | Notes |
|----------|---------|-------|
| **Meta-World** | Test physics priors | `pip install metaworld` |
| **MuJoCo** | Physics simulation | CPU-bound, need strong CPU |

**Why start here?** Isolates physics prior from visual prior. If `AdaptivePhysicsPrior` works on state vectors, we've proven the core hypothesis without ImageNet.

### Phase 2-3: Vision (AFTER Phase 1 passes)

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| **Something-Something v2** | 220K clips | Action recognition, clean labels | HuggingFace |
| **Kinetics-400** | 300K videos | Temporal dynamics | HuggingFace: `kinetics400` |

**SKIP Epic-Kitchens** - Audio-visual alignment is terrible for our purposes.

### Phase 4: Audio (v2.1 - DEFER THIS)

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| **Greatest Hits** | 46K clips | Impact sounds (striking objects) | Cornell |
| **Synthetic Audio** | Generated | Collision impulses from MuJoCo | Self-generated |

**CRITICAL**: Do NOT use AudioSet. It contains speech, music, traffic—irrelevant for hardness learning. Greatest Hits is specifically impact sounds (drumstick hitting materials).

**Recommendation**: Start vision-only + proprioception. Add audio in v2.1. The physics prior works without audio.

### Phase 5: Babbling (Interaction-Based Learning)

**This is where NSCA differs from standard models.**

| Environment | Purpose | Source |
|-------------|---------|--------|
| **Meta-World** | Robotic manipulation | `pip install metaworld` |
| **MuJoCo** | Physics simulation | `pip install mujoco` |

**CRITICAL FIX - Forced Interaction Initialization**:

The "Sparse Reward Problem": Random exploration hits objects ~1% of the time. 99% of data is the arm waving in empty space (useless).

**Solution**: Constrained spawning
```python
# DON'T: Spawn arm far from object (99% airballs)
# DO: Spawn gripper already holding/near object

spawn_modes = {
    'drop': 'gripper_holding_object',      # For drop babbling
    'strike': 'gripper_inches_from_object', # For strike babbling  
    'push': 'gripper_touching_object',      # For push babbling
}
```

**Constrained Action Space** (use primitives, NOT raw torques):
```python
# DON'T: Allow continuous joint torques (agent will never find meaningful actions)
# DO: Use affordance primitives
ACTIONS = ['reach', 'grasp', 'lift', 'shake', 'drop', 'strike', 'push']
```

**Babbling generates its own data**:
- 10,000 random interactions (Phase 1: constrained exploration)
- 90,000 competence-driven interactions (Phase 2: retry learnable actions)

**TIME WARNING**: Babbling is CPU-bound (MuJoCo), not GPU-bound.
- With rendering: ~10 FPS → 100K steps = **2.7 hours**
- Budget **6-10 hours** for babbling data generation

### Phase 7: Evaluation

| Benchmark | Tests | Baseline |
|-----------|-------|----------|
| **Meta-World ML10** | Few-shot manipulation | Compare to SAC, PPO |
| **Physion** | Intuitive physics | Human baseline: 89% |
| **Procgen** | Generalization | Compare to IMPALA |

---

## 2. Data Size Estimates

```
Total Training Data:
├── Visual:     ~50GB (ImageNet + Kinetics subset)
├── Audio:      ~20GB (AudioSet + VGGSound subset)
├── Simulation: Generated on-the-fly (no storage)
└── Total:      ~70GB static + simulation
```

**This is manageable** on a single workstation or cloud GPU.

---

## 3. Compute Requirements

### CRITICAL: CPU Matters for Babbling

MuJoCo runs on **CPU**, not GPU. If you rent a machine with weak CPU, the GPU will sit idle waiting for physics simulation.

**When renting on Vast.ai/RunPod**: Pick instances with **AMD EPYC** or equivalent, not just good GPU.

### Minimum Viable Setup

| Component | Spec | Cost/Hour | Notes |
|-----------|------|-----------|-------|
| GPU | RTX 4090 (24GB) | ~$0.40 (Vast.ai) | For neural net training |
| **CPU** | **16+ cores (EPYC)** | - | **CRITICAL for MuJoCo** |
| RAM | 64GB | - | - |
| Storage | 200GB SSD | - | - |

**Estimated Training Time** (on RTX 4090 + good CPU):
- Lethality test: 2-3 hours
- Babbling data generation: **6-10 hours** (CPU-bound)
- Vision encoder: 4-6 hours
- Full model: 12-18 hours
- **Total: ~30-40 hours**

### Recommended Setup (Faster)

| Component | Spec | Cost/Hour |
|-----------|------|-----------|
| GPU | A100 (40GB) | ~$1.50 (Lambda) |
| **CPU** | **32+ cores** | - |
| RAM | 128GB | - |
| Storage | 500GB NVMe | - |

**Training Time**: ~18-24 hours total

---

## 4. Recommended Training Services

### Tier 1: Best Value (Recommended)

#### **Vast.ai**
- **Cost**: $0.30-0.50/hour for RTX 4090
- **Pros**: Cheapest, good for experiments
- **Cons**: Community GPUs (reliability varies)
- **URL**: https://vast.ai

#### **RunPod**
- **Cost**: $0.44/hour for RTX 4090, $1.99/hour for A100
- **Pros**: Easy UI, reliable, good docs
- **Cons**: Slightly more expensive than Vast
- **URL**: https://runpod.io

### Tier 2: Professional

#### **Lambda Labs**
- **Cost**: $1.10/hour for A10, $2.00/hour for A100
- **Pros**: ML-focused, excellent support
- **Cons**: Often sold out
- **URL**: https://lambdalabs.com/cloud

#### **Modal**
- **Cost**: Pay-per-second, ~$0.001/second for A100
- **Pros**: Serverless (no idle time), great for batched experiments
- **Cons**: Learning curve
- **URL**: https://modal.com

### Tier 3: Enterprise (Overkill for 16M params)

- AWS SageMaker
- Google Cloud AI Platform
- Azure ML

**Not recommended for NSCA** - too expensive for this model size.

---

## 5. Concrete Training Plan

### Step 1: Local Development (Free)
```bash
# Use CPU/MPS for debugging
python scripts/train_world_model.py --device cpu --epochs 1 --debug
```

### Step 2: Small-Scale Validation (~$5)
```bash
# Rent RTX 4090 for 2 hours on Vast.ai
# Train on 10% of data to validate pipeline
python scripts/train_world_model.py \
  --data-fraction 0.1 \
  --epochs 10 \
  --device cuda
```

### Step 3: Full Training (~$20-50)
```bash
# Rent A100 for 12-24 hours
python scripts/train_world_model.py \
  --config configs/training_config.yaml \
  --babbling-steps 100000 \
  --wandb-project nsca-training
```

### Step 4: Ablation Study (~$50-100)
```bash
# Run N=20 seeds for statistical rigor
python -c "from src.evaluation import run_ablation_study; run_ablation_study()"
```

**Total estimated cost: $75-175**

---

## 6. Data Download Script

```python
# scripts/download_data.py
from datasets import load_dataset

# ImageNet (requires agreement)
# imagenet = load_dataset("imagenet-1k", split="train")

# Kinetics-400 subset
kinetics = load_dataset("kinetics400", split="train[:10%]")

# VGGSound
vggsound = load_dataset("vggsound", split="train")

# AudioSet (subset)
audioset = load_dataset("audiocaps", split="train")  # Easier alternative

print("Data downloaded!")
```

---

## 7. Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install metaworld mujoco gymnasium

# 2. Download data (subset for testing)
python scripts/download_data.py --subset 0.1

# 3. Verify setup
python verify_world_model.py

# 4. Train (local, debug mode)
python scripts/train_world_model.py --debug --epochs 2

# 5. Train (cloud, full)
python scripts/train_world_model.py --config configs/training_config.yaml
```

---

## 8. Required Sanity Checks (Before Full Training)

### Check 1: Noisy TV Test

Before babbling, verify the curiosity filter works:

```python
# Place a TV showing random static in the environment
# Without learnability filter: agent stares at TV forever (FAILURE)
# With learnability filter: agent ignores TV after ~10 steps (SUCCESS)

python scripts/noisy_tv_test.py
```

If this fails, fix `RobustCuriosityReward` before proceeding.

### Check 2: Catastrophic Forgetting Test

Verify EWC works:

```python
# 1. Train on Task A (pick-place) → 80% success
# 2. Train on Task B (push) → 75% success  
# 3. Test Task A again
#    - Without EWC: ~30% (catastrophic forgetting)
#    - With EWC: ~70% (mild forgetting, acceptable)

python scripts/forgetting_test.py
```

### Check 3: Prior Override Test

Verify priors can be unlearned:

```python
# Train on balloons (anti-gravity) for 1000 steps
# prior_weight should drop from 0.9 → ~0.35
# If it stays at 0.9, the softplus constraint is broken

python scripts/balloon_test.py
```

---

## 9. What Success Looks Like

After training, you should see:

| Metric | Expected | Baseline |
|--------|----------|----------|
| Meta-World (5 demos) | 65% ± 8% | 22% (random init) |
| Meta-World (100 demos) | 95% ± 2% | 93% |
| Physion | 75-85% | Human: 89% |
| Babbling coverage | >80% affordances | - |

The key result: **NSCA should match random init at high data, but significantly outperform at low data (5-10 demos).**

---

## 10. The Negative Result Contingency

**If the MVP fails** (priors don't help), you still have a publishable paper:

> "Do Biologically-Inspired Priors Improve Sample Efficiency? A Negative Result on Meta-World"

This is publishable at NeurIPS/ICML negative results tracks. **Don't fear null results—fear uninterpretable results.**

Possible negative findings that are still valuable:
1. "Priors help at 5 demos but not 10" → Narrow usefulness
2. "Priors help on pick-place but not push" → Task-specific
3. "Priors hurt when physics differs from Earth" → Expected, interesting

---

## 11. Phased Execution Plan (Final)

| Phase | Cost | Time | Gate |
|-------|------|------|------|
| 0. Lethality Test | $5 | 3h | prior=0.9 > prior=0.5 by 20% |
| 1. Manual Babbling | $10 | 1 day | Grounding works on 100 objects |
| 2. Noisy TV + Forgetting | $5 | 2h | Both tests pass |
| 3. Full Babbling | $30 | 10h | >80% affordance coverage |
| 4. Vision Training | $50 | 12h | - |
| 5. Ablation Study (N=20) | $50 | 12h | Statistical significance |
| **Total** | **$150** | **~3 days** | - |

**DO NOT SKIP PHASES 0-2.** They are cheap insurance against wasting $100.
