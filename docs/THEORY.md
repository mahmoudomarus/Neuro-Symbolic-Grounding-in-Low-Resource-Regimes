# Theoretical Foundations

## Introduction

The Neuro-Symbolic Cognitive Architecture (NSCA) is grounded in research from cognitive science, developmental psychology, and neuroscience. This document explains the scientific basis for each architectural decision.

---

## 1. Innate Priors: The Nativist Foundation

### Scientific Basis

Contrary to the empiricist "blank slate" view, developmental research demonstrates that humans possess **innate perceptual and cognitive structures**:

#### Core Knowledge Theory (Spelke, 2007)

Infants possess innate "core knowledge systems" for:
- **Objects**: Cohesion, continuity, solidity
- **Agents**: Goal-directed action, self-propelled motion
- **Number**: Approximate numerical representations
- **Geometry**: Spatial reorientation

> "Core knowledge systems are: (1) present early in life, (2) found across diverse human cultures, (3) shared with other species, and (4) relatively encapsulated—operating over restricted inputs." — Spelke & Kinzler (2007)

#### Visual System Priors

The primate visual system has well-documented innate structure:

| Structure | Function | NSCA Implementation |
|-----------|----------|---------------------|
| Retinal ganglion cells | Color opponency (R-G, B-Y) | `ColorOpponencyPrior` |
| V1 simple cells | Orientation-selective edge detection | `GaborPrior` |
| MT/V5 | Motion detection | `OnsetDetector` |
| Parietal cortex | Spatial processing | `SpatialPrior3D` |

#### Auditory System Priors

The cochlea performs frequency analysis that we model:

```
Basilar Membrane → Mel-frequency representation
Tonotopic organization → Learned cochlear filterbanks
Weber-Fechner law → Logarithmic intensity scaling
```

### Implementation Rationale

Rather than learning everything from pixels (requiring billions of examples), NSCA incorporates:

```python
# Traditional approach
image → CNN → features  # Must learn color, edges, depth from scratch

# NSCA approach  
image → ColorOpponency → GaborFilters → DepthCues → CNN → features
        ↑ Innate         ↑ Innate        ↑ Innate
```

**Benefit**: Fewer training examples needed because basic perceptual structure is provided.

---

## 2. Property-Based Semantics

### Scientific Basis

#### Prototype Theory (Rosch, 1975)

Categories are organized around **prototypes** with graded membership:

- "Robin" is a more typical BIRD than "penguin"
- Category membership is based on **similarity to prototype**

#### Embodied Cognition (Barsalou, 2008)

Concepts are **grounded in perception and action**:

- "Rock" is not an abstract symbol
- "Rock" is grounded in: visual appearance, sound when struck, weight when lifted

> "Concepts are modal simulations distributed across the brain's systems for perception, action, and introspection." — Barsalou (2008)

### Implementation Rationale

NSCA represents concepts through **perceptually-grounded properties**:

```python
# Traditional approach
"rock" → word embedding [0.23, -0.15, 0.89, ...]  # Arbitrary dimensions

# NSCA approach
"rock" → PropertyVector(
    hardness=0.9,     # Grounded: sound frequency when struck
    weight=0.7,       # Grounded: force required to lift
    size=0.3,         # Grounded: visual extent
    animacy=0.0,      # Grounded: motion patterns
    ...
)
```

**Benefit**: Properties are interpretable and grounded in perception.

---

## 3. Causal Reasoning

### Scientific Basis

#### Intervention vs. Observation (Pearl, 2009)

Correlation is not causation. Causal knowledge requires:

| Type | Formula | Knowledge |
|------|---------|-----------|
| Observation | P(Y \| X) | "When X, often Y" |
| Intervention | P(Y \| do(X)) | "If I do X, then Y" |
| Counterfactual | P(Y_X \| X', Y') | "If X had been different..." |

#### Infant Causal Reasoning (Gopnik, 2012)

Infants learn causation through **active intervention**:

- 18-month-olds use intervention to distinguish causation from correlation
- Children treat their own actions as "experiments"

> "Children learn causal structure by observing contingencies and performing interventions, much like scientists." — Gopnik et al. (2004)

### Implementation Rationale

NSCA implements the **intervention calculus**:

```python
# Learning from intervention
def learn_from_intervention(state_before, action, state_after):
    """When I do action A, state changes from S to S'.
    This is evidence that A CAUSES the change."""
    
# Counterfactual reasoning
def what_would_happen_if(state, hypothetical_action):
    """Imagine doing action A. What would happen?
    Uses learned causal model to predict."""
```

**Benefit**: Agent learns actual causal structure, not just correlations.

---

## 4. Intuitive Physics

### Scientific Basis

#### Infant Physics Expectations (Baillargeon, 2004)

Infants expect:
- **Solidity**: Objects don't pass through each other
- **Support**: Unsupported objects fall
- **Continuity**: Objects move on connected paths

Violations of these expectations cause **surprise** (longer looking times).

#### Intuitive Physics Engine (Battaglia et al., 2013)

Humans have an "intuitive physics engine" that:
- Simulates approximate physical outcomes
- Predicts object trajectories
- Reasons about stability

### Implementation Rationale

NSCA encodes physics as **soft priors** that can be overridden by experience:

```python
class GravityPrior:
    """Expectation: unsupported objects fall."""
    
    def expected_motion(self, object_state):
        is_supported = self.detect_support(object_state)
        if not is_supported:
            return DOWNWARD_ACCELERATION  # Gravity expectation
        return NO_MOTION

class SolidityPrior:
    """Expectation: objects don't pass through each other."""
    
    def violation_probability(self, obj1, obj2):
        overlap = self.compute_overlap(obj1, obj2)
        return overlap  # High overlap = violation
```

**Benefit**: Agent has reasonable physics expectations without simulation data.

---

## 5. Intrinsic Motivation

### Scientific Basis

#### Curiosity and Exploration (Berlyne, 1960)

Intrinsic motivation drives exploration:
- **Novelty**: New stimuli attract attention
- **Complexity**: Moderate complexity is preferred (Goldilocks principle)
- **Surprise**: Violated expectations trigger curiosity

#### Competence Motivation (White, 1959)

Humans seek **mastery** and **effectance**:
- Satisfaction from successful predictions
- Drive to control environment

> "The urge toward competence is an intrinsic motive; the organism enjoys exercising its capacities and gains pleasure from mastering challenging tasks." — White (1959)

#### Prediction Error Minimization (Friston, 2010)

The free energy principle: organisms minimize prediction error:

```
Prediction Error = Expected State - Observed State
Goal: Minimize |Prediction Error|
```

### Implementation Rationale

NSCA implements intrinsic drives:

```python
class CuriosityDrive:
    """Seek novel, learnable experiences."""
    
    def reward(self, state, prediction_error):
        novelty = self.compute_novelty(state)
        learnability = self.estimate_learnability(state)
        
        # Goldilocks: too easy = boring, too hard = frustrating
        return novelty * learnability  # High for novel but learnable

class CompetenceDrive:
    """Seek mastery and successful prediction."""
    
    def reward(self, prediction_accuracy, learning_progress):
        return prediction_accuracy + learning_progress
```

**Benefit**: Agent learns without external reward signal.

---

## 6. Dual Memory Systems

### Scientific Basis

#### Complementary Learning Systems (McClelland et al., 1995)

Two memory systems with different properties:

| System | Brain Region | Function | Properties |
|--------|-------------|----------|------------|
| Episodic | Hippocampus | Specific experiences | Fast learning, sparse |
| Semantic | Neocortex | General knowledge | Slow learning, distributed |

#### Memory Consolidation (Kumaran et al., 2016)

Episodic memories are **consolidated** into semantic knowledge:

```
Experience → Episodic Storage → Sleep/Rehearsal → Semantic Integration
```

### Implementation Rationale

NSCA implements complementary memory:

```python
class EpisodicMemory:
    """Store specific experiences."""
    def store(self, state, metadata):
        self.buffer.append(MemoryEntry(state, metadata, timestamp))

class SemanticMemory:
    """Store general concepts (prototypes)."""
    def consolidate(self, episodic_memories):
        # Find patterns in episodic memories
        # Extract prototypes
        # Store as semantic concepts

class DualMemorySystem:
    """Integrate both systems."""
    def automatic_consolidation(self):
        # Frequent patterns in episodic → semantic prototypes
```

**Benefit**: Fast learning of new experiences + stable general knowledge.

---

## 7. Language Grounding

### Scientific Basis

#### Symbol Grounding Problem (Harnad, 1990)

How do symbols acquire meaning?

> "How can the semantic interpretation of a formal symbol system be made intrinsic to the system, rather than just parasitic on the meanings in our heads?" — Harnad (1990)

**Solution**: Ground symbols in perception and action.

#### Embodied Language (Glenberg & Kaschak, 2002)

Language comprehension involves **mental simulation**:

- "Open the drawer" activates motor representations
- Abstract concepts grounded in concrete experience

### Implementation Rationale

NSCA grounds language in perception:

```python
class LanguageGrounding:
    """Bidirectional grounding between concepts and words."""
    
    def ground_word(self, word):
        """Word → Expected perceptual properties."""
        # "rock" → PropertyVector(hard=0.9, heavy=0.7, ...)
        return self.word_to_properties[word]
    
    def verbalize_concept(self, properties):
        """Perceptual properties → Language description."""
        # PropertyVector(hard=0.9, ...) → "hard, heavy object"
        return self.properties_to_description(properties)
    
    def concept_matches_word(self, observed_properties, word):
        """Does what I see match the word?"""
        expected = self.ground_word(word)
        return similarity(observed_properties, expected)
```

**Benefit**: Language is meaningful because it connects to perception.

---

## 8. Dual-Process Architecture

### Scientific Basis

#### System 1 / System 2 (Kahneman, 2011)

Two modes of thinking:

| System 1 | System 2 |
|----------|----------|
| Fast | Slow |
| Automatic | Deliberate |
| Intuitive | Analytical |
| Low effort | High effort |

### Implementation Rationale

NSCA implements dual-process via uncertainty routing:

```python
def decide(self, state):
    uncertainty = self.estimate_uncertainty(state)
    
    if uncertainty < threshold:
        # System 1: Fast, intuitive response
        return self.direct_action(state)
    else:
        # System 2: Slow, deliberate reasoning
        return self.deliberate_reasoning(state)
```

**Benefit**: Efficient responses when possible, careful reasoning when needed.

---

## Summary: Key Theoretical Commitments

| Aspect | Traditional Deep Learning | NSCA | Theoretical Basis |
|--------|--------------------------|------|-------------------|
| Starting point | Blank slate | Innate priors | Core Knowledge (Spelke) |
| Representation | Learned features | Grounded properties | Embodied Cognition (Barsalou) |
| Learning | Correlation | Causation via intervention | Causal Inference (Pearl) |
| Physics | Learned from data | Innate intuitions | Intuitive Physics (Baillargeon) |
| Motivation | External reward | Intrinsic drives | Curiosity/Competence (Berlyne) |
| Memory | Single system | Dual episodic/semantic | Complementary Systems (McClelland) |
| Language | Token prediction | Perceptual grounding | Symbol Grounding (Harnad) |
| Processing | Single mode | Dual-process | System 1/2 (Kahneman) |

---

## References

1. Baillargeon, R. (2004). Infants' reasoning about hidden objects. *Trends in Cognitive Sciences*.

2. Barsalou, L. W. (2008). Grounded cognition. *Annual Review of Psychology*.

3. Battaglia, P. W., et al. (2013). Simulation as an engine of physical scene understanding. *PNAS*.

4. Berlyne, D. E. (1960). *Conflict, Arousal, and Curiosity*. McGraw-Hill.

5. Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*.

6. Glenberg, A. M., & Kaschak, M. P. (2002). Grounding language in action. *Psychonomic Bulletin & Review*.

7. Gopnik, A., et al. (2004). A theory of causal learning in children. *Psychological Review*.

8. Harnad, S. (1990). The symbol grounding problem. *Physica D*.

9. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

10. Kumaran, D., et al. (2016). What learning systems do intelligent agents need? *Neuron*.

11. Lake, B. M., et al. (2017). Building machines that learn and think like people. *Behavioral and Brain Sciences*.

12. McClelland, J. L., et al. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*.

13. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

14. Rosch, E. (1975). Cognitive representations of semantic categories. *Journal of Experimental Psychology: General*.

15. Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. *Developmental Science*.

16. White, R. W. (1959). Motivation reconsidered: The concept of competence. *Psychological Review*.
