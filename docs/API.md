# API Reference

Complete API documentation for the NSCA cognitive architecture.

---

## Table of Contents

1. [Cognitive Agent](#cognitive-agent)
2. [Layer 0: World Model](#layer-0-world-model)
3. [Layer 1: Semantic Properties](#layer-1-semantic-properties)
4. [Layer 2: Causal Reasoning](#layer-2-causal-reasoning)
5. [Layer 3: Motivation](#layer-3-motivation)
6. [Layer 4: Language](#layer-4-language)

---

## Cognitive Agent

### `src.cognitive_agent.CognitiveAgent`

The unified cognitive architecture integrating all layers.

```python
from src.cognitive_agent import CognitiveAgent, CognitiveConfig, create_cognitive_agent
```

#### Factory Function

```python
def create_cognitive_agent(
    world_model_config: Optional[WorldModelConfig] = None,
    use_llm: bool = False,
    llm_model: str = "gpt-3.5-turbo",
) -> CognitiveAgent:
    """
    Create a cognitive agent with aligned configurations.
    
    Args:
        world_model_config: Custom world model configuration
        use_llm: Whether to enable external LLM integration
        llm_model: Which LLM model to use
        
    Returns:
        Configured CognitiveAgent instance
    """
```

#### Methods

##### `perceive`

```python
def perceive(
    self,
    vision: torch.Tensor,
    audio: Optional[torch.Tensor] = None,
    proprio: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Full perception pipeline through all layers.
    
    Args:
        vision: Visual input [B, T, C, H, W] or [B, C, H, W]
        audio: Audio waveform [B, samples] (optional)
        proprio: Proprioception [B, T, 12] or [B, 12] (optional)
        
    Returns:
        Dict containing:
        - 'world_state': World representation [B, state_dim]
        - 'properties': PropertyVector with semantic properties
        - 'affordances': AffordanceVector
        - 'category': CategoryScores
        - 'drive_state': Current DriveState
        - 'motivation': Motivation signal [B]
        - 'description': Language description
        - 'matched_concept': Best matching concept name
        - 'match_confidence': Confidence score
    """
```

##### `act`

```python
def act(
    self,
    action: torch.Tensor,
    predicted_next_state: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Process an action and update causal understanding.
    
    Args:
        action: Action taken [B, action_dim]
        predicted_next_state: Expected next state (for competence tracking)
        
    Returns:
        Dict containing:
        - 'causal_relation': Learned CausalRelation
        - 'intrinsic_reward': Intrinsic reward signal
        - 'prediction_accuracy': How accurate was our prediction
    """
```

##### `imagine`

```python
def imagine(
    self,
    actions: torch.Tensor,
) -> Dict[str, Any]:
    """
    Imagine future states given planned actions.
    
    Args:
        actions: Sequence of planned actions [B, T, action_dim]
        
    Returns:
        Dict containing:
        - 'predicted_states': Future states [B, T+1, state_dim]
        - 'uncertainties': Prediction uncertainties [B, T, 1]
        - 'physics_plausible': List of physics plausibility flags
    """
```

##### `why_did_this_happen`

```python
def why_did_this_happen(self) -> Dict[str, Any]:
    """
    Ask why the current state occurred.
    
    Returns:
        Dict containing:
        - 'causal_type': Type of causation (intervention, physics, etc.)
        - 'confidence': Confidence in explanation
        - 'explanation': Natural language explanation
    """
```

##### `what_is_this`

```python
def what_is_this(self) -> Dict[str, Any]:
    """
    Describe what is currently being perceived.
    
    Returns:
        Dict containing:
        - 'description': Property-based description
        - 'matched_concept': Best matching concept
        - 'confidence': Match confidence
        - 'category': Fundamental category
        - 'affordances': Top affordances
    """
```

##### `answer_question`

```python
def answer_question(self, question: str) -> str:
    """
    Answer a question about current perception.
    
    Args:
        question: Natural language question
        
    Returns:
        Natural language answer (uses LLM if available)
    """
```

##### `remember` / `recall`

```python
def remember(self, label: Optional[str] = None) -> None:
    """Store current state in memory with optional label."""

def recall(self, query: Optional[str] = None) -> Dict[str, Any]:
    """Recall from memory using current state or text query."""
```

---

## Layer 0: World Model

### Configuration

```python
from src.world_model.unified_world_model import UnifiedWorldModel, WorldModelConfig
from src.encoders.vision_encoder import VisionEncoderConfig
from src.encoders.audio_encoder import AudioEncoderConfig
from src.encoders.proprio_encoder import ProprioEncoderConfig
from src.fusion.cross_modal import FusionConfig
from src.world_model.temporal_world_model import TemporalWorldModelConfig
from src.world_model.enhanced_dynamics import EnhancedDynamicsConfig

config = WorldModelConfig(
    latent_dim=256,        # Encoder output dimension
    state_dim=128,         # World state dimension
    action_dim=16,         # Action dimension
)
config.vision = VisionEncoderConfig(
    input_height=64,
    input_width=64,
    latent_dim=256,
)
config.audio = AudioEncoderConfig(
    sample_rate=16000,
    n_mels=80,
    latent_dim=128,
    output_dim=256,
)
```

### `UnifiedWorldModel`

```python
class UnifiedWorldModel(nn.Module):
    """
    Layer 0: Multi-modal world model with innate priors.
    
    Methods:
        forward(vision, audio, proprio, actions) → Full forward pass
        encode_vision(images) → Visual features
        encode_audio(waveforms) → Audio features
        encode_proprio(states) → Proprioceptive features
        build_world_state(fused_features) → World state
        imagine(state, actions) → Predicted trajectory
        remember(state, metadata) → Store in memory
        recall(query) → Retrieve from memory
    """
```

### Priors

```python
from src.priors.visual_prior import ColorOpponencyPrior, GaborPrior, DepthCuesPrior
from src.priors.audio_prior import AuditoryPrior, OnsetDetector
from src.priors.spatial_prior import SpatialPrior3D
from src.priors.temporal_prior import TemporalPrior

# Color Opponency
color_prior = ColorOpponencyPrior()
opponent = color_prior(rgb_image)  # [B, 3, H, W] → [B, 3, H, W]

# Gabor Filters
gabor_prior = GaborPrior(n_orientations=8, n_scales=4)
edges = gabor_prior(luminance)  # [B, 1, H, W] → [B, 32, H, W]

# Auditory Processing
audio_prior = AuditoryPrior(sample_rate=16000, n_mels=80)
mel = audio_prior(waveform)  # [B, samples] → [B, n_mels, T]
```

---

## Layer 1: Semantic Properties

### `PropertyLayer`

```python
from src.semantics.property_layer import PropertyLayer, PropertyConfig, PropertyVector

config = PropertyConfig(
    world_state_dim=256,
    audio_dim=256,
    proprio_dim=256,
    hidden_dim=512,
)
layer = PropertyLayer(config)

properties, embedding = layer(
    world_state,          # [B, world_state_dim]
    audio_features,       # [B, audio_dim] (optional)
    proprio_features,     # [B, proprio_dim] (optional)
    previous_state,       # [B, world_state_dim] (optional)
)
# properties: PropertyVector with .hardness, .weight, .size, .animacy, etc.
# embedding: [B, hidden_dim]
```

### `PropertyVector`

```python
class PropertyVector(NamedTuple):
    hardness: torch.Tensor      # [0=soft, 1=hard]
    weight: torch.Tensor        # [0=light, 1=heavy]
    size: torch.Tensor          # [0=tiny, 1=large]
    animacy: torch.Tensor       # [0=inanimate, 1=animate]
    rigidity: torch.Tensor      # [0=flexible, 1=rigid]
    transparency: torch.Tensor  # [0=opaque, 1=transparent]
    roughness: torch.Tensor     # [0=smooth, 1=rough]
    temperature: torch.Tensor   # [0=cold, 1=hot]
    containment: torch.Tensor   # [0=solid, 1=hollow]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to [B, 9] tensor."""
        
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'PropertyVector':
        """Create from [B, 9] tensor."""
```

### `AffordanceDetector`

```python
from src.semantics.affordances import AffordanceDetector, AffordanceConfig

detector = AffordanceDetector(AffordanceConfig())
affordances = detector(properties)

# Available affordances:
# graspable, sittable, containable, throwable, rollable,
# stackable, breakable, edible, wearable, openable, pushable, climbable

top = affordances.top_affordances(k=3)
# [('graspable', 0.85), ('throwable', 0.72), ('stackable', 0.61)]
```

### `CategoryClassifier`

```python
from src.semantics.categories import CategoryClassifier, FundamentalCategory

classifier = CategoryClassifier()
logits, scores = classifier(properties)

# scores.primary_category() → FundamentalCategory.AGENT, .OBJECT, etc.
# Categories: AGENT, OBJECT, SUBSTANCE, CONTAINER, SURFACE, TOOL
```

---

## Layer 2: Causal Reasoning

### `CausalReasoner`

```python
from src.reasoning.causal_layer import CausalReasoner, CausalConfig

config = CausalConfig(
    state_dim=256,
    action_dim=32,
    hidden_dim=512,
    num_causal_factors=16,
)
reasoner = CausalReasoner(config)

# Analyze causation
result = reasoner(state_before, state_after, action)
# result['intervention_prob']: Probability action caused change
# result['causal_type_logits']: Logits for causal type
# result['causal_strength']: Strength of causal link

# Learn from intervention
relation = reasoner.learn_from_intervention(state_before, action, state_after)

# Ask why
causal_type, confidence, explanation = reasoner.why_did_this_happen(
    state_before, state_after, action
)
```

### `IntuitivePhysics`

```python
from src.reasoning.intuitive_physics import IntuitivePhysics, PhysicsLaw

physics = IntuitivePhysics(feature_dim=256)

# Check physics expectations
is_supported, expected_motion = physics.gravity(object_state)
violations = physics.check_all(state_before, state_after)

# Physics laws: GRAVITY, SOLIDITY, SUPPORT, INERTIA, CONTACT, CONTAINMENT
```

### `CounterfactualReasoner`

```python
from src.reasoning.counterfactual import CounterfactualReasoner

cf = CounterfactualReasoner(state_dim=256)

# Intervene on a factor
counterfactual_state = cf.intervene(state, factor_idx=0, new_value=1.0)

# What-if query
result = cf.what_if(actual_state, hypothetical_factors)
# result.predicted_outcome, result.confidence, result.explanation
```

---

## Layer 3: Motivation

### `DriveSystem`

```python
from src.motivation.drive_system import DriveSystem, DriveConfig, DriveState

config = DriveConfig(state_dim=256, hidden_dim=128)
drives = DriveSystem(config)

drive_state, motivation = drives(world_state, prediction_error)

# drive_state.curiosity_level: 0-1
# drive_state.competence_level: 0-1
# drive_state.energy_level: 0-1
# drive_state.safety_level: 0-1
# drive_state.most_urgent() → DriveType

# Intrinsic reward
reward = drives.get_intrinsic_reward(state, prediction_error, learning_progress)

# Behavioral queries
should_explore = drives.should_explore()  # curiosity > 0.6
should_rest = drives.should_rest()        # energy < 0.2
```

### `IntrinsicRewardComputer`

```python
from src.motivation.intrinsic_reward import IntrinsicRewardComputer

computer = IntrinsicRewardComputer(
    state_dim=256,
    curiosity_weight=0.4,
    competence_weight=0.4,
    info_gain_weight=0.2,
)

reward, components = computer(state, next_state, action)
# components.curiosity, components.competence, components.information_gain
```

### `AttentionAllocator`

```python
from src.motivation.attention import AttentionAllocator

allocator = AttentionAllocator(feature_dim=256)
attention, components = allocator(features, drive_state)

# components['salience']: Bottom-up salience
# components['relevance']: Goal-directed relevance
# components['novelty']: Novelty signal
# components['threat']: Threat level
```

---

## Layer 4: Language

### `LanguageGrounding`

```python
from src.language.llm_integration import LanguageGrounding, LanguageConfig

config = LanguageConfig(
    concept_dim=256,
    property_dim=9,
    hidden_dim=512,
    use_external_llm=False,  # Set True for LLM integration
    llm_model="gpt-3.5-turbo",
)
lang = LanguageGrounding(config)

# Describe concept
descriptions = lang.describe_concept(property_vector)
# ["A hard, heavy, small object"]

# Ground word
properties = lang.ground_word("rock")
# tensor([0.9, 0.7, 0.3, 0.0, 0.9, 0.0, 0.7, 0.5, 0.0])

# Match concept to word
matches, score = lang.concept_matches_word(observed_properties, "rock")

# Find best matching word
word, confidence = lang.find_matching_word(property_vector)

# Answer question (uses LLM if available)
answer = lang.answer_property_question(properties, "Is this fragile?")
```

### Learned Grounding (v2.0)

**REMOVED**: Hard-coded `CONCEPT_GROUNDINGS` dictionary.
**ADDED**: `LearnedGrounding` populated through sensorimotor babbling.

```python
from src.language.llm_integration import LearnedGrounding

grounder = LearnedGrounding()

# Initially empty (no hard-coded concepts!)
assert len(grounder.get_grounded_concepts()) == 0

# Learn through interaction
grounder.learn_from_interaction(
    object_id='rock_001',
    action='strike',
    sensory_feedback={'audio_frequency': 0.9}  # High freq = hard
)

# Now 'rock_001' is grounded
props = grounder('rock_001')  # Returns learned properties
```

---

## New in v2.0: Adaptive Priors

### `AdaptivePhysicsPrior`

Physics priors that can learn exceptions (balloons, magnets):

```python
from src.reasoning.intuitive_physics import AdaptivePhysicsPrior

prior = AdaptivePhysicsPrior(feature_dim=256)

# Forward pass returns: is_supported, motion, diagnostics
is_supported, motion, diag = prior(object_state)

# Check prior weight (always >= 0.3 due to critical period floor)
print(f"Prior weight: {diag['prior_weight']}")  # ~0.9 initially

# After training on balloons, prior_weight drops to ~0.35
# Correction network learns "balloons go up"
```

### `RobustCuriosityReward`

Curiosity with noisy-TV defense:

```python
from src.motivation.intrinsic_reward import RobustCuriosityReward

curiosity = RobustCuriosityReward(state_dim=256)

reward, diagnostics = curiosity(state, next_state, action)

# diagnostics contains:
# - 'learnability': How much error decreased (high = worth learning)
# - 'prediction_error': Raw prediction error
# - 'filtered_reward': error × learnability
```

### `DynamicPropertyBank`

Open-ended property discovery:

```python
from src.semantics.property_layer import DynamicPropertyBank

bank = DynamicPropertyBank(input_dim=256, num_slots=32)

# 9 known slots + 23 free slots
values, diagnostics = bank(world_state)

# Check for discovered properties
active_free = bank.get_active_free_slots()  # E.g., [9, 12] for stickiness, elasticity

# Ground discovered slots post-hoc
bank.ground_free_slot(slot_idx=0, name="stickiness")
```

### `ElasticWeightConsolidation`

Prevent catastrophic forgetting:

```python
from src.learning.ewc import MemoryAwareEWC, EWCConfig

ewc = MemoryAwareEWC(model, EWCConfig(semantic_multiplier=10.0))

# After training on Task A:
ewc.consolidate(task_a_dataloader)

# During training on Task B:
loss = task_b_loss + ewc.penalty()  # Protects important weights
```

### `CurriculumBabbling`

Two-phase grounding protocol:

```python
from src.learning import CurriculumBabbling, SimulatedBabblingEnvironment

babbling = CurriculumBabbling()
env = SimulatedBabblingEnvironment()

while not babbling.is_complete:
    action = babbling.select_action(env.get_available_actions())
    feedback = env.execute_action(action)
    babbling.record_interaction(env.get_current_object_id(), action, feedback, error)

# Phase 1 (steps 1-1000): Random exploration
# Phase 2 (steps 1001-10000): Competence-driven (retry learnable actions)
```

---

## Memory Systems

### `DualMemorySystem`

```python
from src.memory.dual_memory import DualMemorySystem

memory = DualMemorySystem(
    vector_dim=256,
    max_episodic=10000,
    similarity_threshold=0.85,
)

# Store experience
memory.store(state_vector, metadata={'label': 'rock'})

# Recall similar
episodic_matches = memory.recall_episodic(query_vector, top_k=5)
semantic_matches = memory.recall_semantic(query_vector, top_k=3)

# Learn new concept
memory.learn_new_concept(example_vectors, 'new_concept')

# Consolidation happens automatically
```

---

## Utilities

### Verification

```bash
python verify_world_model.py
```

### Testing

```bash
# All tests
pytest tests/ -v

# Specific layer
pytest tests/test_cognitive_layers.py -v
```

### Configuration Helpers

```python
# Get default config with aligned dimensions
from src.cognitive_agent import CognitiveConfig

config = CognitiveConfig()
# Modify as needed
config.world_model.latent_dim = 512
```
