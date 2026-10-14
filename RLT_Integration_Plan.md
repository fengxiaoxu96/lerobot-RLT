# RLToken Integration Plan for LeRobot (SO-ARM Target)

**Goal:** Integrate the RLToken (RLT) two-stage online RL pipeline into the lerobot codebase, targeting the SO-ARM robot, without modifying any existing files.

**Reference:** [RLToken_Framework.md](../openpi-RLT/RLToken_Framework.md) — full description of the source system in openpi-RLT.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Key Design Decisions](#2-key-design-decisions)
3. [New File Map](#3-new-file-map)
4. [Per-File Specification](#4-per-file-specification)
5. [Inter-File Dependency Graph](#5-inter-file-dependency-graph)
6. [Development Order](#6-development-order)
7. [SO-ARM Specifics](#7-so-arm-specifics)
8. [Open Questions](#8-open-questions)

---

## 1. Architecture Overview

RLT runs in two stages:

**Stage 1 — Offline (VLA + RLToken training)**
- Load a frozen or fine-tuned pi0.5 VLA.
- Add a cross-attention encoder/decoder (RLTokenModel) that compresses the VLA's image prefix embeddings into a fixed-size RL state token `z_rl`.
- Train on robot demonstrations with loss `L_rlt (MSE reconstruction) + α·L_vla (flow matching)`.
- Output: a checkpoint with both VLA weights and trained RLToken weights.

**Stage 2 — Online (Actor-Critic on real robot)**
- **Machine A (GPU):** Serves frozen VLA + RLToken over WebSocket. Per observation, returns `z_rl` (compressed state), `ref_chunk` (VLA reference action sequence), and `proprio` (joint positions).
- **Machine B (GPU/CPU):** Three processes — Replay Manager (B3), Learner (B2, actor-critic training), Actor Service (B1, MLP policy inference). Learner hot-reloads actor to B1 via snapshot file.
- **Robot PC:** Queries Machine A for features, queries B1 for refined actions, executes on robot, sends transitions to B3.

For SO-ARM: the robot PC is typically the same machine as Machine B (USB serial, no ROS needed). Machine A can be the same machine or a separate GPU server.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Offline (this machine, GPU)                                              │
│  scripts/train_rlt.py                                                               │
│  policies/pi05_rlt/  ──▶  policies/rltoken/  ──▶  orbax/torch checkpoint           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                   │ checkpoint
                                   ▼
┌──────────────────────┐    WebSocket     ┌──────────────────────────────────────────┐
│  MACHINE A           │◀─────────────────│  ROBOT PC + MACHINE B                   │
│  scripts/            │  z_rl,ref,proprio │                                         │
│  serve_rlt_policy.py │─────────────────▶│  rl/rltoken/rollout.py (SO-ARM)         │
│  (pi05 + RLToken,    │                  │  rl/rltoken/inference.py (ActorService)  │
│   frozen)            │                  │  rl/rltoken/trainer.py  (LearnerService) │
└──────────────────────┘                  │  rl/rltoken/replay.py   (ReplayManager)  │
                                          └──────────────────────────────────────────┘
```

---

## 2. Key Design Decisions

### 2.1 No existing files modified
All new code lives in new modules. The pi0.5 VLA is extended by **subclassing** `PI05Pytorch` in a new file, not by editing the original.

### 2.2 PyTorch throughout (not JAX)
lerobot is PyTorch-based. The RLToken encoder/decoder and actor-critic networks are ported to PyTorch `nn.Module`. The openpi-RLT used JAX; direct translation is straightforward (cross-attention → `nn.MultiheadAttention`, MLP → `nn.Linear`).

### 2.3 No ROS2
SO-ARM connects via USB serial through `SOFollower` (FeetechMotorsBus). The rollout loop calls `SOFollower.get_observation()` and `SOFollower.send_action()` directly. No ROS bridge needed.

### 2.4 Chunk-level transitions, not single-step
lerobot's existing `Transition` and `ReplayBuffer` are single-step (SAC). RLT is chunk-level (10-step chunks). A new `RLTTransition` and `RLTReplayBuffer` are created rather than adapting the existing ones.

### 2.5 Independent replay HTTP service (same as openpi-RLT)
Replay is served over HTTP (msgpack) to decouple the robot process from the learner process. The existing lerobot gRPC transport is not used for the replay path — it would require significant refactoring of a non-modified file.

### 2.6 SO-ARM action space
6-DOF: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
- `action_dim = 6`
- Delta representation: first 5 dims relative to `state0` of the chunk; dim 5 (gripper) absolute.
- Quantile normalize to `[0, 1]` using pre-computed `q01/q99` from demonstrations.

### 2.7 VLA prefix embedding extraction
`PI05Pytorch` does not expose intermediate embeddings. We subclass it as `PI05RLT` and add `extract_prefix_embeddings()` (image-only prefix forward, no action expert) and `compute_loss_with_prefix()` (full forward returning both flow-matching loss and prefix embeddings).

---

## 3. New File Map

```
src/lerobot/
├── policies/
│   ├── pi05_rlt/
│   │   ├── __init__.py
│   │   └── modeling_pi05_rlt.py          # PI05RLT subclass: prefix embedding extraction
│   └── rltoken/
│       ├── __init__.py
│       ├── configuration_rltoken.py      # RLTokenConfig (draccus)
│       ├── rl_token.py                   # RLTokenEncoder, RLTokenDecoder, RLTokenModel (PyTorch)
│       └── modeling_rltoken.py           # RLTokenPolicy: Stage 1 training + inference
│
└── rl/
    └── rltoken/
        ├── __init__.py
        ├── config.py                     # Online RL config dataclasses (all services)
        ├── transition.py                 # RLTTransition TypedDict, TransitionSource
        ├── action_representation.py      # ActionRepresentationAdapter (delta, quantile norm)
        ├── replay.py                     # RLTReplayBuffer + HTTP server
        ├── networks.py                   # ChunkActor, TwinCritic (PyTorch MLP)
        ├── trainer.py                    # RLTTrainState, LearnerService
        ├── inference.py                  # ActorService, PhaseAwareActorClient, EnvDriver
        └── rollout.py                    # SO-ARM RolloutOrchestrator, PhaseController, SignalBridge

configs/
└── rltoken/
    ├── base.yaml                         # Default hyperparameters
    └── tasks/
        └── so_arm_ethernet/
            ├── online_rl.yaml            # Task-specific overrides
            └── stats/
                └── norm_stats_delta.json # q01/q99 per-dim for SO-ARM demonstrations

scripts/                                  (top-level, alongside existing lerobot scripts)
    ├── train_rlt.py                      # Stage 1: offline RLToken training
    ├── serve_rlt_policy.py               # Machine A: WebSocket VLA+RLToken server
    ├── launch_machine_b_rlt.py           # Machine B: spawn replay+learner+actor
    ├── launch_robot_rollout_rlt.py       # Robot: health-gate + start SO-ARM rollout
    └── launch_actor_eval_rlt.py          # Eval-only: actor service + eval rollout
```

---

## 4. Per-File Specification

### 4.1 `policies/pi05_rlt/modeling_pi05_rlt.py`

**Class:** `PI05RLT(PI05Pytorch)`

**Purpose:** Extends pi0.5 with two additional forward-pass methods needed by Stage 1 training and Machine A serving. No change to the existing file.

**Methods to add:**

```python
def extract_prefix_embeddings(
    self,
    batch: dict,
    image_only: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Run only the PaliGemma backbone (language + vision encoder).
    Return (prefix_embs, prefix_mask):
      prefix_embs: [B, seq_len, hidden_dim=2048]  image token positions only if image_only=True
      prefix_mask: [B, seq_len] bool              True where valid
    No action expert, no flow matching. No gradients needed for frozen VLA.
    """

def compute_loss_with_prefix(
    self,
    batch: dict,
    image_only: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Full flow-matching forward pass that ALSO returns prefix embeddings.
    Return (per_sample_loss, prefix_embs, prefix_mask)
    Used for joint training (alpha > 0) where VLA gradients are needed.
    """
```

**How to implement without modifying PI05Pytorch:** Override `forward()` or, better, call `self.model.language_model(...)` directly using the same input preparation that the parent class uses but captured via a registered forward hook on the backbone's last layer. Alternatively, call the parent's internal tokenizer/processor, then run only the vision encoder layers. The exact approach depends on the internal structure of `PI05Pytorch` — inspect it first and choose the least invasive hook point.

**Dependencies:** `PI05Pytorch` (existing, not modified), standard torch.

---

### 4.2 `policies/rltoken/rl_token.py`

**Purpose:** PyTorch port of the JAX `RLTokenEncoder`, `RLTokenDecoder`, `RLTokenModel` from openpi-RLT's `src/openpi/models/rl_token.py`.

**Classes:**

```python
@dataclass
class RLTokenConfig:
    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 2048       # matches PaliGemma 2B hidden size for SO-ARM
    input_dim: int = 2048
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

class RLTokenEncoder(nn.Module):
    """
    Learned query embeddings [num_tokens, embed_dim] cross-attend to
    prefix_embs [B, seq, input_dim] to produce z_rl [B, num_tokens, embed_dim].
    Stack of num_layers CrossAttentionBlock(query=queries, kv=prefix).
    Optional input projection if embed_dim != input_dim.
    """
    def forward(self, prefix_embs: Tensor, prefix_mask: Tensor) -> Tensor: ...

class RLTokenDecoder(nn.Module):
    """
    Learned positional queries [seq, embed_dim] cross-attend to
    z_rl [B, num_tokens, embed_dim] to reconstruct prefix_embs [B, seq, embed_dim].
    Stack of num_layers CrossAttentionBlock.
    """
    def forward(self, z_rl: Tensor, seq_len: int) -> Tensor: ...

class RLTokenModel(nn.Module):
    """
    Wraps encoder + decoder.
    loss(prefix_embs, prefix_mask):
        z_rl = encoder(prefix_embs, prefix_mask)
        recon = decoder(z_rl, seq_len)
        return mse_loss(recon, stop_gradient(prefix_embs[mask]))
    Also exposes encode() for inference-only use.
    """
```

**Key building block:** `CrossAttentionBlock(nn.Module)` — Q from one source, K/V from another, with LayerNorm and MLP. Uses `nn.MultiheadAttention` with `batch_first=True`.

**Dependencies:** `torch`, `torch.nn`. No lerobot dependencies.

---

### 4.3 `policies/rltoken/configuration_rltoken.py`

**Purpose:** draccus-registered config for the RLTokenPolicy (Stage 1).

```python
@PreTrainedConfig.register_subclass("rltoken")
@dataclass
class RLTokenConfig(PreTrainedConfig):
    # VLA backbone
    pi05_pretrained_path: str = ""
    alpha: float = 0.0          # 0 = frozen VLA; 1 = joint training
    image_only_prefix: bool = True

    # RLToken architecture
    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 2048
    input_dim: int = 2048

    # Action space (SO-ARM)
    action_dim: int = 6
    action_horizon: int = 50    # VLA chunk length (full)
    chunk_len: int = 10         # RL execution chunk length
```

**Dependencies:** `lerobot.configs.policies.PreTrainedConfig` (existing).

---

### 4.4 `policies/rltoken/modeling_rltoken.py`

**Class:** `RLTokenPolicy(PreTrainedPolicy)`

**Purpose:** Stage 1 training wrapper. Owns `PI05RLT` (VLA) + `RLTokenModel`.

**Key methods:**

```python
def forward(self, batch: dict) -> tuple[Tensor, dict]:
    """
    Stage 1 loss computation.
    if alpha == 0:
        prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(batch)
        rlt_loss = self.rlt_model.loss(prefix_embs.detach(), prefix_mask)
        return rlt_loss, {}
    else:
        vla_loss, prefix_embs, prefix_mask = self.vla.compute_loss_with_prefix(batch)
        rlt_loss = self.rlt_model.loss(prefix_embs.detach(), prefix_mask)
        return rlt_loss + alpha * vla_loss, {"vla_loss": vla_loss, "rlt_loss": rlt_loss}
    """

def encode(self, batch: dict) -> Tensor:
    """
    Inference: run VLA prefix + RLToken encoder, return z_rl [B, embed_dim].
    Used by Machine A serving.
    """

def predict_action_chunk(self, batch: dict) -> Tensor:
    """
    Inference: run full VLA sample_actions to get ref_chunk [B, chunk_len, action_dim].
    Machine A serving returns this alongside z_rl.
    """
```

**Trainable parameter filter:** Controlled by `alpha`. If `alpha=0`, only `rlt_model` parameters have `requires_grad=True`; VLA is fully frozen. Set this in `__init__()` based on config.

**Dependencies:** `PI05RLT`, `RLTokenModel`, `RLTokenConfig`.

---

### 4.5 `rl/rltoken/transition.py`

**Purpose:** Define the chunk-level transition structure and source enum.

```python
from enum import IntEnum
from typing import TypedDict
import numpy as np

class TransitionSource(IntEnum):
    BASE = 0    # VLA ref_chunk used directly, no actor
    RL = 1      # actor refined chunk
    HUMAN = 2   # teleop override
    MIXED = 3   # partial override mid-chunk

class RLTTransition(TypedDict):
    # Current features
    z_rl: np.ndarray           # [z_dim]        float16
    proprio: np.ndarray        # [proprio_dim]  float32
    ref_chunk: np.ndarray      # [chunk_len, action_dim]  float16

    # Executed action and outcome
    action_chunk: np.ndarray   # [chunk_len, action_dim]  float16
    rewards: np.ndarray        # [chunk_len]   float32  (0 except terminal)
    done: bool

    # Next-step features (for TD backup)
    next_z_rl: np.ndarray      # [z_dim]        float16
    next_proprio: np.ndarray   # [proprio_dim]  float32
    next_ref_chunk: np.ndarray # [chunk_len, action_dim]  float16

    # Metadata
    source: TransitionSource
    source_chunk: np.ndarray   # [chunk_len] TransitionSource per step
    collection_phase: str      # "warmup" | "online"
    success: bool
    episode_id: str
    step_id: int
```

**Dependencies:** `numpy`, standard library only.

---

### 4.6 `rl/rltoken/config.py`

**Purpose:** All configuration dataclasses for the online RL system, mirroring `rlt_online_rl/src/rlt_online_rl/config.py` from openpi-RLT but adapted for lerobot conventions.

**Dataclasses to define:**

```python
@dataclass
class RLTOnlineRLConfig:
    # Action space
    action_dim: int = 6          # SO-ARM: 6-DOF
    chunk_len: int = 10
    z_dim: int = 2048
    proprio_dim: int = 6         # SO-ARM: 6 joint positions

    # Action representation
    action_representation: str = "delta_chunk"   # "abs" | "delta_chunk"
    action_norm_stats_path: str = ""

    # RL algorithm
    gamma: float = 0.99
    fixed_std: float = 0.002
    reference_dropout_prob: float = 0.5

    # Loss weights
    warmup_bc_weight: float = 10.0
    warmup_q_weight: float = 0.1
    online_bc_weight: float = 5.0
    online_q_weight: float = 0.1
    delta_weight: float = 10.0

    # Network architecture
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    # Optimization
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    target_tau: float = 0.005
    actor_update_period: int = 2

    # Warmup
    warmup_min_size: int = 600
    warmup_post_collect_updates: int | None = 20000
    grad_updates_per_cycle: int = 5

@dataclass
class ActorServiceConfig:
    host: str = "127.0.0.1"
    port: int = 9101
    snapshot_path: str = ""
    pull_params_interval_sec: float = 0.25
    xla_mem_fraction: float = 0.10    # unused in PyTorch, kept for compatibility

@dataclass
class LearnerServiceConfig:
    push_actor_every_n_steps: int = 500
    checkpoint_every_n_steps: int = 1000
    batch_size: int = 128
    replay_url: str = "http://127.0.0.1:9102"

@dataclass
class ReplayConfig:
    capacity: int = 200_000
    journal_path: str = ""
    host: str = "127.0.0.1"
    port: int = 9102
    sampling_strategy: str = "stratified"   # "uniform" | "stratified"

@dataclass
class EnvDriverConfig:
    machine_a_ws_url: str = "ws://127.0.0.1:8000"
    actor_url: str = "http://127.0.0.1:9101"
    replay_url: str = "http://127.0.0.1:9102"
    task_mode: str = "critical_phase"     # "critical_phase" | "full_task"
    control_frequency_hz: float = 20.0
    chunk_exec_horizon: int = 10
    actor_deterministic: bool = False
    safe_fallback_to_ref: bool = True
    enable_human_override: bool = False
    replay_feature_batch_size: int = 16

@dataclass
class OnlineRLSystemConfig:
    rl: RLTOnlineRLConfig = field(default_factory=RLTOnlineRLConfig)
    actor_service: ActorServiceConfig = field(default_factory=ActorServiceConfig)
    learner: LearnerServiceConfig = field(default_factory=LearnerServiceConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    env_driver: EnvDriverConfig = field(default_factory=EnvDriverConfig)
    wandb_enabled: bool = False
    run_dir: str = "./runs/so_arm_rlt"
```

Also provide `load_system_config_yaml(path)` and `save_system_config_yaml(cfg, path)` using `dataclasses.asdict` + PyYAML with config-relative path resolution.

**Dependencies:** `dataclasses`, `pathlib`, `yaml`. No lerobot-specific imports.

---

### 4.7 `rl/rltoken/action_representation.py`

**Purpose:** PyTorch/NumPy port of `ActionRepresentationAdapter` from openpi-RLT. Handles delta-chunk conversion and quantile normalization.

**Class:**

```python
class ActionRepresentationAdapter:
    """
    Loads q01/q99 from a JSON file (same format as openpi-RLT).
    action_dim = 6 for SO-ARM; first 5 dims are delta, dim 5 (gripper) is absolute.
    """
    def __init__(self, norm_stats_path: str): ...

    def normalize_chunk(
        self,
        chunk_abs: np.ndarray,    # [chunk_len, action_dim] absolute joint angles
        state0: np.ndarray,       # [action_dim] joint state at chunk start
    ) -> np.ndarray:
        """
        1. delta[..., :5] = abs[..., :5] - state0[:5]  (first 5 joints relative)
        2. delta[..., 5]  = abs[..., 5]                 (gripper absolute)
        3. quantile normalize: (x - q01) / (q99 - q01), clip to [0, 1]
        """

    def denormalize_to_abs_chunk(
        self,
        chunk_norm: np.ndarray,   # [chunk_len, action_dim] normalized delta
        state0: np.ndarray,       # [action_dim] joint state at chunk start
    ) -> np.ndarray:
        """Reverse: quantile denormalize, then add state0[:5] to first 5 dims."""

    def prepare_training_batch(self, batch: dict) -> dict:
        """
        Normalize ref_chunk, action_chunk, next_ref_chunk in-place.
        state0 is taken from batch["proprio"] (current joint state).
        """

    # PyTorch versions for JIT-able training
    def normalize_chunk_torch(self, chunk_abs: Tensor, state0: Tensor) -> Tensor: ...
    def denormalize_to_abs_chunk_torch(self, chunk_norm: Tensor, state0: Tensor) -> Tensor: ...
```

**Dependencies:** `numpy`, `torch`, `json`, `pathlib`. No lerobot dependencies.

---

### 4.8 `rl/rltoken/replay.py`

**Purpose:** PyTorch port of the replay buffer and HTTP service from openpi-RLT's `replay.py`.

**Classes:**

```python
class RLTReplayBuffer:
    """
    Thread-safe CPU ring buffer for RLTTransition objects.
    Lazily allocates NumPy arrays on first add().
    """
    def __init__(self, config: ReplayConfig): ...

    def add(self, transition: RLTTransition) -> None: ...
    def extend(self, transitions: list[RLTTransition]) -> None: ...

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Uniform or stratified sampling.
        Stratified: separate pools for recent online, warmup-demo, human-intervention.
        Returns dict of stacked numpy arrays ready for torch conversion.
        """

    def size(self) -> int: ...
    def stats(self) -> dict: ...

class ReplayManager:
    """
    Wraps RLTReplayBuffer + append-only pickle journal.
    Journal enables crash recovery via _restore_from_journal() on startup.
    HTTP service: ThreadingHTTPServer (msgpack-encoded).
      GET  /healthz  → {"status": "ok"}
      GET  /stats    → buffer stats dict
      POST /add      → add one RLTTransition
      POST /extend   → add list of RLTTransitions
      POST /sample   → {"batch_size": N} → dict of stacked arrays
    """

class ReplayClient:
    """HTTP client that posts to ReplayManager."""
    def add(self, transition: RLTTransition) -> None: ...
    def extend(self, transitions: list[RLTTransition]) -> None: ...
    def sample(self, batch_size: int) -> dict[str, np.ndarray]: ...
    def healthz(self) -> bool: ...
```

Also: `build_chunk_transitions_from_episode(trace, config, adapter)` and `build_terminal_aligned_chunk_transition(trace, config, adapter)` — converts a completed episode trace into a list of `RLTTransition` objects.

**Dependencies:** `transition.py`, `config.py`, `action_representation.py`, `numpy`, `msgpack`, `http.server`.

---

### 4.9 `rl/rltoken/networks.py`

**Purpose:** PyTorch port of `ChunkActor`, `TwinCritic`, and TD-target utilities from openpi-RLT's `networks.py`.

**Classes:**

```python
class ChunkActor(nn.Module):
    """
    Inputs: z_rl [B, z_dim], proprio [B, proprio_dim], ref_chunk [B, chunk_len, action_dim]
    Projections:
      z_rl     → Linear(z_dim, hidden_dim=256)
      proprio  → Linear(proprio_dim, 64)
      ref_chunk → flatten to [B, chunk_len*action_dim], then Linear(→256)
    Concat → [B, 256+64+256=576] → LayerNorm → GELU MLP (num_layers) → [B, chunk_len*action_dim]
    Reshape → [B, chunk_len, action_dim]
    Fixed std: self.log_std = nn.Parameter(torch.full([action_dim], log(fixed_std)), requires_grad=False)
    """
    def forward(self, z_rl, proprio, ref_chunk) -> Tensor:
        """Returns action mean [B, chunk_len, action_dim]."""

    def sample(self, z_rl, proprio, ref_chunk, deterministic=False) -> tuple[Tensor, Tensor]:
        """Returns (sampled_chunk, log_prob) where log_prob is summed over chunk×dim."""

    def apply_reference_dropout(self, ref_chunk: Tensor, prob: float) -> Tensor:
        """Zero entire ref_chunk for a sample with probability prob (whole-chunk, not per-step)."""

class QNetwork(nn.Module):
    """
    Inputs: z_rl, proprio, action_chunk (NOT ref_chunk)
    Same projection structure as ChunkActor but outputs scalar Q-value.
    """
    def forward(self, z_rl, proprio, action_chunk) -> Tensor:
        """Returns Q-value [B, 1]."""

class TwinCritic(nn.Module):
    """Holds two QNetwork instances (q1, q2). Returns both Q-values."""
    def q_values(self, z_rl, proprio, action_chunk) -> tuple[Tensor, Tensor]:
        """Returns (q1, q2) each [B, 1]."""
    def min_q(self, z_rl, proprio, action_chunk) -> Tensor:
        """Returns min(q1, q2) [B, 1]."""

def build_td_target(
    next_z_rl: Tensor, next_proprio: Tensor, next_ref_chunk: Tensor,
    rewards: Tensor,       # [B, chunk_len]
    done: Tensor,          # [B]
    target_actor: ChunkActor,
    target_critic: TwinCritic,
    gamma: float,
    chunk_len: int,
    reference_dropout_prob: float,
) -> Tensor:
    """
    1. Sample target action: target_actor.sample(next_z_rl, next_proprio, next_ref_chunk, deterministic=True)
    2. Compute discounted reward: sum(r_t * gamma^t) for t in [0, chunk_len)
    3. Bootstrap: (1 - done) * gamma^chunk_len * min(target_critic.q_values(next_*, target_action))
    4. Return discounted_reward + bootstrap   [B, 1]
    """
```

**Dependencies:** `torch`, `torch.nn`. No lerobot dependencies.

---

### 4.10 `rl/rltoken/trainer.py`

**Purpose:** Training loop for the actor-critic. PyTorch port of openpi-RLT `trainer.py`.

**Classes:**

```python
@dataclass
class RLTTrainState:
    actor: ChunkActor
    target_actor: ChunkActor
    critic: TwinCritic
    target_critic: TwinCritic
    actor_optimizer: torch.optim.Adam
    critic_optimizer: torch.optim.Adam
    global_step: int = 0
    actor_version: int = 0

def init_train_state(config: RLTOnlineRLConfig) -> RLTTrainState:
    """Initialize all networks and optimizers from config."""

def update_critic(
    state: RLTTrainState,
    batch: dict,
    config: RLTOnlineRLConfig,
    adapter: ActionRepresentationAdapter,
) -> dict:
    """
    Compute TD target, MSE loss for both Q networks, backward, step optimizer.
    Returns {"critic_loss": float, "q1_mean": float, "q2_mean": float}.
    """

def update_actor(
    state: RLTTrainState,
    batch: dict,
    config: RLTOnlineRLConfig,
    adapter: ActionRepresentationAdapter,
    bc_weight: float,
    q_weight: float,
) -> dict:
    """
    Composite actor loss:
      bc_penalty  = MSE(actor_chunk, source_aware_target)
                    where source_aware_target is action_chunk for HUMAN/MIXED steps,
                    ref_chunk for BASE/RL steps
      q_loss      = -min_q(z_rl, proprio, actor_chunk)
      delta_penalty = MSE of consecutive joint diffs (first 5 dims only) in abs space
      total = bc_weight * bc_penalty + q_weight * q_loss + delta_weight * delta_penalty
    Apply reference dropout to ref_chunk before actor forward.
    Returns {"actor_loss": float, "bc_penalty": float, "q_actor": float, "delta_penalty": float}.
    """

def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """target = tau*source + (1-tau)*target"""

class LearnerService:
    """
    Owns RLTTrainState + ReplayClient + ActionRepresentationAdapter.
    run_forever(): polling loop
      train_once():
        1. Check warmup readiness (replay.size() >= warmup_min_size)
        2. Sample batch from replay (via ReplayClient)
        3. prepare_training_batch() via adapter
        4. update_critic()
        5. Every actor_update_period steps: update_actor(), polyak_update()
        6. Every push_actor_every_n_steps: export_actor_snapshot()
        7. Every checkpoint_every_n_steps: save_checkpoint()
        8. Write learner_status.json ({"ready_for_online": bool, "global_step": int})
        9. Log metrics to JSONL

    export_actor_snapshot(): atomic pickle write of actor.state_dict() + config to snapshot_path
    save_checkpoint() / load_checkpoint(): full RLTTrainState persistence (torch.save/load)
    """
```

**Dependencies:** `networks.py`, `replay.py`, `config.py`, `action_representation.py`, `torch`.

---

### 4.11 `rl/rltoken/inference.py`

**Purpose:** Actor HTTP service and environment-driver orchestration. PyTorch port of openpi-RLT's `inference.py`.

**Data classes:**

```python
class ActorRequest(TypedDict):
    z_rl: np.ndarray           # [z_dim]
    proprio: np.ndarray        # [proprio_dim]
    ref_chunk: np.ndarray      # [chunk_len, action_dim]  normalized delta space
    request_id: str
    deterministic: bool

class ActorResponse(TypedDict):
    refined_chunk: np.ndarray  # [chunk_len, action_dim]  normalized delta space
    actor_param_version: int
    request_id: str
    source: TransitionSource
```

**Classes:**

```python
class ActorService:
    """
    HTTP server at :9101 serving ChunkActor inference.
    POST /infer  → ActorResponse
    GET  /healthz → {"status": "ok"}
    Polls snapshot file every pull_params_interval_sec and hot-reloads actor weights.
    Normalizes ref_chunk via adapter before feeding to actor; returns normalized delta chunk.
    """

class ActorClient:
    """HTTP client matching ActorService."""
    def infer(self, request: ActorRequest) -> ActorResponse: ...
    def healthz(self) -> bool: ...

class PhaseAwareActorClient:
    """
    Wraps ActorClient.
    In warmup or base-mode episodes: return ref_chunk directly with source=BASE.
    In online+critical phase: call ActorClient.infer().
    Also manages reference_dropout when source=BASE.
    """

class MachineAClient:
    """
    WebSocket client to Machine A.
    query(observation: dict) -> (z_rl, ref_chunk, proprio)
    where observation = {'images': {...}, 'state': np.ndarray}
    """

class EnvDriver:
    """
    Top-level episode orchestrator.
    run_episode():
      1. env.reset()
      2. Loop at control_frequency_hz:
           a. env.get_observation()
           b. machine_a_client.query(obs) → z_rl, ref_chunk, proprio
           c. phase_aware_client.infer(z_rl, proprio, ref_chunk) → action_chunk
           d. env.execute_chunk(action_chunk)  (or env.step() × chunk_len)
           e. record step (state, reward, done)
      3. Episode end:
           a. Backfill missing anchor features (batch query to Machine A)
           b. build_chunk_transitions_from_episode()
           c. replay_client.extend(transitions)
      4. Return episode_result (success, n_steps, total_reward)
    """
```

**Dependencies:** `networks.py`, `config.py`, `transition.py`, `replay.py`, `action_representation.py`, `torch`, `websockets`, `http`.

---

### 4.12 `rl/rltoken/rollout.py`

**Purpose:** SO-ARM-specific rollout logic. Wraps `SOFollower` into the EnvDriver's environment interface, manages the phase state machine, and provides keyboard signal handling.

**Classes:**

```python
class SOArmEnv:
    """
    Wraps SOFollower (via ThreadSafeRobot) into the interface EnvDriver expects.
    reset(reset_pose: np.ndarray): move to reset pose, wait reset_time_s
    get_observation() -> dict:
      {"state": joint_pos_array [6], "images": {"cam_key": frame_array}}
    execute_chunk(abs_chunk: np.ndarray, hz: float) -> list[StepResult]:
      Control loop: for each action in abs_chunk at hz:
        denormalize abs_chunk step, call SOFollower.send_action({motor: val, ...})
        read next obs, record (obs, action, reward, done)
      Returns list of StepResult per step.
    """

class RolloutPhaseController:
    """
    Episode-level state machine: warmup_collect → warmup_wait_online → online.
    Polls learner_status.json (written by LearnerService) for "ready_for_online" gate.
    """
    def current_phase(self) -> str: ...
    def should_use_actor(self, episode_num: int) -> bool: ...
    def on_episode_end(self, success: bool): ...

class KeyboardSignalBridge:
    """
    Non-blocking keyboard listener for operator signals.
    Keys: [n] next episode, [s] success, [f] failure, [c] critical-start,
          [a] actor-mode, [b] base-mode, [t] teleop-toggle.
    Uses threading + queue; EnvDriver polls get_signal() between steps.
    No ROS, no external sidecar — in-process on the robot machine.
    """
    def get_signal(self) -> str | None: ...

class RolloutOrchestrator:
    """
    Top-level for the robot machine.
    Assembles: SOArmEnv, PhaseAwareActorClient, MachineAClient, ReplayClient,
               RolloutPhaseController, KeyboardSignalBridge.
    run_forever(): episode loop, calls EnvDriver.run_episode(), handles signals,
                   logs episode result.
    """
```

**Dependencies:** `inference.py`, `config.py`, `transition.py`, lerobot `SOFollower`, `ThreadSafeRobot`, `pynput` or `tty/termios` for keyboard.

---

### 4.13 `scripts/train_rlt.py`

**Purpose:** Stage 1 entry point. Uses lerobot's existing training infrastructure.

**Key structure:**

```python
@dataclass
class RLTTrainConfig:
    policy: RLTokenConfig
    dataset: DatasetConfig         # LeRobotDataset config
    optimizer: OptimizerConfig     # existing lerobot optimizer
    steps: int = 5000
    batch_size: int = 32
    log_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    output_dir: str = "./runs/rlt_stage1"
    wandb: WandBConfig | None = None
```

Main loop:
1. Load `LeRobotDataset` → `DataLoader`
2. Instantiate `RLTokenPolicy`
3. Load pi0.5 pretrained weights into `policy.vla` (from `RLTokenConfig.pi05_pretrained_path`)
4. Set `requires_grad` based on alpha (via `policy.freeze_vla()` helper)
5. Standard train loop: `batch → policy.forward(batch) → loss.backward() → optimizer.step()`
6. Checkpoint via `torch.save` every N steps

**CLI:** `python scripts/train_rlt.py --config configs/rltoken/tasks/so_arm_ethernet/stage1.yaml`

**Dependencies:** `modeling_rltoken.py`, lerobot `LeRobotDataset`, standard training utilities.

---

### 4.14 `scripts/serve_rlt_policy.py`

**Purpose:** Machine A WebSocket server. Loads frozen pi0.5 + RLToken, serves features.

**Key structure:**

```python
class RLTInferenceServer:
    """
    Loads PI05RLT + RLTokenModel from checkpoint.
    WebSocket handler:
      Receives: msgpack dict {"images": {...}, "state": array, "request_id": str}
      Returns:  msgpack dict {"z_rl": array, "ref_chunk": array, "proprio": array}
    Uses shared prefix: run PaliGemma backbone once → both RLTokenEncoder and VLA sample_actions.
    Applies AgilexBagImageInputs-equivalent (camera slot mapping) for SO-ARM cameras.
    """

async def ws_handler(websocket, path): ...
async def main(): ...  # asyncio.run(serve(ws_handler, host, port))
```

**Dependencies:** `policies/pi05_rlt/`, `policies/rltoken/`, `websockets`, `msgpack`.

---

### 4.15 `scripts/launch_machine_b_rlt.py`

**Purpose:** Spawn replay manager → learner → actor service, monitor for crashes.

Mirrors `rlt_online_rl/launch/launch_machine_b.py` but invokes the lerobot versions.

```python
ROLES = ["replay_manager", "learner_service", "actor_service"]
# spawn: python -m lerobot.rl.rltoken.run_role --role <role> --config <yaml>
```

---

### 4.16 `scripts/launch_robot_rollout_rlt.py`

**Purpose:** Wait for actor and replay health checks, then start `RolloutOrchestrator`.

---

### 4.17 `scripts/launch_actor_eval_rlt.py`

**Purpose:** Start actor service only (no learner/replay), run rollout in eval mode.

---

### 4.18 Config files

**`configs/rltoken/base.yaml`** — Default values matching the dataclass defaults in `config.py`.

**`configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml`** — SO-ARM Ethernet insertion overrides:

```yaml
rl:
  action_dim: 6
  proprio_dim: 6
  chunk_len: 10
  fixed_std: 0.002
  delta_weight: 10.0
  warmup_min_size: 600
  warmup_post_collect_updates: 20000
  action_norm_stats_path: ./configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json

env_driver:
  machine_a_ws_url: "ws://MACHINE_A_IP:8000"
  control_frequency_hz: 20.0
  chunk_exec_horizon: 10
  task_mode: critical_phase

runtime:
  run_dir: ./runs/so_arm_ethernet
```

**`configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json`** — Pre-computed from SO-ARM demonstration data:

```json
{
  "actions": {
    "q01": [<6 floats, delta space lower bound>],
    "q99": [<6 floats, delta space upper bound>]
  }
}
```

---

## 5. Inter-File Dependency Graph

```
                         ┌──────────────────────────┐
                         │  transition.py            │
                         │  (RLTTransition, Source)  │
                         └──────────┬───────────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
          ┌─────────▼──────┐  ┌─────▼──────┐  ┌─────▼───────────┐
          │  config.py     │  │  action_   │  │  rl_token.py    │
          │  (dataclasses) │  │  repr.py   │  │  (Encoder/Dec.) │
          └────────┬───────┘  └─────┬──────┘  └─────┬───────────┘
                   │                │                │
          ┌────────▼───────────┐   │          ┌─────▼────────────────┐
          │  replay.py         │◀──┘          │  configuration_      │
          │  (Buffer+HTTP)     │              │  rltoken.py          │
          └────────┬───────────┘              └─────┬────────────────┘
                   │                                │
          ┌────────▼───────────┐          ┌─────────▼───────────────┐
          │  networks.py       │          │  modeling_pi05_rlt.py   │
          │  (Actor, Critic)   │          │  (PI05RLT subclass)     │
          └────────┬───────────┘          └─────────┬───────────────┘
                   │                                │
          ┌────────▼───────────────────────────┐   │
          │  trainer.py                         │   │
          │  (RLTTrainState, LearnerService)    │   │
          └────────┬────────────────────────────┘   │
                   │                                 │
          ┌────────▼────────────┐          ┌─────────▼──────────────────┐
          │  inference.py       │          │  modeling_rltoken.py        │
          │  (ActorService,     │          │  (RLTokenPolicy, Stage 1)   │
          │   EnvDriver)        │          └─────────┬──────────────────┘
          └────────┬────────────┘                    │
                   │                        ┌────────▼───────────┐
          ┌────────▼────────────┐           │  scripts/           │
          │  rollout.py         │           │  train_rlt.py       │
          │  (SOArmEnv,         │           └────────────────────┘
          │   RolloutOrch.)     │
          └────────┬────────────┘
                   │
          ┌────────▼────────────────────┐
          │  scripts/                   │
          │  launch_robot_rollout_rlt.py│
          │  launch_machine_b_rlt.py    │
          │  serve_rlt_policy.py        │
          └─────────────────────────────┘
```

---

## 6. Development Order

The order prioritizes: (a) no upward dependencies, (b) testability at each step, (c) end-to-end verifiability as early as possible.

### Step 1 — Foundation (no dependencies, unit-testable in isolation)

| File | Why first |
|---|---|
| `rl/rltoken/transition.py` | Pure data structure, no imports from new code. Defines the vocabulary everything else uses. |
| `rl/rltoken/config.py` | All other files import configs. Can be validated with a round-trip YAML test immediately. |
| `rl/rltoken/action_representation.py` | Pure math. Test with synthetic SO-ARM joint arrays and verify delta/denormalize roundtrip. |

**Verification:** Unit tests for config YAML round-trip, delta normalization/denormalization invertibility on synthetic data.

---

### Step 2 — RLToken model (no online RL deps)

| File | Why second |
|---|---|
| `policies/rltoken/rl_token.py` | Standalone PyTorch module. No lerobot imports. Verify MSE loss decreases on random prefix tensors. |
| `policies/rltoken/configuration_rltoken.py` | Needed before `modeling_rltoken.py`. |

**Verification:** `RLTokenModel(config).loss(random_prefix, mask)` produces a finite scalar; encoder/decoder shapes are correct.

---

### Step 3 — VLA extension (requires reading PI05Pytorch internals)

| File | Why third |
|---|---|
| `policies/pi05_rlt/modeling_pi05_rlt.py` | Requires understanding PI05Pytorch's forward pass. Must be done before Stage 1 training. |

This step requires the most code reading: inspect `PI05Pytorch.__init__`, `PI05Pytorch.forward()`, the PaliGemma model architecture, and identify where prefix embeddings live. Use a forward hook or override; do not import from or modify the original file.

**Verification:** Load a pi0.5 checkpoint, call `extract_prefix_embeddings()` on a dummy batch, verify shape `[B, seq, 2048]` and that returned embeddings match what you'd get from a manual hook.

---

### Step 4 — Stage 1 training pipeline

| File | Why fourth |
|---|---|
| `policies/rltoken/modeling_rltoken.py` | Combines pi0.5 extension + RLToken model into a trainable policy. |
| `scripts/train_rlt.py` | End-to-end Stage 1 training script. |

**Verification:** Run `train_rlt.py` for 100 steps on a small SO-ARM demo dataset (even 10 episodes). Verify RLT loss decreases. If `alpha=0`, verify VLA parameters do not change (compare checksums before/after). Save and reload checkpoint.

---

### Step 5 — Online RL networks

| File | Why fifth |
|---|---|
| `rl/rltoken/networks.py` | Actor and critic can be tested with random tensors before any replay or training infrastructure. |

**Verification:** `ChunkActor.forward(random_z_rl, random_proprio, random_ref_chunk)` → shape `[B, 10, 6]`. `TwinCritic.q_values(...)` → two scalars. `build_td_target(...)` → shape `[B, 1]`, non-negative for positive rewards.

---

### Step 6 — Replay buffer

| File | Why sixth |
|---|---|
| `rl/rltoken/replay.py` | Depends on `transition.py`, `config.py`, `action_representation.py`. Test before introducing HTTP. |

**Verification:** Add 1000 synthetic transitions, sample 128, verify stratified sampling draws proportionally. Shut down and restart `ReplayManager` — verify journal recovery restores all transitions.

---

### Step 7 — Learner service

| File | Why seventh |
|---|---|
| `rl/rltoken/trainer.py` | Depends on networks + replay. Can be tested with a local `ReplayBuffer` (no HTTP needed). |

**Verification:** Fill buffer with 700 synthetic transitions, call `LearnerService.train_once()` in a loop for 1000 steps. Verify actor loss decreases, critic loss converges. Verify `learner_status.json` is written after warmup threshold.

---

### Step 8 — Actor service and inference layer

| File | Why eighth |
|---|---|
| `rl/rltoken/inference.py` | Depends on networks + config. Test `ActorService` locally with a saved snapshot. |

**Verification:** Start `ActorService` in a thread, send a synthetic `ActorRequest`, receive `ActorResponse`. Verify hot-reload: overwrite snapshot file, send another request, verify actor version incremented.

---

### Step 9 — Machine A serving script

| File | Why ninth |
|---|---|
| `scripts/serve_rlt_policy.py` | Requires Stage 1 checkpoint from Step 4 + `PI05RLT` from Step 3. |

**Verification:** Start server, connect via `websockets` client, send a dummy observation, receive `z_rl` of shape `[2048]` and `ref_chunk` of shape `[50, 6]`. Use `fake_machine_a.py` equivalent for integration testing before bringing real VLA.

---

### Step 10 — Robot rollout (SO-ARM)

| File | Why tenth |
|---|---|
| `rl/rltoken/rollout.py` | Last to implement because it needs all upstream pieces and the physical robot. |

**Verification order:**
1. Test `SOArmEnv` standalone (without inference): connect SO-ARM, call `get_observation()`, verify shapes, call `execute_chunk` with a fixed absolute chunk.
2. Test `KeyboardSignalBridge` standalone: verify keys are received correctly.
3. Test `RolloutPhaseController` standalone: mock `learner_status.json` transitions.
4. Test full `RolloutOrchestrator` with `fake_machine_a.py` and a random actor (no real VLA, no trained model).

---

### Step 11 — Launch scripts and config files

| File | Why last |
|---|---|
| `scripts/launch_machine_b_rlt.py` | Trivial once all components work. |
| `scripts/launch_robot_rollout_rlt.py` | Same. |
| `configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml` | Fill in concrete hyperparameters after Step 10 testing. |
| `configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json` | Compute from collected SO-ARM demos. |

---

### Step 12 — Norm stats collection

Before the first online RL run, collect a small set of SO-ARM demonstrations and compute `q01/q99` quantile statistics for delta-space actions. Write a one-off script:

```
scripts/compute_norm_stats.py  →  configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json
```

This is a prerequisite for `action_representation.py` to work correctly on real data.

---

### Development Order Summary Table

| Step | File(s) | Test |
|---|---|---|
| 1 | `transition.py`, `config.py`, `action_repr.py` | Unit: YAML roundtrip, delta invertibility |
| 2 | `rl_token.py`, `configuration_rltoken.py` | Unit: shape + finite loss |
| 3 | `modeling_pi05_rlt.py` | Unit: prefix shape with real pi0.5 checkpoint |
| 4 | `modeling_rltoken.py`, `train_rlt.py` | Integration: Stage 1 loss decrease on demo data |
| 5 | `networks.py` | Unit: tensor shapes, TD target finite |
| 6 | `replay.py` | Unit: add/sample/journal recovery |
| 7 | `trainer.py` | Integration: loss convergence, status file write |
| 8 | `inference.py` | Integration: ActorService HTTP roundtrip |
| 9 | `serve_rlt_policy.py` | Integration: WebSocket roundtrip |
| 10 | `rollout.py` | Hardware: SO-ARM obs/action + fake Machine A |
| 11 | Launch scripts, YAML configs | System: full multi-process run |
| 12 | `norm_stats.json` | Data: collect demos, verify normalization |

---

## 7. SO-ARM Specifics

### Hardware interface
- `SOFollower` (lerobot) connects via USB serial using `FeetechMotorsBus`
- Motors: `shoulder_pan(1), shoulder_lift(2), elbow_flex(3), wrist_flex(4), wrist_roll(5), gripper(6)`
- Observation dict keys: `shoulder_pan.pos`, ..., `gripper.pos` (plus camera keys)
- Action dict keys: same as observation (position targets)

### Mapping to RLT tensors
```python
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
action_dim = 6
proprio_dim = 6

def obs_to_proprio(obs: dict) -> np.ndarray:
    return np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)

def chunk_to_robot_action(chunk_step: np.ndarray) -> dict:
    return {f"{m}.pos": float(chunk_step[i]) for i, m in enumerate(MOTOR_NAMES)}
```

### Delta action convention
- Dims 0–4 (shoulder through wrist_roll): delta from `state0` (joint position at chunk start)
- Dim 5 (gripper): absolute position (gripper behaves differently — open/close rather than continuous delta)

### Control frequency
- 20 Hz execution (same as openpi-RLT for Agilex)
- Chunk length 10 → each chunk covers 0.5 seconds of motion

### Camera configuration
- SO-ARM can use one or more USB cameras via lerobot `OpenCVCamera`
- Minimum: one wrist camera + one fixed overhead camera
- Keys will be `observation.images.wrist` and `observation.images.overhead` (or similar)
- Machine A serving must map these to pi0.5's expected slots (`base_0_rgb`, `left_wrist_0_rgb`, etc.)
- Create an `SOArmImageInputs` class (analogous to `AgilexBagImageInputs`) to handle this mapping

### Questions.md adaptation guidance
The openpi-RLT `questions.md` section on SO-ARM adaptation lists these steps:
1. Register SO-ARM dataset format in lerobot data config
2. Implement `SOArmImageInputs` for camera slot mapping
3. Adapt norm stats to 6-DOF space
4. Replace `pika_sync_ros.py` with `rollout.py` (no ROS)
5. Adjust reset poses to SO-ARM joint limits

---

## 8. Open Questions

The following must be resolved before or during implementation:

| # | Question | Impact |
|---|---|---|
| Q1 | What is the exact internal structure of `PI05Pytorch`? Where are PaliGemma's image prefix hiddens accessible? | Determines implementation of `extract_prefix_embeddings()` in Step 3 |
| Q2 | Does lerobot's pi0.5 use a shared or separate prefix for language tokens vs image tokens? | Determines whether `image_only=True` filtering applies the same way |
| Q3 | What cameras does the target SO-ARM setup have, and what are their lerobot config keys? | Required for `SOArmImageInputs` slot mapping in `serve_rlt_policy.py` |
| Q4 | Single machine or two machines? (VLA on same GPU as learner?) | If single machine: Machine A serving can be in-process; simplifies `serve_rlt_policy.py` |
| Q5 | Will Stage 1 training use existing SO-ARM LeRobot demonstrations or new collections? | Needed before `compute_norm_stats.py` and Stage 1 config |
| Q6 | What pi0.5 checkpoint should be loaded? HuggingFace Hub ID or local path? | Required for `RLTokenConfig.pi05_pretrained_path` |
| Q7 | Should the lerobot `LeRobotDataset` be used for Stage 1, or a custom dataloader? | If using `LeRobotDataset`, the data config mapping (Step 4) must be finalized first |

---

*Plan written 2026-10-14. Start with Step 1 after resolving Q1 and Q3.*
