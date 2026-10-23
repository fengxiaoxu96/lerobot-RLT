# RLT Integration — Developer Notes

Per-file implementation notes written as each file is completed. Each entry
covers: what the file does, key design choices, and any deviations from
the openpi-RLT reference.

Reference plan: [RLT_Integration_Plan.md](RLT_Integration_Plan.md)
Reference source: [openpi-RLT framework](../openpi-RLT/RLToken_Framework.md)

---

## Table of Contents

| Step | File | Status |
|------|------|--------|
| 1 | [`rl/rltoken/transition.py`](#step-1--rlrltokentransitionpy) | Done |
| 1 | `rl/rltoken/config.py` | Done |
| 1 | `rl/rltoken/action_representation.py` | Done |
| 2 | `policies/rltoken/rl_token.py` | Done |
| 2 | `policies/rltoken/configuration_rltoken.py` | Done |
| 3 | `policies/pi05_rlt/modeling_pi05_rlt.py` | Done |
| 4 | `policies/rltoken/modeling_rltoken.py` | Done |
| 4 | `scripts/train_rlt.py` | Done |
| 5 | `rl/rltoken/networks.py` | Done |
| 6 | `rl/rltoken/replay.py` | Done |
| 7 | `rl/rltoken/trainer.py` | Done |
| 8 | `rl/rltoken/inference.py` | Done |
| 9 | `scripts/serve_rlt_policy.py` | Done |
| 10 | `rl/rltoken/rollout.py` | Done |
| 11 | `scripts/launch_*.py` / `scripts/run_rlt_role.py` | Done |
| 12 | `configs/rltoken/base.yaml` / `online_rl.yaml` / `norm_stats_delta.json` | Done |

---

## Step 1 — `rl/rltoken/transition.py`

**File:** [src/lerobot/rl/rltoken/transition.py](src/lerobot/rl/rltoken/transition.py)

### Purpose

Defines the vocabulary of data structures that flow through the entire online
RL pipeline — from robot rollout through the replay buffer to the learner.
All other files in `rl/rltoken/` import from this module.  No lerobot
dependencies; only `numpy` and stdlib.

### Public API

#### `TransitionSource` (IntEnum)

Tags *how* the `action_chunk` in a transition was produced:

| Value | Meaning |
|-------|---------|
| `BASE = 0` | VLA `ref_chunk` used directly; actor not called (warmup or base-mode episodes) |
| `RL = 1` | Actor-refined chunk returned by `ActorService` |
| `HUMAN = 2` | Full teleop override for the entire chunk |
| `MIXED = 3` | Partial teleop mid-chunk (some steps actor, some human) |

The actor loss in `trainer.py` uses this per-step signal to select the
correct BC target: HUMAN/MIXED steps use the recorded `action_chunk` as
target; BASE/RL steps use `ref_chunk`.

#### `EpisodeStepRecord` (dataclass, slots)

Holds the raw data collected for **one control step** during `execute_chunk()`
in `rollout.py`. Fields are float32 numpy arrays; dtype compaction happens
only when promoting to `RLTTransition`.  At episode end, `replay.py` walks
the step list and groups steps into chunk-sized `RLTTransition` objects.

Key fields: `z_rl`, `proprio`, `ref_action`, `action`, `reward`, `done`,
`next_z_rl`, `next_proprio`, `source`, `collection_phase`, `success`,
`intervention_flag`, `episode_id`, `step_id`.

`from_mapping(data)` reconstructs from a plain dict (used when deserialising
from the raw episode pickle trace).

#### `RLTTransition` (dataclass, slots)

The **chunk-level** unit stored in `RLTReplayBuffer`.  Aggregates
`chunk_len` steps into a single record:

| Field | Shape | dtype (storage) | Notes |
|-------|-------|-----------------|-------|
| `z_rl` | `[z_dim]` | float16 | Compressed VLA state token |
| `proprio` | `[proprio_dim]` | float32 | Joint positions at chunk start |
| `ref_chunk` | `[chunk_len, action_dim]` | float16 | VLA reference actions |
| `action_chunk` | `[chunk_len, action_dim]` | float16 | Actions sent to robot |
| `rewards` | `[chunk_len]` | float32 | Per-step rewards (0 except terminal) |
| `done` | scalar | bool | True if episode ended inside chunk |
| `next_z_rl` | `[z_dim]` | float16 | VLA state after chunk |
| `next_proprio` | `[proprio_dim]` | float32 | Joint positions after chunk |
| `next_ref_chunk` | `[chunk_len, action_dim]` | float16 | VLA reference after chunk |
| `source` | scalar | uint8 | Dominant `TransitionSource` |
| `source_chunk` | `[chunk_len]` | uint8 | Per-step `TransitionSource` |
| `collection_phase` | string | — | `"warmup"` / `"online"` |
| `success` | int8 | — | 1 if episode was labelled success |
| `intervention_flag` | bool | — | Any human step in chunk |
| `episode_id` | int32 | — | Monotonic episode counter |
| `step_id` | int32 | — | Step index of first step in chunk |

`to_numpy()` compacts large tensors to float16 for memory efficiency in the
200k-transition ring buffer (~5 GB uncompressed for z_dim=2048).

`to_journal_record()` adds the raw `collection_phase` string on top of
`to_numpy()` so the journal can be filtered by phase after a crash.

`from_mapping(data)` reconstructs from a stored dict with safe defaults for
optional fields added in later versions (`source_chunk`, `intervention_flag`).

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Module location | `rlt_online_rl/src/rlt_online_rl/replay.py` | `src/lerobot/rl/rltoken/transition.py` |
| Split out from | Lived inside `replay.py` | Extracted to its own file so `replay.py`, `trainer.py`, and `inference.py` can all import without circular deps |
| `RLTTransition` type | `@dataclass(slots=True)` | Same — keeps `__slots__` for memory efficiency |
| Framework | JAX/NumPy | NumPy only (no JAX); dtype handling via `np.asarray` |
| `to_numpy` dtypes | float16 for big tensors | Identical |
| `collection_phase_id` | int stored in `to_numpy()` | Same; human-readable string kept separately in `to_journal_record()` |

---

## Step 1 — `rl/rltoken/config.py`

**File:** [src/lerobot/rl/rltoken/config.py](src/lerobot/rl/rltoken/config.py)

### Purpose

Defines every configuration dataclass for the online RL system and provides
the YAML load/save API.  All other `rl/rltoken/` files receive a config
object rather than reading files themselves, so this module is the single
source of truth for defaults and path resolution.

The file has three logical sections:

1. **Seven frozen dataclasses** — one per service / concern.
2. **Path resolution helpers** — `_resolve_path` / `_relativize_path` make
   the config file portable: relative paths are stored on disk but resolved to
   absolute paths at load time so callers never have to do `os.path.join`.
3. **YAML round-trip** — `load_system_config_yaml` / `save_system_config_yaml`
   plus internal helpers for merging deep-nested YAML dicts and coercing YAML
   scalars to the annotated Python types.

### Public API

#### Dataclasses

| Class | Covers |
|-------|--------|
| `RLTOnlineRLConfig` | Algorithm: action dims, RL discount, loss weights, network sizes, learning rates, warmup budget |
| `ActorServiceConfig` | B1: bind host/port, snapshot path, hot-reload interval |
| `LearnerServiceConfig` | B2: push-actor interval, checkpoint interval, batch size, replay URL |
| `ReplayConfig` | B3: ring-buffer capacity, journal path, port, stratified sampling ratios |
| `EnvDriverConfig` | B4: Machine A WebSocket URL, control frequency, task mode, reset poses, timeouts |
| `MonitoringConfig` | W&B project/entity/mode/run-name |
| `OnlineRLSystemConfig` | Top-level container; also holds `role` (which process to start) and `local_debug_mode` |

`OnlineRLSystemConfig.role` selects which process `run_role.py` will launch:
`"all"` | `"actor_service"` | `"learner_service"` | `"replay_manager"` | `"env_driver"`.

All dataclasses are `frozen=True` — mutation uses `dataclasses.replace(...)`,
which prevents accidental in-place modification deep in the call stack.

#### Path fields and their resolution

| Dataclass | Path field | Resolved against |
|-----------|------------|-----------------|
| `RLTOnlineRLConfig` | `action_norm_stats_path` | config file dir, then repo root |
| `ActorServiceConfig` | `snapshot_path` | config file dir (output, no exist check) |
| `LearnerServiceConfig` | `checkpoint_dir`, `actor_snapshot_path` | same |
| `ReplayConfig` | `journal_path` | same |
| `MonitoringConfig` | `wandb_dir` | same |

Output directories (`checkpoint_dir`, `journal_path`, etc.) use
`require_exists=False` so they don't need to pre-exist.  Input paths
(`action_norm_stats_path`) use `require_exists=True` and raise if missing.

#### `load_system_config_yaml(path, *, base=None)`

1. Loads the YAML file with `yaml.safe_load`.
2. Deep-merges it on top of the serialised `base` (default: dataclass
   defaults).  Any field omitted from the YAML retains its default value.
3. Accepts two YAML layouts:
   - **Grouped** (`experiment:` / `runtime:` sections) — the canonical format
     written by `save_system_config_yaml`.
   - **Flat** (all keys at the top level) — for backward compatibility.
4. Resolves all path fields before returning.

#### `save_system_config_yaml(system, path)`

Relativises all absolute paths before serialising so the YAML file is
self-contained and can be moved with its run directory.

### SO-ARM defaults vs openpi-RLT defaults

| Field | openpi-RLT default | lerobot-RLT default | Reason |
|-------|--------------------|---------------------|--------|
| `action_dim` | 7 | **6** | SO-ARM has 6-DOF |
| `proprio_dim` | 7 | **6** | Same |
| `action_representation` | `"abs"` | **`"delta_chunk"`** | RLT paper recommends delta for smooth RL |
| `fixed_std` | 0.05 | **0.002** | Near-deterministic actor (from Ethernet task config) |
| `delta_weight` | 0.0 | **10.0** | Smoothness penalty enabled by default |
| `warmup_bc_weight` | 1.0 | **10.0** | Stronger BC supervision during warmup |
| `warmup_min_size` | 1000 | **600** | Fewer demos needed for 6-DOF |
| `control_frequency_hz` | 50.0 | **20.0** | SO-ARM USB latency is ~50 ms |
| `push_actor_interval_steps` | 50 | **500** | Less aggressive push reduces I/O overhead |
| `sample_batch_size` | 256 | **128** | Smaller batch fits single-GPU setup |
| `sample_strategy` | `"uniform"` | **`"stratified"`** | Better data balance for mixed sources |

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| `_resolve_path` anchor resolution | 3-way: absolute → config-dir → project-root | Same logic, simplified variable names |
| Grouped YAML format | `experiment` / `runtime` split | Identical; flat layout also accepted |
| `_dataclass_from_mapping` | Full type coercion with `Union` handling | Ported verbatim; renamed to `_dataclass_from_dict` |
| `MonitoringConfig.wandb_run_name` | `"online-rl"` | `"so-arm-online-rl"` |
| `_REPO_ROOT` derivation | `parents[2]` of `config.py` in `rlt_online_rl/src/` | `parents[4]` of `config.py` in `src/lerobot/rl/rltoken/` |

---

## Step 2 — `policies/rltoken/rl_token.py`

**File:** [src/lerobot/policies/rltoken/rl_token.py](src/lerobot/policies/rltoken/rl_token.py)

### Purpose

PyTorch implementation of the RLToken encoder-decoder bottleneck.  Ported
from the JAX/Flax reference at `openpi-RLT/src/openpi/models/rl_token.py`.

The file has four sections:

1. **`_sinusoidal_pe`** — helper that produces a `[seq_len, d_embed]`
   sinusoidal positional encoding tensor, matching the JAX `sinusoidal_pe_init`.
2. **Private building blocks** — `_GeGLUGate` and `_CrossAttentionBlock`
   (prefixed `_` to signal they are not part of the public API).
3. **`RLTokenConfig`** — frozen architecture dataclass.
4. **Public modules** — `RLTokenEncoder`, `RLTokenDecoder`, `RLTokenModel`.

### Public API

#### `RLTokenConfig`

```python
@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    num_rl_tokens: int = 1      # number of compressed state tokens
    num_layers:    int = 2      # cross-attention layers in encoder & decoder
    embed_dim:     int = 2048   # internal dimension
    input_dim:     int = 2048   # VLA prefix hidden size (PaliGemma 2B = 2048)
    num_heads:     int = 8
    mlp_ratio:     float = 4.0
    dropout_rate:  float = 0.0
    max_seq_len:   int = 1024   # pre-allocated PE length (actual seq <= this)
```

Note: this is *not* a draccus ChoiceRegistry config.  The draccus policy config
lives in `configuration_rltoken.py` and holds a superset of these fields.

#### `RLTokenEncoder`

Compresses `prefix_embs [B, seq_len, input_dim]` into `rl_tokens [B, num_rl_tokens, embed_dim]`.

Learnable parameters:
- `q_embed [1, num_rl_tokens, embed_dim]` — the "probe" queries, sinusoidally
  initialized.  These are the most important learnable part: they define which
  information the RL tokens will absorb from the prefix.
- `kv_pos_enc [1, max_seq_len, embed_dim]` — positional encoding added to the
  prefix before cross-attention; sinusoidally initialized, learnable.
- (optional) `input_proj` linear if `input_dim != embed_dim`.

Forward signature:
```python
def forward(prefix_embs: Tensor, prefix_mask: Tensor | None) -> Tensor
# prefix_mask: True = valid  (same as JAX convention; inverted internally for PyTorch)
```

#### `RLTokenDecoder`

Reconstructs `[B, target_seq_len, input_dim]` from `rl_tokens`.
Used *only during Stage 1 training* — never called at inference time.

Learnable parameters:
- `q_embed [1, max_seq_len, embed_dim]` — one query per reconstructed position;
  sliced to `target_seq_len` at forward time.
- `kv_pos_enc [1, num_rl_tokens, embed_dim]` — positional encoding for the
  (fixed-size) RL token KV.
- (optional) `output_proj` linear if `input_dim != embed_dim`.

#### `RLTokenModel`

Thin wrapper that owns encoder + decoder and exposes:

| Method | Used by | Description |
|--------|---------|-------------|
| `encode(prefix_embs, prefix_mask)` | Machine A serving | Encoder only; returns `[B, num_rl_tokens, embed_dim]` |
| `loss(prefix_embs, prefix_mask)` | Stage 1 training | Encode → decode → masked MSE; returns `(scalar, {"mse": scalar})` |

The `loss()` method calls `.detach()` on the target as a safety net, but the
*caller* (`RLTokenPolicy.forward`) is responsible for passing already-detached
prefix embeddings (equivalent to `jax.lax.stop_gradient`).

### Building blocks

#### `_GeGLUGate`

Faithful PyTorch port of the JAX `GeGLU` class:

```
input [B, seq, d]  →  Linear(d, 2d)  →  split  →  x * gelu(gate)  →  [B, seq, d]
```

Used inside `_CrossAttentionBlock`'s MLP after the initial expansion linear.

#### `_CrossAttentionBlock`

Mirrors JAX `CrossAttentionLayer`:

```
x, kv  →  LN + Self-Attn(x,x)   + residual
       →  LN + Cross-Attn(x,kv) + residual   (key_padding_mask for variable kv_len)
       →  LN + Linear(d→d_ff) + Dropout + GeGLU + Linear(d_ff→d) + residual
```

Uses `nn.MultiheadAttention(batch_first=True)` for both attention sub-layers.
`kv_padding_mask` follows PyTorch convention (`True = ignore`), so the encoder
inverts the `prefix_mask` before passing it.

### JAX → PyTorch translation notes

| JAX concept | PyTorch equivalent | Notes |
|-------------|-------------------|-------|
| `nn.Module` with `@nn.compact` | `nn.Module` with `__init__` + `forward` | Explicit sub-module registration |
| `self.param(name, init_fn, shape)` | `nn.Parameter(init_tensor)` | Parameters in encoder/decoder are sinusoidally initialized `nn.Parameter` |
| `jax.lax.stop_gradient(x)` | `x.detach()` | Applied to `prefix_embs` in `loss()` |
| `nn.MultiHeadDotProductAttention` | `nn.MultiheadAttention(batch_first=True)` | Mask convention inverted (JAX True=attend, PyTorch True=ignore) |
| `jnp.broadcast_to(x, ...)` | `tensor.expand(B, -1, -1)` | Query broadcast over batch |
| Dynamic param shapes at `@compact` time | `max_seq_len` pre-allocation + slice | Only difference: unused PE entries for `seq < max_seq_len` |

### New field: `max_seq_len`

The JAX implementation creates `q_embed` and `kv_pos_enc` with shapes derived
from the actual runtime input (`prefix_embs.shape[-2:]`).  In PyTorch,
`nn.Parameter` dimensions must be fixed at construction time.

Solution: pre-allocate both parameters for `max_seq_len` positions and slice
to `[:, :seq_len, :]` at each forward call.  The unused entries accumulate
zero gradient during training and are harmless.

`max_seq_len=1024` is generous for PaliGemma with 3 cameras at 224×224 / patch-14,
which produces ~768 image tokens.  Reduce to the exact prefix length for tighter
memory usage if needed.

### Naming note

`RLTokenConfig` in this file is a plain `@dataclass(frozen=True)`.  The draccus
policy config in `configuration_rltoken.py` is named `RLTokenPolicyConfig` to
avoid an import-time clash when both are used in `modeling_rltoken.py`.

---

## Step 2 — `policies/rltoken/configuration_rltoken.py`

**File:** [src/lerobot/policies/rltoken/configuration_rltoken.py](src/lerobot/policies/rltoken/configuration_rltoken.py)

### Purpose

draccus-registered policy config for **Stage 1** (offline RLToken training).  Acts as the
single source of truth for all hyperparameters that `RLTokenPolicy` (in `modeling_rltoken.py`)
and `train_rlt.py` consume.  Inherits `PreTrainedConfig` (`draccus.ChoiceRegistry` +
`HubMixin`) so it can be serialised / pushed to the Hub exactly like any other lerobot policy.

### Public API

#### `RLTokenPolicyConfig` (registered as `"rltoken"`)

| Group | Fields |
|-------|--------|
| VLA backbone | `pi05_pretrained_path`, `alpha`, `image_only_prefix` |
| RLToken architecture | `num_rl_tokens`, `num_layers`, `embed_dim`, `input_dim`, `num_heads`, `mlp_ratio`, `dropout_rate`, `max_seq_len` |
| Action / obs space | `action_dim`, `action_horizon`, `chunk_len`, `image_resolution` |
| Normalization | `normalization_mapping` (VISUAL=IDENTITY, STATE/ACTION=QUANTILES) |
| Optimizer | `optimizer_lr/betas/eps/weight_decay/grad_clip_norm` |
| Scheduler | `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr` |

`alpha` controls the VLA gradient contribution:
- `alpha = 0.0` — VLA is frozen; only RLToken parameters receive gradients.
- `0 < alpha <= 1.0` — joint training; VLA loss is added at weight `alpha`.

`max_seq_len` mirrors the field added to `RLTokenConfig` in `rl_token.py` to pre-allocate
fixed-size positional encodings.  Both must be kept in sync; the policy copies this value when
constructing `RLTokenModel`.

#### `get_optimizer_preset()` / `get_scheduler_preset()`

Return `AdamWConfig` and `CosineDecayWithWarmupSchedulerConfig` instances ready for
`make_optimizer_and_scheduler()` in lerobot's training utilities.

#### `validate_features()`

Checks that at least one `VISUAL` input feature is present (Stage 1 requires images),
then inserts default `OBS_STATE` and `ACTION` features if the dataset did not provide them.

#### `action_delta_indices`

Returns `list(range(action_horizon))` — signals to `DataProcessorPipeline` that all
`action_horizon` steps are available as chunk targets, matching pi0.5 convention.

### Naming note

The class is `RLTokenPolicyConfig` (not `RLTokenConfig`) to avoid a name clash with
the plain architecture dataclass `RLTokenConfig` defined in `rl_token.py`.  Both live
in `modeling_rltoken.py` imports; keeping separate names prevents an attribute-override
collision at draccus registration time.

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Config format | Python dataclass, no registration | draccus `ChoiceRegistry` (`"rltoken"` key) |
| VLA path | Hardcoded or env var | `pi05_pretrained_path` field |
| Normalization | Handled by pipeline step outside config | `normalization_mapping` inside config |
| Optimizer | Configured separately in train script | `get_optimizer_preset()` / `get_scheduler_preset()` |
| Image resolution | Fixed 224×224 in preprocessing | `image_resolution` field (default 224×224) |

---

## Step 3 — `policies/pi05_rlt/modeling_pi05_rlt.py`

**File:** [src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py](src/lerobot/policies/pi05_rlt/modeling_pi05_rlt.py)

### Purpose

Subclasses `PI05Policy` to add two new forward-pass methods that expose the VLA's internal
prefix embeddings.  These are the only new entry points; all existing behaviour is
inherited unchanged.  No existing files are touched.

`RLTokenPolicy` (in `modeling_rltoken.py`) holds an instance of this class as `self.vla`
and calls one of the two methods depending on `RLTokenPolicyConfig.alpha`:

| `alpha` | Called method |
|---------|---------------|
| 0 (VLA frozen) | `extract_prefix_embeddings` |
| > 0 (joint training) | `compute_loss_with_prefix` |

### Public API

#### `extract_prefix_embeddings(batch, image_only=True) -> (prefix_embs, prefix_mask)`

| Arg | Shape | Notes |
|-----|-------|-------|
| `prefix_embs` | `[B, seq_len, 2048]` | float32 / bfloat16 (matches VLA dtype) |
| `prefix_mask` | `[B, seq_len]` | True = valid token |

Implementation:
1. Calls `self._preprocess_images(batch)` (inherited from `PI05Policy`).
2. Calls `self.model._embed_images(images, img_masks)` — this respects both the
   per-camera gradient-checkpointing path (training) and the fused batched path
   (inference), exactly as `PI05Pytorch.embed_prefix` does.
3. For each camera: stacks `[B, n_tok, hidden]` embedding + `[B, n_tok]` validity mask.
4. If `image_only=False`, appends language token embeddings (via
   `embed_language_tokens`) and the batch attention mask.
5. Returns concatenated tensors.

`image_only=True` is the default because `z_rl` is meant to compress the visual
state; language tokens are conditioning context, not state.

#### `compute_loss_with_prefix(batch, image_only=True) -> (vla_loss, prefix_embs, prefix_mask)`

1. Calls `super().forward(batch)` — runs the full VLA flow-matching forward with
   gradients flowing through VLA parameters.  Returns scalar `vla_loss`.
2. Calls `extract_prefix_embeddings` under `torch.no_grad` to obtain detached prefix
   embeddings for the RLToken encoder.

The two-pass design (vision tower called twice) is intentional:

- **Correctness**: VLA gradients must flow through the first pass (flow-matching loss).
  The second pass needs no gradients because the caller always calls
  `prefix_embs.detach()` before feeding to `RLTokenModel.loss()`.
- **Efficiency**: The double-vision-tower cost only applies when `alpha > 0` (joint
  training, rare configuration).  When `alpha = 0`, `extract_prefix_embeddings` is
  called directly (single pass, `no_grad`).
- **Simplicity**: Avoids inlining `PI05Pytorch.forward()` logic, keeping the subclass
  minimal and not fragile against parent changes.

### Why subclass `PI05Policy` (not `PI05Pytorch`)?

`PI05Policy` provides the full preprocessing pipeline: `_preprocess_images()`,
`_prepare_memory_states()`, `prepare_action()`, and `super().forward()`.  Accessing
these directly avoids duplicating or re-implementing the preprocessing logic.
`PI05Pytorch` is the bare `nn.Module` core; it is reached via `self.model` as usual.

### Mask convention

`prefix_mask` follows the **True = valid** convention (same as the JAX reference and
`RLTokenEncoder`'s expected input).  The `img_mask` tensors from lerobot can be 1-D
`[B]` (single frame) or 2-D `[B, T]` (MEM video); the implementation takes the last
frame's mask in the video case, consistent with `embed_prefix()` in the parent.

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Hook point | `PaliGemmaWithExpertModel.forward` with `prefix_only=True` flag | Direct call to `PI05Pytorch._embed_images` (no flag added to parent) |
| Language token inclusion | Always included in VLA prefix | Optional via `image_only` param; default True |
| Gradient flow | JAX `stop_gradient` on prefix before RLT loss | `torch.no_grad()` in second pass; caller also calls `.detach()` as a safety net |
| Class registration | Not a draccus policy; used as sub-module | Same — `PI05RLT` is not `@PreTrainedConfig.register_subclass`-registered |

---

## Step 4 — `policies/rltoken/modeling_rltoken.py`

**File:** [src/lerobot/policies/rltoken/modeling_rltoken.py](src/lerobot/policies/rltoken/modeling_rltoken.py)

### Purpose

The **Stage 1 trainable policy**.  Wraps two sub-modules:

| Attribute | Type | Role |
|-----------|------|------|
| `self.vla` | `PI05RLT` | VLA backbone; frozen or jointly fine-tuned |
| `self.rlt_model` | `RLTokenModel` | RLToken encoder + decoder |

At inference time the same policy is used by Machine A:
- `encode(batch)` → `z_rl [B, num_rl_tokens, embed_dim]`
- `predict_action_chunk(batch)` → `ref_chunk [B, action_horizon, action_dim]`

### Public API

#### `__init__(config: RLTokenPolicyConfig)`

1. Builds `RLTokenConfig` (plain frozen dataclass) from `config` fields and creates `RLTokenModel`.
2. Loads or constructs `PI05RLT`:
   - `config.pi05_pretrained_path` non-empty → `PI05RLT.from_pretrained(path)` (handles openpi key-remapping).
   - Empty path → `PI05RLT(_build_pi05_config(config))` (fresh init; for tests / ablations).
3. If `config.alpha == 0.0` calls `_freeze_vla()`.

#### `forward(batch) -> (loss, log_dict)`  *(Stage 1 training)*

| `alpha` | Code path |
|---------|-----------|
| `0.0` | `no_grad` prefix extraction → `rlt_model.loss` → return `rlt_loss` |
| `> 0` | `vla.compute_loss_with_prefix` → `rlt_model.loss` → return `rlt_loss + α*vla_loss` |

`rlt_model.loss` calls `.detach()` internally, so no VLA gradient flows via the RLT reconstruction path regardless of `alpha`.

#### `encode(batch) -> Tensor`

Calls `vla.extract_prefix_embeddings` then `rlt_model.encode`; wrapped in `torch.no_grad`.
Returns `[B, num_rl_tokens, embed_dim]`.  The serving script flattens to `[B, z_dim]`.

#### `predict_action_chunk(batch) -> Tensor`

Delegates to `self.vla.predict_action_chunk` (full VLA flow-matching inference).
Returns `[B, action_horizon, action_dim]`; the serving script slices to `chunk_len` steps.

#### `select_action(batch) -> Tensor` / `reset()`

Both delegate to `self.vla` so that the VLA's action queue and episode-boundary bookkeeping
are used.  lerobot's `drop_queued_actions` should call `self.vla.reset()` if invoked.

#### `freeze_vla()` / `unfreeze_vla()`

Public helpers called by `train_rlt.py` for curriculum-style training schedules (e.g., start
with VLA frozen, then unfreeze for fine-tuning after a threshold).

### Helper: `_build_pi05_config`

When `pi05_pretrained_path` is empty, this helper derives a `PI05Config` from
`RLTokenPolicyConfig` fields:

| PI05Config field | Source |
|-----------------|--------|
| `chunk_size` | `config.action_horizon` |
| `n_action_steps` | `config.action_horizon` |
| `max_state_dim` | `max(config.action_dim, 32)` (pi0.5 pads to ≥32) |
| `max_action_dim` | same |
| `image_resolution` | `config.image_resolution` |
| `input_features` | copied from `config.input_features` |
| `output_features` | copied from `config.output_features` |

### Checkpoint round-trip

`save_pretrained` (inherited from `PreTrainedPolicy`) serialises all parameters under
their natural keys: `vla.*` (VLA weights) and `rlt_model.*` (RLToken weights).  When a
`RLTokenPolicy` checkpoint is loaded via `from_pretrained`, the VLA is first
constructed (possibly from scratch if `pi05_pretrained_path` is empty), then all
parameters are overwritten by the saved state dict in one `load_state_dict` call.

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Policy container | No `PreTrainedPolicy` — loose JAX functions | `PreTrainedPolicy` subclass; Hub-compatible |
| VLA loading | JAX parameter restore | `PI05RLT.from_pretrained()` with key remapping |
| Gradient gating | `jax.lax.stop_gradient` inside loss fn | `torch.no_grad()` + `rlt_model.loss` internal `.detach()` |
| Action serving | Separate `serve.py` functions | `encode()` + `predict_action_chunk()` on the same policy object |
| `select_action` | Not applicable (online RL only) | Delegates to VLA for offline-eval compatibility |

---

## Step 4 — `scripts/train_rlt.py`

**File:** [scripts/train_rlt.py](scripts/train_rlt.py)

### Purpose

Standalone Stage 1 training entry point.  Orchestrates dataset loading,
policy construction, optimizer / scheduler creation, and the training loop.
Intentionally simpler than `lerobot-train` (no distributed, no EMA, no
PEFT, no env eval) — Stage 1 only needs a standard single-GPU SGD loop.

### Config: `RLTTrainConfig`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `policy` | `RLTokenPolicyConfig` | defaults | Full policy config |
| `dataset` | `DatasetConfig` | `repo_id=""` | LeRobot dataset |
| `steps` | `int` | 5 000 | Total training steps |
| `batch_size` | `int` | 32 | |
| `num_workers` | `int` | 4 | DataLoader workers |
| `seed` | `int` | 42 | |
| `log_every_n_steps` | `int` | 100 | Console + JSONL log interval |
| `save_every_n_steps` | `int` | 1 000 | `policy.save_pretrained` interval |
| `output_dir` | `str` | `"./runs/rlt_stage1"` | |
| `wandb` | `WandBConfig` | disabled | Optional W&B logging |

Parsed by `draccus.wrap()` — supports both `--key=value` CLI flags and
a `--config path/to/stage1.yaml` config file.

### Key steps inside `train(cfg)`

1. **Feature resolution** — `_resolve_policy_features(policy_cfg, ds_meta)` calls
   `dataset_to_policy_features(ds_meta.features)` to populate
   `input_features` and `output_features` on the policy config.  This replicates
   the feature-injection logic from `make_policy` without requiring a registered
   factory for `"rltoken"`.

2. **Delta-timestamps** — `resolve_delta_timestamps(policy_cfg, ds_meta)` converts
   `RLTokenPolicyConfig.action_delta_indices = range(action_horizon)` to float
   offsets at dataset FPS.  This ensures the dataloader loads a full `action_horizon`-
   step action chunk per sample.

3. **Preprocessor** — `make_pre_post_processors(policy.vla.config, ds_meta)` reuses
   the pi05 processor (text tokenisation + state/action normalisation) since
   `RLTokenPolicy` does not yet have its own registered processor.  Applies to every
   batch before calling `policy.forward`.

4. **Optimizer / scheduler** — built from `policy_cfg.get_optimizer_preset()` and
   `get_scheduler_preset()` (AdamW + cosine decay with warmup) over trainable params
   only.  When `alpha=0`, only `rlt_model.*` parameters have `requires_grad=True`.

5. **Training loop** — standard: forward → backward → grad-clip → optimizer step.
   `torch.nn.utils.clip_grad_norm_` uses `optimizer_grad_clip_norm` from the policy config.

6. **Checkpointing** — `policy.save_pretrained(ckpt_dir)` every `save_every_n_steps`.
   Saves both VLA weights and RLToken weights in a Hub-compatible safetensors layout.

7. **Logging** — console + `train_log.jsonl` (one record per `log_every_n_steps`);
   optional W&B via `cfg.wandb.enable=true`.

### Outputs

```
output_dir/
├── train_log.jsonl
├── checkpoint_001000/
│   ├── config.json          # RLTokenPolicyConfig
│   └── model.safetensors    # vla.* + rlt_model.* weights
├── checkpoint_002000/
│   └── ...
└── checkpoint_005000/
```

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Config system | Python dicts + Gin | draccus dataclass + YAML |
| Dataset | Custom TFRecord loader | `LeRobotDataset` with `delta_timestamps` |
| Preprocessor | Inline JAX transforms | lerobot `PolicyProcessorPipeline` (pi05) |
| Checkpointing | Orbax checkpoint | `PreTrainedPolicy.save_pretrained` (safetensors) |
| W&B | Built-in | Optional `WandBConfig.enable=true` |
| Distributed | JAX pmap | Not required for Stage 1; add `accelerate` if needed |

---

## Step 1 — `rl/rltoken/action_representation.py`

**File:** [src/lerobot/rl/rltoken/action_representation.py](src/lerobot/rl/rltoken/action_representation.py)

### Purpose

Translates between the robot's native absolute joint-position space and the
normalised representation that the RL actor operates in.  Two transforms are
composed in sequence:

1. **Delta encoding** — subtract the chunk-start joint state (`state0`) from
   the first `action_dim - 1` joints to produce relative deltas.  The last
   dim (gripper) is left absolute because its range is semantically different
   (open/close, not a continuous trajectory).
2. **Quantile normalisation** — linearly map the delta (or abs) representation
   to `[-1, 1]` using per-dim `q01` / `q99` statistics loaded from a JSON file.

No lerobot imports except `config.py`; torch is an optional dependency.

### Public API

#### `QuantileStats` (frozen dataclass)

Holds `q01` and `q99` as `[action_dim]` float32 arrays.

#### `_load_quantile_stats(path)`

Loads a `QuantileStats` from JSON.  Accepts two layouts:

| Layout | Key path |
|--------|----------|
| openpi-RLT canonical | `payload["norm_stats"]["actions"]["q01/q99"]` |
| flat (lerobot plan) | `payload["actions"]["q01/q99"]` |

#### `ActionRepresentationAdapter` (frozen dataclass)

| Method | Direction | Notes |
|--------|-----------|-------|
| `from_config(rl_config)` | — | Factory; returns `None` if `action_norm_stats_path` is not set |
| `normalize_chunk(chunk_abs, state0)` | abs → normalised | NumPy; zero rows preserved |
| `denormalize_to_abs_chunk(chunk_norm, state0)` | normalised → abs | NumPy |
| `normalize_ref_chunk(ref_chunk_abs, state0)` | abs → normalised | Alias for `normalize_chunk` |
| `prepare_training_batch(batch_np)` | in-place batch | Normalises `ref_chunk`, `action_chunk`, `next_ref_chunk`; uses `proprio` / `next_proprio` as state0 anchors |
| `normalize_chunk_torch(chunk_abs, state0)` | abs → normalised | Torch; compatible with batched `[B, chunk_len, action_dim]` |
| `denormalize_to_abs_chunk_torch(chunk_norm, state0)` | normalised → abs | Torch |

The `_delta_dim` property returns `action_dim - 1` — the number of joints that
receive delta encoding.  For SO-ARM with `action_dim=6` this is 5 (all joints
except gripper).

### Normalisation range: `[-1, 1]` not `[0, 1]`

The integration plan spec says to clip to `[0, 1]`.  The openpi-RLT reference
implementation uses `[-1, 1]` (formula: `(x - q01) / (q99 - q01) * 2 - 1`).
We follow the reference because:

- The RL actor outputs logits in `[-1, 1]` (tanh-squashed or fixed-std Gaussian
  centered at zero); matching the normalisation range avoids needing a bias shift
  in the actor loss.
- The reference's critic and BC penalty code assume `[-1, 1]` inputs.

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Framework | JAX / `jnp` | NumPy + optional PyTorch |
| Delta dims | All `action_dim` joints | Only first `action_dim - 1`; gripper absolute |
| JSON layout | `norm_stats.actions` nested | Both `norm_stats.actions` (openpi) and flat `actions` supported |
| Factory pattern | `from_config` class method | Same; returns `None` if path is unset |
| JAX standalone helpers | `jax_quantile_denormalize`, `jax_delta_to_abs_chunk` | Replaced by `_quantile_{normalize,denormalize}_torch` private helpers |
| Zero-row preservation | `np.where(zero_mask, 0.0, ...)` | Same; mask computed from `chunk_abs` before normalisation |
| Batch shape | `[..., chunk_len, action_dim]` | Same; `_broadcast_state0` adds dims until ndim matches |

---

## Step 6 — `rl/rltoken/replay.py`

**File:** [src/lerobot/rl/rltoken/replay.py](src/lerobot/rl/rltoken/replay.py)

### Purpose

Three concerns in one file:

1. **Episode → chunk conversion** — two public functions consume a flat list of
   `EpisodeStepRecord` objects (one per control step) and emit chunk-level
   `RLTTransition` objects suitable for the replay buffer.
2. **`RLTReplayBuffer`** — thread-safe CPU ring buffer.  Lazily allocates a dict
   of pre-shaped numpy arrays on the first `add()` call; supports both uniform
   and stratified sampling.
3. **`ReplayManager` / `ReplayClient` / `NullReplayClient`** — the B3 process
   that owns the buffer, an append-only pickle journal, and a lightweight HTTP
   service so the learner (B2) and robot PC (B4) can talk to it.

### Public API

#### Episode builders

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `build_chunk_transitions_from_episode` | `list[EpisodeStepRecord]`, `chunk_len`, `stride`, `allow_partial` | `list[RLTTransition]` | Overlapping sliding window; zero-pads the tail |
| `build_terminal_aligned_chunk_transition` | same, no stride | `RLTTransition \| None` | Extracts the last `chunk_len` steps; returns `None` if episode too short or not done |

`stride=2` (default) gives ~50% overlap between successive transitions.  The
terminal-aligned variant guarantees the final reward appears in `rewards[-1]`,
which matters for TD-target computation.

Both accept `EpisodeStepRecord` objects or plain dicts (for deserialization from
journal or test fixtures) via `_coerce_step()` which uses duck-typing so that
cross-module-path imports don't break `isinstance` checks.

#### `RLTReplayBuffer`

| Method | Notes |
|--------|-------|
| `add(transition)` | Writes into `storage[position]`, wraps around at capacity |
| `extend(transitions)` | Calls `add` in a loop |
| `sample(batch_size)` | Returns `ArrayDict` (copy); raises if empty |
| `stats()` | Returns capacity/size/position/strategy info |

Storage is lazily initialised on the first `add()`.  Each field is stored as a
`[capacity, *field_shape]` pre-typed numpy array so sampling is a single index
gather per field.

**Stratified sampling** draws from four pools:

| Pool | Filter | Target fraction |
|------|--------|-----------------|
| Recent online | `collection_phase_id == ONLINE` and `episode_id >= max_ep - window` | `recent_online_ratio` (default 0.4) |
| Warmup demos | `collection_phase_id == WARMUP` | `warmup_demo_ratio` (default 0.3) |
| Human intervention | `intervention_flag \| source in {HUMAN, MIXED}` | `human_intervention_ratio` (default 0.2) |
| Uniform | All indices | remainder |

Each pool is sampled with replacement (`np.random.default_rng.choice`).  If a
pool is empty the budget falls through to the uniform fallback.

#### `ReplayManager`

Constructor takes a `ReplayConfig` from `config.py`.  Restores the buffer from
journal on startup before opening the HTTP server.

HTTP routes (pickle over `application/octet-stream`):

| Verb | Path | Action |
|------|------|--------|
| GET | `/healthz` | Returns `b"OK"` |
| GET | `/stats` | Returns JSON stats dict |
| POST | `/add` | Deserialise one transition, add to buffer + journal |
| POST | `/extend` | Deserialise `{"transitions": [...]}`, bulk add |
| POST | `/sample` | Deserialise `{"batch_size": N}`, return sampled `ArrayDict` |

The server runs in `ThreadingHTTPServer` (one thread per request) with a
`stop_event` argument for clean shutdown in tests.

#### `ReplayClient`

Standard HTTP client using `urllib.request` — no extra dependencies.

| Method | Notes |
|--------|-------|
| `add_transition(t)` | Serialises via `to_journal_record()` |
| `add_transitions(ts)` | Batches into one `/extend` call |
| `sample_batch(n)` | Returns `ArrayDict` |
| `healthz()` | Returns bool; never raises |
| `wait_until_healthy(timeout_sec)` | Polls `/healthz` until ready or timeout |

#### `NullReplayClient`

Drop-in for eval runs.  Discards all writes silently; `healthz()` returns True;
`sample_batch()` raises to make accidental reads loud.

### Journal format

The journal is an append-only file where each record is written with
`pickle.dump(transition.to_journal_record(), ...)`.  `to_journal_record()`
returns the compact `to_numpy()` dict plus the human-readable
`collection_phase` string (so the journal can be filtered by phase after a
crash without decoding the `collection_phase_id` integer).

Writes are flushed and fsync'd after each batch so partial writes are
distinguishable from complete records at restore time (an incomplete pickle
frame raises `UnpicklingError` or `EOFError`, both caught during restore).

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| HTTP transport | `msgpack_numpy` (openpi_client) | `pickle` — no extra dependency; equally efficient for numpy |
| `ReplayBuffer` constructor | Individual keyword args | Accepts `ReplayConfig` from `config.py` |
| `EpisodeStepRecord` / `RLTTransition` | Defined inside `replay.py` | Imported from `transition.py` (already extracted in Step 1) |
| `RawEpisodeTrace` / `RawEpisodeChunk` | Defined in `replay.py` | Not implemented — `rollout.py` produces a flat `list[EpisodeStepRecord]` directly |
| Duck-type coercion | `isinstance(step, EpisodeStepRecord)` | `_coerce_step()` with `hasattr` fallback to handle cross-path `isinstance` failures |
| Metrics logging | `runtime_logging.append_jsonl` | Python stdlib `logging.info` (no extra file) |
| `NullReplayClient` | Present | Same — used by eval rollout |

---

## Step 5 — `rl/rltoken/networks.py`

**File:** [src/lerobot/rl/rltoken/networks.py](src/lerobot/rl/rltoken/networks.py)

### Purpose

PyTorch port of the JAX actor-critic networks from `openpi-RLT/rlt_online_rl/networks.py`.
Three `nn.Module` classes, two loss functions, and one TD-target helper — everything
needed by `trainer.py` to run the B2 learner loop.

### Architecture

#### Feature encoding (shared by actor and critic)

```
z_rl     [B, z_dim]            →  Linear(z_dim, 256)  →  LN(no affine)            →  z_feat    [B, 256]
proprio  [B, proprio_dim]      →  Linear(P, 64)       →  tanh(LN(no affine))      →  p_feat    [B, 64]
ref/act  [B, chunk_len*A]      →  Linear(C*A, 256)    →  tanh(LN(no affine))      →  r_feat    [B, 256]
                                                                    concat → [B, 576]
                                                                        ↓
                                                              Trunk MLP (hidden_dim×num_layers)
                                                              each hidden: Linear → LN → GELU
                                                              output: Linear (no activation)
```

The z_feat has LayerNorm but **no tanh** — matching the reference where z_rl is
normalized but not squashed.  proprio and ref/action features are tanh-squashed after
LayerNorm.

#### `ChunkActor`

Outputs action mean `[B, chunk_len, action_dim]`.

| Method | Notes |
|--------|-------|
| `forward(z_rl, proprio, ref_chunk)` | Returns mean chunk |
| `sample(z_rl, proprio, ref_chunk, deterministic)` | Returns `(chunk, log_prob_sum)` |
| `apply_reference_dropout(ref_chunk, prob)` | Zero whole-chunk per-element; only active in `training=True` mode |

`fixed_std` is registered as a `buffer` (not a trainable parameter) so it
moves with `.to(device/dtype)`.  Log-prob is summed over all `chunk_len ×
action_dim` dimensions.

#### `QNetwork`

Same projection structure as `ChunkActor` but uses `action_chunk` (not
`ref_chunk`) and outputs a scalar Q-value per batch element `[B]`.

#### `TwinCritic`

Owns `q1` and `q2` (`QNetwork` instances).  Exposes `q_values(...)→(q1,q2)`
and `min_q(...)`.

### Loss functions

#### `build_td_target` (`@torch.no_grad()`)

```
td_target = Σ_t r_t * γ^t  +  (1 - done) * γ^T * min(q1', q2')
```

where `T = chunk_len`.  The target actor is called **stochastically** (matching
the reference; `deterministic=False`).  Since `fixed_std=0.002` the noise is
negligible in practice.

#### `compute_critic_loss`

```
loss = MSE(q1, td_target) + MSE(q2, td_target)
```

#### `compute_actor_loss`

```
loss = bc_weight * BC_penalty  +  q_weight * (-mean(q1))
```

**Source-aware BC target** (lerobot-RLT addition):

| `source_chunk` value | BC target |
|----------------------|-----------|
| `HUMAN` or `MIXED` | `action_chunk` (human demonstration) |
| `BASE` or `RL` | `ref_chunk` (VLA reference) |

The mask is computed per-step: `human_mask [B, chunk_len, 1]` blends the two targets.

Reference dropout (`apply_reference_dropout`) is applied to `ref_chunk` before the
actor forward pass.  The critic Q-evaluation always uses the clean `ref_chunk`.

#### `apply_reference_dropout` (standalone)

Same as the member method but without the `self.training` gate — for use from
`inference.py` where the actor is in eval mode but dropout is still needed for
exploration.

### Design choices vs openpi-RLT

| Aspect | openpi-RLT | lerobot-RLT |
|--------|-----------|-------------|
| Framework | JAX functional (stateless params dict) | PyTorch `nn.Module` |
| LayerNorm | Manual `_layer_norm` (no affine) | `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` |
| Fixed std | Float field on frozen dataclass | `register_buffer('fixed_std', ...)` |
| Actor loss | `-mean(q1) + beta * MSE(action, ref)` | Same + **source-aware BC target** (HUMAN steps use action_chunk) |
| Delta smoothness penalty | Not in `networks.py`; in `trainer.py` | Same — deferred to `trainer.py` where adapter can convert to absolute space |
| `compute_actor_loss` signature | Returns `(loss, metrics)` | Same |
| TD target actor | Stochastic (`deterministic=False`) | Same (plan incorrectly said `deterministic=True`; reference wins) |
| Standalone `apply_reference_dropout` | Module-level function | Added — needed for eval-mode inference in `inference.py` |

---

## Step 7 — `rl/rltoken/trainer.py`

**File:** [src/lerobot/rl/rltoken/trainer.py](src/lerobot/rl/rltoken/trainer.py)

### Purpose

The B2 (Learner) process: samples batches from the replay buffer, updates the
actor-critic, exports actor snapshots for B1 to hot-swap, and checkpoints the
full training state periodically.  The PyTorch port of the JAX/Flax
`LearnerService` in openpi-RLT.

### Public API

- `RLTTrainState` — mutable dataclass: `actor`, `target_actor`, `critic`, `target_critic`, `actor_optimizer`, `critic_optimizer`, `global_step`, `actor_version`.
- `init_train_state(config)` — builds both network pairs (target = `deepcopy`) with Adam optimizers; target networks are `eval()` and have `requires_grad=False`.
- `polyak_update(source, target, tau)` — in-place EMA: `p_t = (1-τ)p_t + τ·p_s`.
- `LearnerService` — full training shell with budget pacing, checkpoint save/restore, and snapshot export.

### Key design choices

#### Budget-based pacing (warmup latch)

Mirrors the JAX reference exactly.  When `replay_size` first reaches
`warmup_min_size`, `_warmup_ready_adds_total` is latched.  From that point:

```
warmup_required = warmup_post_collect_updates   (or adds_at_latch * G)
desired_total   = warmup_required + max(adds_total - adds_at_latch, 0) * G
pending_budget  = max(desired_total - global_step, 0)
```

`G = grad_updates_per_cycle`.  If `pending_budget <= 0` the learner skips a
step and sleeps `poll_interval_sec`.  This keeps the update-to-data ratio
constant as the robot collects new transitions.

#### Critic-first, actor every `actor_update_period` steps

Critic is updated every step.  Actor is updated only on steps where
`(global_step + 1) % actor_update_period == 0` (matches reference).
After each actor update, both target networks receive a Polyak update and
`actor_version` is incremented.

#### Source-aware BC target (inline in `LearnerService._train_step_actor`)

Rather than calling `networks.compute_actor_loss()` (which does the same
thing but can't reach the adapter for the delta penalty), the actor loss is
computed inline:

```
human_mask   = (source_chunk == HUMAN) | (source_chunk == MIXED)  → [B, T, 1]
bc_target    = human_mask * action_chunk + (1 - human_mask) * ref_chunk
bc_penalty   = MSE(sampled_chunk, bc_target)
q_loss       = -mean(q1)
delta_penalty = MSE(Δ_pred_abs[:, 1:], Δ_bc_abs[:, 1:])  (first delta_dim joints)
actor_loss   = bc_w * bc + q_w * q + δ_w * delta
```

#### Delta penalty in absolute space

Because the actor operates in normalized delta space, jerk smoothness must be
measured after denormalization.  `adapter.denormalize_to_abs_chunk_torch()` is
called for both `sampled_chunk` and `bc_target`; consecutive-step differences
along the first `action_dim - 1` joints are compared with MSE.  The penalty is
skipped when `delta_weight == 0` or `adapter is None`.

#### Atomic writes

`learner_status.json` — polled by `rollout.py` — is written with tmp+rename
via `_atomic_write_json()`.  Checkpoints and actor snapshots use
`_atomic_write_pickle()` for the same reason.

### Deviations from JAX reference

| Topic | Reference (JAX) | lerobot-RLT (PyTorch) |
|-------|-----------------|----------------------|
| State representation | Flax `TrainState` struct | Plain Python dataclass; optimizers are first-class fields |
| JIT compilation | `jax.jit(train_step)` | PyTorch eager; no explicit JIT needed for single-GPU |
| Functional update | Returns new params dict | In-place `optimizer.step()` on `nn.Module` parameters |
| Actor loss function | `update_actor()` standalone fn | Inline in `LearnerService.train_once()` to share adapter |
| Delta penalty | Separate `bc_ref_penalty` / `bc_human_penalty` metrics | Combined `delta_penalty` metric |
| Checkpoint format | Orbax serialization | `pickle` with `state_dict()` for each module |
| Status file | Same `learner_status.json` layout | Identical; field names preserved for `rollout.py` compatibility |

---

## Step 8 — `rl/rltoken/inference.py`

**File:** [src/lerobot/rl/rltoken/inference.py](src/lerobot/rl/rltoken/inference.py)

### Purpose

The B1 (Actor Service) process: serves `ChunkActor` inference over HTTP
(pickle transport), polls the snapshot file written by B2 for hot-param
reload, and provides `MachineAFeatureClient` for the B4 rollout process to
fetch `z_rl` / `ref_chunk` features over WebSocket.

### Public API

- `ActorRequest` / `ActorResponse` / `RefinementResult` — `slots=True` dataclasses for typed transport.
- `ActorService` — HTTP inference server + background snapshot poller.  Thread-safe via a single `threading.Lock` protecting all actor weight access.
- `ActorClient` — HTTP client; pickle transport; `infer()`, `get_version()`, `healthz()`, `wait_until_healthy()`.
- `maybe_refine_chunk()` — convenience wrapper returning `RefinementResult`; falls back to `BASE` (returns `ref_chunk` unchanged) on error when `on_error_fallback=True`.
- `MachineAFeatureClient` — WebSocket client using `websockets.sync.client`; pickle payloads; `get_features()` / `get_features_batch()`; auto-reconnect on failure.

### Key design choices

#### Snapshot hot-reload without downtime

A background daemon thread (`_poll_snapshot_loop`) calls `_try_reload_snapshot()`
every `pull_params_interval_sec`.  The double-checked locking pattern is used:
version check outside the lock (cheap), full `load_state_dict` + warmup inside
the lock (safe).  This means `infer()` is never blocked by a stale-version
check, only by the brief window of a genuine weight swap.

#### Fallback to `ref_chunk` before first snapshot

Until the first snapshot arrives, `_params_loaded = False` and `infer()`
returns the `ref_chunk` directly with `source=BASE`.  This lets B4 start
collecting warmup data before B2 has produced a trained actor.

#### Normalize / denormalize split across the lock

`adapter.normalize_ref_chunk()` (adapter is frozen/immutable) runs *before*
acquiring the lock; `adapter.denormalize_to_abs_chunk()` runs *after* releasing
it.  The PyTorch forward pass runs *inside* the lock under `torch.no_grad()`.
This minimises contention while keeping weight access serialized.

#### MachineAFeatureClient reconnect strategy

On send/recv failure the client calls `_reconnect()` which:
1. Closes the current socket.
2. Polls the HTTP `/healthz` endpoint until it returns 200.
3. Opens a fresh WebSocket and waits for the metadata handshake packet.

`_infer()` retries once after reconnect before raising, so transient
disconnects are transparent to the caller.

#### Batch feature requests

`get_features_batch()` sends `{"batch": [obs, ...]}` and expects
`{"batch_results": [...]}`.  If the server returns an unexpected shape or the
request raises, it falls back silently to sequential `get_features()` calls.
This keeps the rollout loop running even against a server that only supports
single requests.

### Deviations from JAX reference

| Topic | Reference (JAX) | lerobot-RLT (PyTorch) |
|-------|-----------------|----------------------|
| Transport | `msgpack_numpy` (HTTP) | `pickle` — msgpack_numpy not installed |
| WebSocket payload | `msgpack_numpy` | `pickle` |
| Actor forward | Stateless JAX `apply(params, ...)` | `nn.Module` with `load_state_dict()` |
| `EnvDriver` | Complex class with `RawEpisodeTrace`, prefetch queue | Not in `inference.py`; lives in `rollout.py` |
| `ActorService` HTTP server | Flask-style routing | `ThreadingHTTPServer` from stdlib |
| Warmup on load | Full model warmup with JAX tracing | One dummy `actor.sample()` call under `torch.no_grad()` |

---

## Step 10 — `rl/rltoken/rollout.py`

**File:** [src/lerobot/rl/rltoken/rollout.py](src/lerobot/rl/rltoken/rollout.py)

### Purpose

SO-ARM-specific rollout logic.  Wraps `SOFollower`/`ThreadSafeRobot` into a chunk-level executor, drives the Machine A → Actor → robot execution loop, and pushes `RLTTransition` objects to the replay buffer at episode end.  No JAX dependency.

### Public API

- `StepResult` — per-step result from `SOArmEnv.execute_chunk()`.
- `SOArmEnv` — rate-controlled hardware layer; `obs_to_state()`, `reset()`, `execute_chunk()`.
- `ChunkFeatures` — holds `(z_rl, proprio, ref_chunk)` returned by Machine A for one chunk.
- `RolloutPhaseController` — polls `learner_status.json` to implement warmup → online transition.
- `KeyboardSignalBridge` — non-blocking pynput keyboard queue; falls back to no-op if pynput absent.
- `EnvDriver` — single-episode orchestrator with two-pass `EpisodeStepRecord` construction.
- `RolloutOrchestrator` — top-level episode loop; wires all components together.
- Signal constants: `SIG_NEXT_EPISODE`, `SIG_MARK_SUCCESS`, `SIG_MARK_FAILURE`, `SIG_SET_ACTOR_MODE`, `SIG_SET_BASE_MODE`, `SIG_TOGGLE_TELEOP`, `SIG_QUIT`.

### Key design choices

#### Two-pass `EpisodeStepRecord` construction

`next_z_rl` for step k cannot be known during chunk k execution because Machine A
is only queried at chunk boundaries.  The episode loop therefore:
1. Collects `(step_results, features, source, actor_version)` per chunk in a list.
2. After the loop, queries Machine A once more on the terminal observation to get `z_rl_terminal`.
3. In a post-processing pass, assigns `next_z_rl = features_{k+1}.z_rl` for all steps in chunk k (or `z_rl_terminal` for the last chunk).

This is a clean approximation: `z_rl` changes slowly, and using the chunk-start
value for all steps within that chunk introduces negligible bias.

#### Rate control in `execute_chunk`

Uses `time.monotonic()` + `time.sleep(remaining)` per step rather than an
absolute wall-clock schedule.  This avoids cumulative drift when `send_action`
takes variable time, but doesn't compensate if a single step overshoots the
budget.

#### Phase controller and learner_status.json

`RolloutPhaseController` reads `{run_dir}/metrics/learner_status.json` (written
atomically by `LearnerService._write_status()`).  It caches the result for
`poll_interval_sec` to avoid hammering the filesystem, and falls back to
`"warmup"` on any read error.  The `ready_for_online` boolean field is the sole
gate for transitioning to the `"online"` phase.

#### Keyboard signal handling

The `done_signal` callback passed to `run_episode()` is a closure that reads
from `KeyboardSignalBridge.get_signal()` once per chunk.  Only stop signals
(`SIG_NEXT_EPISODE`, `SIG_MARK_SUCCESS`, `SIG_MARK_FAILURE`, `SIG_QUIT`)
terminate the episode.  Mode-override signals (`SIG_SET_ACTOR_MODE`,
`SIG_SET_BASE_MODE`) are consumed by `RolloutOrchestrator.run_forever()` before
and after each episode.

#### `EnvDriver` is stateless across episodes

`EnvDriver` has no per-episode mutable state; everything is local to
`run_episode()`.  This makes it safe to call repeatedly without restarting the
process.

### Deviations from reference (openpi-RLT `inference.py` + `rollout`)

| Topic | Reference (JAX) | lerobot-RLT (PyTorch) |
|-------|-----------------|----------------------|
| Episode trace structure | `RawEpisodeTrace` with separate obs, steps, chunks lists | Flat `list[EpisodeStepRecord]` — simpler, no backfill queue |
| Feature backfill | Batch Machine A query at episode end for all intermediate obs | Not needed: z_rl assigned per-chunk, not per-step |
| Transition stride | `stride=2` sliding window | `stride=chunk_len` (non-overlapping) — one transition per chunk executed |
| Env interface | Protocol-based `env.reset()`, `execute_chunk()`, `current_phase_name()` | Concrete `SOArmEnv` with `ThreadSafeRobot`; phase owned by `RolloutPhaseController` |
| Human override | `human_override_fn` callback injected at `EnvDriver` level | Not implemented; operator signals `SIG_MARK_SUCCESS/FAILURE` to end episode |
| raw episode pickle | Persists `RawEpisodeTrace` to disk for post-hoc analysis | Not implemented in this version |
| Keyboard | pynput via `EnvDriver` | Owned by `RolloutOrchestrator`; `EnvDriver` receives a `done_signal` callable |

---

## Step 9 — `scripts/serve_rlt_policy.py`

**File:** [src/lerobot/scripts/serve_rlt_policy.py](src/lerobot/scripts/serve_rlt_policy.py)

### Purpose

Machine A WebSocket server.  Loads a trained `RLTokenPolicy` checkpoint and
serves VLA features (`z_rl`, `ref_chunk`, `proprio`) to the robot rollout
process (`MachineAFeatureClient` in `inference.py`) over WebSocket.

### Protocol

All messages use **pickle** (consistent with the rest of the pipeline; no
msgpack_numpy dependency).

On connect the server sends a metadata handshake packet:
```python
{"z_dim": int, "chunk_len": int, "action_dim": int, "proprio_dim": int,
 "device": str, "supports_batch": True}
```

Single-observation request/response:
```
→ {"images": {cam_key: HxWxC uint8 numpy}, "state": [state_dim,], "prompt": str}
← {"z_rl": [z_dim,], "ref_chunk": [chunk_len, action_dim], "proprio": [proprio_dim,], "infer_ms": float}
```

Batch request/response:
```
→ {"batch": [obs1, obs2, ...]}
← {"batch_results": [res1, res2, ...], "batch_size": N}
```

### Key design choices

**Two-call forward pass** — `_forward_once()` calls `policy.encode()` (vision
encoder → z_rl) and `policy.predict_action_chunk()` (full VLA flow matching →
ref_chunk) separately.  Each call internally runs `extract_prefix_embeddings()`,
so the prefix extraction happens twice per observation.  This is intentional:
it avoids coupling the two model entry points and keeps the serve script simple.
A shared-prefix optimisation can be added later if latency becomes a bottleneck.

**Image preprocessing** — images arriving as `HxWxC uint8` are converted to
`[1, C, H, W] float32 [0, 1]` and stored under `observation.images.{cam_key}`.
The policy's internal preprocessing (PI05/PaliGemma normalisation) is applied
inside `policy.encode()` / `policy.predict_action_chunk()`.

**HTTP healthz sidecar** — a `ThreadingHTTPServer` running in a daemon thread
answers `GET /healthz → 200 OK` on the same port as the WebSocket.  This lets
`MachineAFeatureClient._wait_for_server()` poll HTTP before opening the
WebSocket.

**Warmup** — one dummy forward pass runs at startup to trigger CUDA graph
compilation / any JIT overhead before the first real request arrives.

**CLI** — `--checkpoint-path`, `--port`, `--host`, `--chunk-len`, `--action-dim`,
`--proprio-dim`, `--z-dim`, `--device`.  The default port is 8000.

### Deviations from reference (openpi-RLT `serve_rlt_policy.py`)

| Topic | Reference (JAX) | lerobot-RLT (PyTorch) |
|-------|-----------------|----------------------|
| Backend | JAX JIT + TPU/GPU | PyTorch + CUDA |
| Policy class | `RLTPolicy` (openpi wrapper) | `RLTokenPolicy` |
| Transport | msgpack via openpi `WebsocketPolicyServer` | pickle over raw websockets |
| Warmup | JIT compile for each batch size in a set | Single dummy forward pass |
| Shared prefix | extract_prefix_embeddings cached per request | Not cached (two-call design) |
| Health check | separate FastAPI `/healthz` process | ThreadingHTTPServer sidecar in same process |

---

## Step 11 — `scripts/launch_*.py` and `scripts/run_rlt_role.py`

**Files:**
- [src/lerobot/scripts/run_rlt_role.py](src/lerobot/scripts/run_rlt_role.py)
- [src/lerobot/scripts/launch_machine_b_rlt.py](src/lerobot/scripts/launch_machine_b_rlt.py)
- [src/lerobot/scripts/launch_robot_rollout_rlt.py](src/lerobot/scripts/launch_robot_rollout_rlt.py)
- [src/lerobot/scripts/launch_actor_eval_rlt.py](src/lerobot/scripts/launch_actor_eval_rlt.py)

### `run_rlt_role.py` — role dispatcher

A thin subprocess entry point that accepts `--role {replay_manager,
learner_service, actor_service}` and `--config <yaml>` and starts the
corresponding service.  Used by `launch_machine_b_rlt.py` (one subprocess per
role).  Derives `run_dir` from the config's `checkpoint_dir` parent when not
explicitly passed.

### `launch_machine_b_rlt.py`

Spawns three subprocesses via `run_rlt_role.py`, registers SIGINT/SIGTERM
handlers, then polls every 1 second.  If any subprocess exits non-zero the
launcher terminates all siblings and exits 1.  A 5-second grace period after
spawn avoids false-positive crash detection during service startup.

Optional `--wandb-project` / `--wandb-run-name` arguments look for a
`_rlt_wandb_monitor.py` sidecar script (not yet implemented; a warning is
logged if it is absent).

### `launch_robot_rollout_rlt.py`

Before starting `RolloutOrchestrator`, waits for:
1. `ActorService GET /healthz → 200`
2. `ReplayManager GET /healthz → 200`
3. `Machine A GET {http_url}/healthz → 200` (derived from the `ws://` URL)

Timeout defaults to 300 seconds per service.  Pass `--no-wait` to skip all
checks.  CLI overrides for `--actor-url`, `--replay-url`, `--machine-a-ws-url`
allow running the robot machine against services on different hosts without
editing the YAML.

### `launch_actor_eval_rlt.py`

Eval-only mode: starts `ActorService` in a daemon thread (no learner writes
new snapshots), waits for it to become healthy, then runs
`RolloutOrchestrator(eval_only=True)`.  Useful for demonstrating a trained
checkpoint without spinning up Machine B.

Passes `actor_deterministic=True` by default so the actor uses its mean
action (no exploration noise).

### Deviations from reference

| Topic | Reference (openpi-RLT) | lerobot-RLT |
|-------|------------------------|-------------|
| B service spawning | `subprocess.Popen` + multiprocessing monitor | Same pattern |
| Health wait | `/version` + `/stats` endpoints | `/healthz` on all services |
| Robot wait | `pika_sync_ros.py` ROS bridge | `RolloutOrchestrator` directly |
| W&B monitor | Inline metric streaming | Separate sidecar (stub only) |

---

## Step 12 — YAML configs and `norm_stats_delta.json`

**Files:**
- [configs/rltoken/base.yaml](configs/rltoken/base.yaml)
- [configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml](configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml)
- [configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json](configs/rltoken/tasks/so_arm_ethernet/stats/norm_stats_delta.json)

### YAML layout

Both files use the `experiment` / `runtime` two-section layout supported by
`load_system_config_yaml()`.  `base.yaml` lists every field with its default
value as documentation.  `online_rl.yaml` only overrides what is task-specific
(paths, IP addresses, SO-ARM dimension values).

### `norm_stats_delta.json`

**Currently a placeholder.**  The file exists so that the config can be parsed
without errors, but the `q01`/`q99` values are dummy.  **These must be replaced
with real quantile statistics computed from the demonstration dataset before
training.**

The intended workflow:
1. Record SO-ARM demonstrations using the standard lerobot record pipeline.
2. Run `scripts/compute_delta_norm_stats.py` (to be written) on the resulting
   `LeRobotDataset` to compute per-dimension 1st/99th percentile of the
   delta-chunk actions.
3. Write the output JSON to this path.

The file format matches what `DeltaChunkNormAdapter.from_config()` expects:
```json
{"actions": {"q01": [...], "q99": [...]}}
```

---

