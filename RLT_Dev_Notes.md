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
| 1 | `rl/rltoken/action_representation.py` | Pending |
| 2 | `policies/rltoken/rl_token.py` | Done |
| 2 | `policies/rltoken/configuration_rltoken.py` | Done |
| 3 | `policies/pi05_rlt/modeling_pi05_rlt.py` | Done |
| 4 | `policies/rltoken/modeling_rltoken.py` | Done |
| 4 | `scripts/train_rlt.py` | Done |
| 5 | `rl/rltoken/networks.py` | Pending |
| 6 | `rl/rltoken/replay.py` | Pending |
| 7 | `rl/rltoken/trainer.py` | Pending |
| 8 | `rl/rltoken/inference.py` | Pending |
| 9 | `scripts/serve_rlt_policy.py` | Pending |
| 10 | `rl/rltoken/rollout.py` | Pending |
| 11 | `scripts/launch_*.py` / YAML configs | Pending |
| 12 | `configs/.../norm_stats_delta.json` | Pending |

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
