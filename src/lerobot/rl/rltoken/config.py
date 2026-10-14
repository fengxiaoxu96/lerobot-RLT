#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration dataclasses for the RLToken online RL system.

Six frozen dataclasses cover the full system:
    RLTOnlineRLConfig      — algorithm hyperparameters
    ActorServiceConfig     — B1: actor HTTP service
    LearnerServiceConfig   — B2: learner training loop
    ReplayConfig           — B3: replay buffer + HTTP service
    EnvDriverConfig        — B4: robot rollout / env driver
    MonitoringConfig       — optional W&B sidecar
    OnlineRLSystemConfig   — top-level container

All path fields accept relative paths.  load_system_config_yaml() resolves
them first against the config file's directory, then against the repo root.
save_system_config_yaml() writes them back as relative paths.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

# Repo root: two levels up from this file  (src/lerobot/rl/rltoken/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CONFIG_FILENAME = "online_rl_config.yaml"


# ---------------------------------------------------------------------------
# Per-service dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RLTOnlineRLConfig:
    """Algorithm-only configuration for chunk-level online RL."""

    # Action / observation dimensions  (SO-ARM defaults)
    action_dim: int = 6
    chunk_len: int = 10
    z_dim: int = 2048
    proprio_dim: int = 6

    # Action representation
    action_representation: Literal["abs", "delta_chunk"] = "delta_chunk"
    action_norm_stats_path: str | None = None

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

    # Actor / critic architecture
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    # Optimisation
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    target_tau: float = 5e-3
    actor_update_period: int = 2

    # Warmup / online schedule
    warmup_min_size: int = 600
    warmup_post_collect_updates: int | None = 20_000
    freeze_after_warmup: bool = False
    grad_updates_per_cycle: int = 5


@dataclasses.dataclass(frozen=True)
class ActorServiceConfig:
    """Configuration for B1 actor_service (ChunkActor HTTP inference)."""

    bind_host: str = "127.0.0.1"
    port: int = 9101
    snapshot_path: str = "./runs/so_arm_rlt/actor_snapshot/actor_snapshot.pkl"
    pull_params_interval_sec: float = 0.25
    rpc_timeout_sec: float = 0.2


@dataclasses.dataclass(frozen=True)
class LearnerServiceConfig:
    """Configuration for B2 learner_service (actor-critic training loop)."""

    push_actor_interval_steps: int = 500
    checkpoint_interval_steps: int = 1_000
    sample_batch_size: int = 128
    checkpoint_dir: str = "./runs/so_arm_rlt/checkpoints"
    actor_snapshot_path: str = "./runs/so_arm_rlt/actor_snapshot/actor_snapshot.pkl"
    replay_url: str = "http://127.0.0.1:9102"
    poll_interval_sec: float = 0.01


@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    """Configuration for B3 replay_manager (ring buffer + HTTP service)."""

    capacity: int = 200_000
    journal_path: str = "./runs/so_arm_rlt/replay/replay_journal.pkl"
    bind_host: str = "127.0.0.1"
    port: int = 9102
    seed: int = 0
    sample_strategy: Literal["uniform", "stratified"] = "stratified"
    # Stratified sampling pool parameters
    recent_episode_window: int = 20
    recent_online_ratio: float = 0.4
    warmup_demo_ratio: float = 0.3
    human_intervention_ratio: float = 0.2


@dataclasses.dataclass(frozen=True)
class EnvDriverConfig:
    """Configuration for B4 env_driver / robot rollout."""

    machine_a_ws_url: str = "ws://127.0.0.1:8000"
    actor_service_url: str = "http://127.0.0.1:9101"
    replay_service_url: str = "http://127.0.0.1:9102"

    task_mode: Literal["full_task", "critical_phase"] = "critical_phase"
    episode_start_control_mode: Literal["sticky", "policy", "human"] = "policy"

    # Reset poses: 6-element absolute joint arrays for SO-ARM
    full_task_reset_action: list[float] | None = None
    critical_phase_reset_action: list[float] | None = None

    actor_deterministic: bool = False
    chunk_exec_horizon: int = 10
    control_frequency_hz: float = 20.0
    step_trace_stride: int = 0
    replay_feature_batch_size: int = 16
    enable_human_override: bool = False
    safe_fallback_to_ref: bool = True

    # Machine A connection settings
    machine_a_connect_timeout_sec: float = 10.0
    machine_a_recv_timeout_sec: float = 5.0
    machine_a_retry_interval_sec: float = 0.5
    actor_request_timeout_sec: float = 1.0
    replay_request_timeout_sec: float = 30.0


@dataclasses.dataclass(frozen=True)
class MonitoringConfig:
    """Optional W&B monitoring sidecar configuration."""

    enable_wandb: bool = False
    wandb_project: str = "rlt-online-rl"
    wandb_entity: str | None = None
    wandb_mode: Literal["offline", "online"] = "offline"
    wandb_run_name: str = "so-arm-online-rl"
    wandb_dir: str = "./runs/so_arm_rlt/wandb"


@dataclasses.dataclass(frozen=True)
class OnlineRLSystemConfig:
    """Top-level configuration container for the full online RL system."""

    rl: RLTOnlineRLConfig = dataclasses.field(default_factory=RLTOnlineRLConfig)
    actor_service: ActorServiceConfig = dataclasses.field(default_factory=ActorServiceConfig)
    learner_service: LearnerServiceConfig = dataclasses.field(default_factory=LearnerServiceConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    env_driver: EnvDriverConfig = dataclasses.field(default_factory=EnvDriverConfig)
    monitoring: MonitoringConfig = dataclasses.field(default_factory=MonitoringConfig)

    role: Literal[
        "all",
        "actor_service",
        "learner_service",
        "replay_manager",
        "env_driver",
    ] = "all"
    local_debug_mode: bool = False


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_path(value: str | None, anchor: str, *, require_exists: bool) -> str | None:
    """Resolve a relative path stored in a config field.

    Resolution order:
    1. Absolute paths are returned unchanged.
    2. Relative paths starting with './' or '../' are resolved against the
       directory containing the config file (anchor).
    3. Other relative paths are tried against anchor first, then repo root.
    """
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    anchor_dir = Path(anchor).resolve().parent
    from_anchor = (anchor_dir / candidate).resolve()
    from_root = (_REPO_ROOT / candidate).resolve()
    if from_anchor.exists():
        return str(from_anchor)
    if from_root.exists():
        return str(from_root)
    # Path does not exist yet (e.g. output dirs) — fall back to anchor-relative
    if candidate.parts and candidate.parts[0] in {".", ".."}:
        return str(from_anchor)
    return str(from_root) if not require_exists else value


def _relativize_path(value: str | None, anchor: str) -> str | None:
    """Convert an absolute path back to a relative path for YAML serialisation."""
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return value
    try:
        candidate.relative_to(_REPO_ROOT)
        return os.path.relpath(str(candidate), start=str(_REPO_ROOT))
    except ValueError:
        return os.path.relpath(str(candidate), start=os.path.dirname(os.path.abspath(anchor)))


def resolve_config_paths(system: OnlineRLSystemConfig, anchor: str) -> OnlineRLSystemConfig:
    """Resolve all path fields in a loaded config against *anchor* (the YAML file path)."""
    return dataclasses.replace(
        system,
        rl=dataclasses.replace(
            system.rl,
            action_norm_stats_path=_resolve_path(
                system.rl.action_norm_stats_path, anchor, require_exists=True
            ),
        ),
        actor_service=dataclasses.replace(
            system.actor_service,
            snapshot_path=_resolve_path(system.actor_service.snapshot_path, anchor, require_exists=False),
        ),
        learner_service=dataclasses.replace(
            system.learner_service,
            checkpoint_dir=_resolve_path(system.learner_service.checkpoint_dir, anchor, require_exists=False),
            actor_snapshot_path=_resolve_path(
                system.learner_service.actor_snapshot_path, anchor, require_exists=False
            ),
        ),
        replay=dataclasses.replace(
            system.replay,
            journal_path=_resolve_path(system.replay.journal_path, anchor, require_exists=False),
        ),
        monitoring=dataclasses.replace(
            system.monitoring,
            wandb_dir=_resolve_path(system.monitoring.wandb_dir, anchor, require_exists=False),
        ),
    )


def relativize_config_paths(system: OnlineRLSystemConfig, anchor: str) -> OnlineRLSystemConfig:
    """Convert absolute paths back to relative paths before saving a YAML."""
    return dataclasses.replace(
        system,
        rl=dataclasses.replace(
            system.rl,
            action_norm_stats_path=_relativize_path(system.rl.action_norm_stats_path, anchor),
        ),
        actor_service=dataclasses.replace(
            system.actor_service,
            snapshot_path=_relativize_path(system.actor_service.snapshot_path, anchor),
        ),
        learner_service=dataclasses.replace(
            system.learner_service,
            checkpoint_dir=_relativize_path(system.learner_service.checkpoint_dir, anchor),
            actor_snapshot_path=_relativize_path(system.learner_service.actor_snapshot_path, anchor),
        ),
        replay=dataclasses.replace(
            system.replay,
            journal_path=_relativize_path(system.replay.journal_path, anchor),
        ),
        monitoring=dataclasses.replace(
            system.monitoring,
            wandb_dir=_relativize_path(system.monitoring.wandb_dir, anchor),
        ),
    )


# ---------------------------------------------------------------------------
# YAML serialisation helpers
# ---------------------------------------------------------------------------


def _split_for_yaml(system: OnlineRLSystemConfig) -> dict[str, Any]:
    """Organise config into experiment / runtime sections for the YAML file."""
    return {
        "experiment": {
            "rl": dataclasses.asdict(system.rl),
        },
        "runtime": {
            "actor_service": dataclasses.asdict(system.actor_service),
            "learner_service": dataclasses.asdict(system.learner_service),
            "replay": dataclasses.asdict(system.replay),
            "env_driver": dataclasses.asdict(system.env_driver),
            "monitoring": dataclasses.asdict(system.monitoring),
            "role": system.role,
            "local_debug_mode": system.local_debug_mode,
        },
    }


def _flatten_yaml_sections(grouped: Mapping[str, Any]) -> dict[str, Any]:
    """Accept both grouped (experiment/runtime) and flat YAML layouts."""
    default = OnlineRLSystemConfig()
    top = set(grouped)
    runtime_keys = {
        "actor_service", "learner_service", "replay", "env_driver",
        "monitoring", "role", "local_debug_mode",
    }
    if top & {"experiment", "runtime"}:
        experiment = grouped.get("experiment") or {}
        runtime = grouped.get("runtime") or {}
        return {
            "rl": dict(experiment.get("rl", {})),
            "actor_service": dict(runtime.get("actor_service", {})),
            "learner_service": dict(runtime.get("learner_service", {})),
            "replay": dict(runtime.get("replay", {})),
            "env_driver": dict(runtime.get("env_driver", {})),
            "monitoring": dict(runtime.get("monitoring", {})),
            "role": runtime.get("role", default.role),
            "local_debug_mode": runtime.get("local_debug_mode", default.local_debug_mode),
        }
    # Flat layout (backward compat)
    return {
        "rl": dict(grouped.get("rl", {})),
        "actor_service": dict(grouped.get("actor_service", {})),
        "learner_service": dict(grouped.get("learner_service", {})),
        "replay": dict(grouped.get("replay", {})),
        "env_driver": dict(grouped.get("env_driver", {})),
        "monitoring": dict(grouped.get("monitoring", {})),
        "role": grouped.get("role", default.role),
        "local_debug_mode": grouped.get("local_debug_mode", default.local_debug_mode),
    }


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Dataclass construction from plain dicts
# ---------------------------------------------------------------------------


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort coercion of a YAML scalar to the annotated Python type."""
    origin = get_origin(annotation)
    if origin in (Union, type(int | str)):  # handles X | Y syntax
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        for arg in non_none:
            try:
                return _coerce(value, arg)
            except (TypeError, ValueError):
                continue
        return value
    if origin is Literal:
        return value
    if annotation is float:
        return float(value)
    if annotation is int:
        if isinstance(value, bool):
            raise TypeError("bool cannot coerce to int")
        return int(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low == "true":
                return True
            if low == "false":
                return False
        raise TypeError(f"Cannot coerce {value!r} to bool")
    return value


def _dataclass_from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    default = cls()
    field_types = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(default):
        value = data.get(field.name, getattr(default, field.name))
        default_val = getattr(default, field.name)
        if dataclasses.is_dataclass(default_val):
            if not isinstance(value, Mapping):
                raise TypeError(f"{cls.__name__}.{field.name} must be a mapping")
            kwargs[field.name] = _dataclass_from_dict(type(default_val), value)
        else:
            kwargs[field.name] = _coerce(value, field_types.get(field.name, field.type))
    return cls(**kwargs)


def _system_from_flat(flat: dict[str, Any]) -> OnlineRLSystemConfig:
    default = OnlineRLSystemConfig()
    return OnlineRLSystemConfig(
        rl=_dataclass_from_dict(RLTOnlineRLConfig, flat["rl"]),
        actor_service=_dataclass_from_dict(ActorServiceConfig, flat["actor_service"]),
        learner_service=_dataclass_from_dict(LearnerServiceConfig, flat["learner_service"]),
        replay=_dataclass_from_dict(ReplayConfig, flat["replay"]),
        env_driver=_dataclass_from_dict(EnvDriverConfig, flat["env_driver"]),
        monitoring=_dataclass_from_dict(MonitoringConfig, flat["monitoring"]),
        role=flat.get("role", default.role),
        local_debug_mode=flat.get("local_debug_mode", default.local_debug_mode),
    )


# ---------------------------------------------------------------------------
# Public load / save API
# ---------------------------------------------------------------------------


def load_system_config_yaml(
    path: str,
    *,
    base: OnlineRLSystemConfig | None = None,
) -> OnlineRLSystemConfig:
    """Load an OnlineRLSystemConfig from a YAML file.

    The file may contain only the fields that differ from defaults; all other
    fields retain their default values (or those from *base* if supplied).
    Relative paths inside the config are resolved against the directory that
    contains the YAML file.
    """
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a top-level mapping, got {type(payload).__name__}")
    merged = _split_for_yaml(base or OnlineRLSystemConfig())
    _deep_update(merged, payload)
    system = _system_from_flat(_flatten_yaml_sections(merged))
    return resolve_config_paths(system, path)


def save_system_config_yaml(system: OnlineRLSystemConfig, path: str) -> str:
    """Serialise *system* to *path* as YAML.

    Absolute paths are relativised before writing so the resulting file is
    portable (can be moved together with its run directory).
    Returns *path*.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    portable = relativize_config_paths(system, path)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_split_for_yaml(portable), fh, sort_keys=False)
    return path


def default_config_path(system: OnlineRLSystemConfig) -> str:
    """Canonical path where the config is saved alongside the learner checkpoint."""
    return os.path.join(system.learner_service.checkpoint_dir, DEFAULT_CONFIG_FILENAME)
