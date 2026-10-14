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

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Collection-phase constants
# ---------------------------------------------------------------------------

COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2
DEFAULT_COLLECTION_PHASE = "unknown"


def collection_phase_to_id(phase: str) -> int:
    """Map a phase string to a compact integer for storage."""
    label = str(phase).split(":", 1)[0].lower()
    if label == "warmup":
        return COLLECTION_PHASE_WARMUP
    if label == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


# ---------------------------------------------------------------------------
# Source enum
# ---------------------------------------------------------------------------


class TransitionSource(enum.IntEnum):
    """Tags how the action_chunk for a transition was produced."""

    BASE = 0   # VLA ref_chunk used directly; actor not called
    RL = 1     # Actor-refined chunk
    HUMAN = 2  # Full teleop override for the entire chunk
    MIXED = 3  # Partial teleop mid-chunk


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _as_array(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


# ---------------------------------------------------------------------------
# Per-step record collected during robot rollout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class EpisodeStepRecord:
    """Raw per-step data gathered inside execute_chunk().

    Stored in a list on the RawEpisodTrace; converted to RLTTransitions at
    episode end.  All arrays are kept as float32 here — dtype compaction
    (float16 for large tensors) happens only in RLTTransition.to_numpy().
    """

    # Features returned by Machine A for this step's observation
    z_rl: np.ndarray        # [z_dim]
    proprio: np.ndarray     # [proprio_dim]
    ref_action: np.ndarray  # [action_dim]  one step of the ref_chunk

    # Action actually sent to the robot
    action: np.ndarray      # [action_dim]

    # Environment feedback
    reward: float
    done: bool

    # Features for the *next* observation (used as TD next-state)
    next_z_rl: np.ndarray   # [z_dim]
    next_proprio: np.ndarray  # [proprio_dim]

    # Provenance
    source: int = int(TransitionSource.RL)
    collection_phase: str = DEFAULT_COLLECTION_PHASE
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EpisodeStepRecord:
        return cls(
            z_rl=_as_array(data["z_rl"], dtype=np.float32),
            proprio=_as_array(data["proprio"], dtype=np.float32),
            ref_action=_as_array(data["ref_action"], dtype=np.float32),
            action=_as_array(data["action"], dtype=np.float32),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            next_z_rl=_as_array(data["next_z_rl"], dtype=np.float32),
            next_proprio=_as_array(data["next_proprio"], dtype=np.float32),
            source=int(data.get("source", int(TransitionSource.RL))),
            collection_phase=str(data.get("collection_phase", DEFAULT_COLLECTION_PHASE)),
            success=int(data.get("success", 0)),
            intervention_flag=bool(data.get("intervention_flag", False)),
            episode_id=int(data.get("episode_id", 0)),
            step_id=int(data.get("step_id", 0)),
        )


# ---------------------------------------------------------------------------
# Chunk-level transition stored in the replay buffer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class RLTTransition:
    """One chunk-level transition as stored in the replay buffer.

    All array fields are deliberately typed as np.ndarray rather than tensors
    so this object can live on CPU without touching PyTorch.  The replay
    buffer converts to torch only at sampling time.

    Shapes (with default SO-ARM / ethernet-insertion values):
        z_rl, next_z_rl          : [z_dim=2048]
        proprio, next_proprio    : [proprio_dim=6]
        ref_chunk, next_ref_chunk: [chunk_len=10, action_dim=6]
        action_chunk             : [chunk_len=10, action_dim=6]
        rewards                  : [chunk_len=10]
        source_chunk             : [chunk_len=10]  uint8 TransitionSource per step
    """

    # Current-step features
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray

    # Executed action and outcome
    action_chunk: np.ndarray
    rewards: np.ndarray
    done: bool

    # Next-step features for TD bootstrap
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray

    # Provenance
    source: int                  # Dominant TransitionSource for the chunk
    source_chunk: np.ndarray     # Per-step source, shape [chunk_len], dtype uint8
    collection_phase: str
    success: int
    intervention_flag: bool
    episode_id: int
    step_id: int                 # Step index of the *first* step in the chunk

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Return all fields as compact numpy arrays (large tensors in float16)."""
        return {
            "z_rl": _as_array(self.z_rl, dtype=np.float16),
            "proprio": _as_array(self.proprio, dtype=np.float32),
            "ref_chunk": _as_array(self.ref_chunk, dtype=np.float16),
            "action_chunk": _as_array(self.action_chunk, dtype=np.float16),
            "rewards": _as_array(self.rewards, dtype=np.float32),
            "done": _as_array(self.done, dtype=np.bool_),
            "next_z_rl": _as_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _as_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _as_array(self.next_ref_chunk, dtype=np.float16),
            "source": _as_array(self.source, dtype=np.uint8),
            "source_chunk": _as_array(self.source_chunk, dtype=np.uint8),
            "collection_phase_id": _as_array(
                collection_phase_to_id(self.collection_phase), dtype=np.uint8
            ),
            "success": _as_array(self.success, dtype=np.int8),
            "intervention_flag": _as_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _as_array(self.episode_id, dtype=np.int32),
            "step_id": _as_array(self.step_id, dtype=np.int32),
        }

    def to_journal_record(self) -> dict[str, Any]:
        """Like to_numpy() but keeps the human-readable collection_phase string for the journal."""
        return {**self.to_numpy(), "collection_phase": self.collection_phase}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RLTTransition:
        """Reconstruct from a dict (e.g. loaded from the journal)."""
        ref_chunk = _as_array(data["ref_chunk"])
        chunk_len = ref_chunk.shape[0]
        default_source_chunk = np.full(
            (chunk_len,), int(data.get("source", int(TransitionSource.RL))), dtype=np.uint8
        )
        return cls(
            z_rl=_as_array(data["z_rl"]),
            proprio=_as_array(data["proprio"]),
            ref_chunk=ref_chunk,
            action_chunk=_as_array(data["action_chunk"]),
            rewards=_as_array(data["rewards"]),
            done=bool(data["done"]),
            next_z_rl=_as_array(data["next_z_rl"]),
            next_proprio=_as_array(data["next_proprio"]),
            next_ref_chunk=_as_array(data["next_ref_chunk"]),
            source=int(data.get("source", int(TransitionSource.RL))),
            source_chunk=_as_array(data.get("source_chunk", default_source_chunk), dtype=np.uint8),
            collection_phase=str(data.get("collection_phase", DEFAULT_COLLECTION_PHASE)),
            success=int(data.get("success", 0)),
            intervention_flag=bool(data.get("intervention_flag", False)),
            episode_id=int(data.get("episode_id", 0)),
            step_id=int(data.get("step_id", 0)),
        )
