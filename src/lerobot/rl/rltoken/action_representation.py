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

"""Action-representation adapter for the RLToken online RL pipeline.

Handles two concerns:
  1. Delta encoding — convert absolute joint positions to chunk-relative deltas
     for the first (action_dim - 1) joints; the gripper (last dim) remains
     absolute because it controls open/close state, not a continuous trajectory.
  2. Quantile normalisation — linearly map the delta/abs representation to
     the [-1, 1] range using per-dim q01/q99 statistics computed from demos.

The adapter is a frozen dataclass so that it can be embedded in other frozen
configs and safely shared across threads without a lock.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.rl.rltoken.config import RLTOnlineRLConfig

try:
    import torch
    from torch import Tensor as TorchTensor

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Quantile stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantileStats:
    """Per-dimension q01 / q99 bounds used for normalisation."""

    q01: np.ndarray  # [action_dim]  float32
    q99: np.ndarray  # [action_dim]  float32


def _load_quantile_stats(path: str) -> QuantileStats:
    """Load QuantileStats from a JSON file.

    Accepts two layouts:
      {"norm_stats": {"actions": {"q01": [...], "q99": [...]}}}  (openpi-RLT)
      {"actions": {"q01": [...], "q99": [...]}}                  (flat)
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if "norm_stats" in payload:
        stats = payload["norm_stats"]["actions"]
    else:
        stats = payload["actions"]

    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return QuantileStats(q01=q01, q99=q99)


# ---------------------------------------------------------------------------
# Pure functions (NumPy)
# ---------------------------------------------------------------------------


def _zero_row_mask(chunk: np.ndarray) -> np.ndarray:
    """Return a broadcastable bool mask: True for rows that are all-zero (padding)."""
    return np.all(np.isclose(chunk, 0.0), axis=-1, keepdims=True)


def _broadcast_state0(state0: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Expand state0 to match chunk's leading batch dims for broadcasting."""
    state0 = np.asarray(state0, dtype=np.float32)
    while state0.ndim < chunk.ndim:
        state0 = np.expand_dims(state0, axis=-2)
    return state0


def _quantile_normalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    """Map x from [q01, q99] to [-1, 1].  Values outside the range are not clipped."""
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x - q01) / scale * 2.0 - 1.0


def _quantile_denormalize(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    """Inverse of _quantile_normalize: map x from [-1, 1] back to joint-angle space."""
    q01 = stats.q01.astype(np.float32, copy=False)
    q99 = stats.q99.astype(np.float32, copy=False)
    scale = q99 - q01 + 1e-6
    return (x + 1.0) * 0.5 * scale + q01


# ---------------------------------------------------------------------------
# Pure functions (PyTorch)
# ---------------------------------------------------------------------------


def _quantile_normalize_torch(x: "TorchTensor", q01: "TorchTensor", q99: "TorchTensor") -> "TorchTensor":
    scale = (q99 - q01 + 1e-6).to(x.dtype)
    return (x - q01.to(x.dtype)) / scale * 2.0 - 1.0


def _quantile_denormalize_torch(x: "TorchTensor", q01: "TorchTensor", q99: "TorchTensor") -> "TorchTensor":
    scale = (q99 - q01 + 1e-6).to(x.dtype)
    return (x + 1.0) * 0.5 * scale + q01.to(x.dtype)


def _broadcast_state0_torch(state0: "TorchTensor", chunk: "TorchTensor") -> "TorchTensor":
    while state0.dim() < chunk.dim():
        state0 = state0.unsqueeze(-2)
    return state0


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ActionRepresentationAdapter:
    """Translates between absolute joint positions and the RL action representation.

    The RL actor operates in a normalised delta space:
      - dims 0 … (action_dim - 2): delta from state0 at chunk start
      - dim  (action_dim - 1): absolute (gripper open/close)
      - all dims: quantile-normalised to [-1, 1] via per-dim q01/q99

    When action_representation == "abs", the delta conversion is skipped and
    only quantile normalisation is applied.

    Args:
        rl_config: Algorithm config; action_dim and action_representation are read.
        stats: Pre-loaded QuantileStats.
    """

    rl_config: RLTOnlineRLConfig
    stats: QuantileStats

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, rl_config: RLTOnlineRLConfig) -> ActionRepresentationAdapter | None:
        """Build an adapter from config.  Returns None if no norm_stats_path is set."""
        if not rl_config.action_norm_stats_path:
            return None
        path = rl_config.action_norm_stats_path
        if not Path(path).exists():
            raise FileNotFoundError(f"action_norm_stats_path not found: {path}")
        return cls(rl_config=rl_config, stats=_load_quantile_stats(path))

    # ------------------------------------------------------------------
    # Delta conversion helpers (NumPy)
    # ------------------------------------------------------------------

    @property
    def _delta_dim(self) -> int:
        """Number of leading dims that participate in the delta encoding."""
        return self.rl_config.action_dim - 1

    def _abs_to_delta_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Subtract state0 from the first delta_dim joints; keep gripper absolute."""
        chunk_abs = np.asarray(chunk_abs, dtype=np.float32)
        s0 = _broadcast_state0(state0, chunk_abs)
        chunk_delta = chunk_abs.copy()
        zero_mask = _zero_row_mask(chunk_abs)
        n = self._delta_dim
        chunk_delta[..., :n] = chunk_abs[..., :n] - s0[..., :n]
        # Preserve zero-padded rows (e.g. the tail of a short episode chunk).
        return np.where(zero_mask, 0.0, chunk_delta)

    def _delta_to_abs_chunk(self, chunk_delta: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Add state0 back to the first delta_dim joints; gripper is already absolute."""
        chunk_delta = np.asarray(chunk_delta, dtype=np.float32)
        s0 = _broadcast_state0(state0, chunk_delta)
        chunk_abs = chunk_delta.copy()
        n = self._delta_dim
        chunk_abs[..., :n] = chunk_delta[..., :n] + s0[..., :n]
        return chunk_abs

    def _to_representation(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_abs, dtype=np.float32)
        return self._abs_to_delta_chunk(chunk_abs, state0)

    def _from_representation(self, chunk_repr: np.ndarray, state0: np.ndarray) -> np.ndarray:
        if self.rl_config.action_representation == "abs":
            return np.asarray(chunk_repr, dtype=np.float32)
        return self._delta_to_abs_chunk(chunk_repr, state0)

    # ------------------------------------------------------------------
    # Public NumPy API
    # ------------------------------------------------------------------

    def normalize_chunk(self, chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Convert an absolute chunk to the normalised RL representation.

        Args:
            chunk_abs: [..., chunk_len, action_dim]  absolute joint positions
            state0:    [..., action_dim]              joint state at chunk start

        Returns:
            [..., chunk_len, action_dim]  in [-1, 1] (zero rows preserved as 0)
        """
        chunk_repr = self._to_representation(chunk_abs, state0)
        normalized = _quantile_normalize(chunk_repr, self.stats)
        # Re-zero padded rows that were zero in chunk_abs.
        return np.where(_zero_row_mask(chunk_abs), 0.0, normalized).astype(np.float32, copy=False)

    def denormalize_to_abs_chunk(self, chunk_norm: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Inverse: convert normalised RL chunk back to absolute joint positions.

        Args:
            chunk_norm: [..., chunk_len, action_dim]  in [-1, 1]
            state0:     [..., action_dim]              joint state at chunk start

        Returns:
            [..., chunk_len, action_dim]  absolute joint positions, float32
        """
        chunk_repr = _quantile_denormalize(np.asarray(chunk_norm, dtype=np.float32), self.stats)
        return self._from_representation(chunk_repr, state0).astype(np.float32, copy=False)

    def normalize_ref_chunk(self, ref_chunk_abs: np.ndarray, state0: np.ndarray) -> np.ndarray:
        """Alias for normalize_chunk; kept for readability at call sites."""
        return self.normalize_chunk(ref_chunk_abs, state0)

    def prepare_training_batch(self, batch_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalise all action-like fields in a sampled replay batch in-place.

        Modifies ref_chunk, action_chunk, and next_ref_chunk using
        proprio / next_proprio as the respective state0 anchors.

        Args:
            batch_np: Dict of stacked numpy arrays as returned by RLTReplayBuffer.sample().

        Returns:
            A new dict with the three action fields replaced by their normalised versions.
            All other fields are passed through unchanged.
        """
        proprio = np.asarray(batch_np["proprio"], dtype=np.float32)
        next_proprio = np.asarray(batch_np["next_proprio"], dtype=np.float32)
        out = dict(batch_np)
        out["ref_chunk"] = self.normalize_ref_chunk(batch_np["ref_chunk"], proprio)
        out["action_chunk"] = self.normalize_chunk(batch_np["action_chunk"], proprio)
        out["next_ref_chunk"] = self.normalize_ref_chunk(batch_np["next_ref_chunk"], next_proprio)
        out["proprio"] = proprio
        out["next_proprio"] = next_proprio
        return out

    # ------------------------------------------------------------------
    # PyTorch API (for the training hot-path)
    # ------------------------------------------------------------------

    def _stats_as_torch(self, device: Any = None, dtype: Any = None) -> tuple["TorchTensor", "TorchTensor"]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for the PyTorch adapter methods")
        q01 = torch.as_tensor(self.stats.q01, dtype=dtype, device=device)
        q99 = torch.as_tensor(self.stats.q99, dtype=dtype, device=device)
        return q01, q99

    def normalize_chunk_torch(self, chunk_abs: "TorchTensor", state0: "TorchTensor") -> "TorchTensor":
        """PyTorch version of normalize_chunk.

        Args:
            chunk_abs: [..., chunk_len, action_dim]  absolute joint positions
            state0:    [..., action_dim]              joint state at chunk start

        Returns:
            [..., chunk_len, action_dim]  normalised to [-1, 1], float32
        """
        q01, q99 = self._stats_as_torch(device=chunk_abs.device, dtype=torch.float32)
        chunk_abs = chunk_abs.float()
        state0 = state0.float()

        if self.rl_config.action_representation == "delta_chunk":
            s0 = _broadcast_state0_torch(state0, chunk_abs)
            chunk_repr = chunk_abs.clone()
            n = self._delta_dim
            chunk_repr[..., :n] = chunk_abs[..., :n] - s0[..., :n]
        else:
            chunk_repr = chunk_abs

        return _quantile_normalize_torch(chunk_repr, q01, q99)

    def denormalize_to_abs_chunk_torch(
        self, chunk_norm: "TorchTensor", state0: "TorchTensor"
    ) -> "TorchTensor":
        """PyTorch version of denormalize_to_abs_chunk.

        Args:
            chunk_norm: [..., chunk_len, action_dim]  in [-1, 1]
            state0:     [..., action_dim]              joint state at chunk start

        Returns:
            [..., chunk_len, action_dim]  absolute joint positions, float32
        """
        q01, q99 = self._stats_as_torch(device=chunk_norm.device, dtype=torch.float32)
        chunk_repr = _quantile_denormalize_torch(chunk_norm.float(), q01, q99)

        if self.rl_config.action_representation == "delta_chunk":
            s0 = _broadcast_state0_torch(state0.float(), chunk_repr)
            chunk_abs = chunk_repr.clone()
            n = self._delta_dim
            chunk_abs[..., :n] = chunk_repr[..., :n] + s0[..., :n]
            return chunk_abs

        return chunk_repr
