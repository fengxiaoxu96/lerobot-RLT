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

"""LearnerService (B2 process) and training utilities for the RLToken online RL pipeline.

Public API:
    RLTTrainState      — mutable dataclass holding all network/optimizer state.
    init_train_state   — construct networks, target copies, and optimizers.
    polyak_update      — in-place Polyak / EMA parameter update.
    LearnerService     — training outer shell: sampling, updating, checkpointing.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import pickle
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.rl.rltoken.action_representation import ActionRepresentationAdapter
from lerobot.rl.rltoken.config import LearnerServiceConfig, RLTOnlineRLConfig
from lerobot.rl.rltoken.networks import ChunkActor, TwinCritic, compute_critic_loss
from lerobot.rl.rltoken.transition import TransitionSource

logger = logging.getLogger(__name__)

# Keys that are non-numeric and must be skipped during tensor conversion
_NON_NUMERIC_KEYS = frozenset({"collection_phase"})


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RLTTrainState:
    """Mutable container for all actor-critic training state."""

    actor: ChunkActor
    target_actor: ChunkActor
    critic: TwinCritic
    target_critic: TwinCritic
    actor_optimizer: torch.optim.Adam
    critic_optimizer: torch.optim.Adam
    global_step: int = 0
    actor_version: int = 0


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_train_state(config: RLTOnlineRLConfig) -> RLTTrainState:
    """Build actor, target_actor (deepcopy), critic, target_critic (deepcopy) and their Adam optimizers.

    Target networks are initialised from the main network weights and frozen
    (requires_grad=False).  They are always kept in eval mode.
    """
    actor = ChunkActor(config)
    target_actor = copy.deepcopy(actor)
    critic = TwinCritic(config)
    target_critic = copy.deepcopy(critic)

    # Target networks: eval mode, no grad
    target_actor.eval()
    target_critic.eval()
    for p in target_actor.parameters():
        p.requires_grad_(False)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    return RLTTrainState(
        actor=actor,
        target_actor=target_actor,
        critic=critic,
        target_critic=target_critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )


# ---------------------------------------------------------------------------
# Polyak update
# ---------------------------------------------------------------------------


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """In-place Polyak / EMA update: target_param = (1-tau)*target_param + tau*source_param."""
    with torch.no_grad():
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.mul_(1.0 - tau)
            tgt_p.data.add_(tau * src_p.data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_source_chunk(batch_np: dict[str, np.ndarray], chunk_len: int) -> dict[str, np.ndarray]:
    """If 'source_chunk' is absent, broadcast scalar 'source' to [B, chunk_len]."""
    if "source_chunk" in batch_np:
        return batch_np
    batch_np = dict(batch_np)
    source = np.asarray(batch_np["source"], dtype=np.uint8).reshape(-1, 1)
    batch_np["source_chunk"] = np.repeat(source, int(chunk_len), axis=1)
    return batch_np


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically via tmp file + os.replace()."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_pickle(path: str, payload: Any) -> None:
    """Write pickle atomically via tmp file + os.replace()."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# LearnerService
# ---------------------------------------------------------------------------


class LearnerService:
    """B2 training loop: samples batches from replay, updates actor-critic, exports snapshots.

    Constructor args:
        rl_config       — algorithm hyperparameters.
        learner_config  — service configuration (paths, intervals, batch size).
        replay_client   — ReplayClient (or NullReplayClient) instance.
        run_dir         — root directory for this run; metrics/ is created beneath it.
    """

    def __init__(
        self,
        rl_config: RLTOnlineRLConfig,
        learner_config: LearnerServiceConfig,
        replay_client: Any,
        *,
        run_dir: str,
    ) -> None:
        self._rl_config = rl_config
        self._learner_config = learner_config
        self._replay_client = replay_client
        self._run_dir = run_dir

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self._state = init_train_state(rl_config)
        self._state.actor.to(self._device)
        self._state.target_actor.to(self._device)
        self._state.critic.to(self._device)
        self._state.target_critic.to(self._device)

        # Action representation adapter (may be None if no norm stats path is configured)
        self._adapter: ActionRepresentationAdapter | None = ActionRepresentationAdapter.from_config(
            rl_config
        )

        # Progress / scheduling state
        self._warmup_ready_adds_total: int | None = None
        self._pending_update_budget: int = 0

        # Rate-limited logging timestamps
        self._last_warmup_log_time: float = 0.0
        self._last_budget_idle_log_time: float = 0.0
        self._freeze_logged: bool = False

        # Paths
        self._status_path = os.path.join(run_dir, "metrics", "learner_status.json")
        self._log_path = os.path.join(run_dir, "metrics", "learner_log.jsonl")
        self._checkpoint_dir = learner_config.checkpoint_dir
        self._snapshot_path = learner_config.actor_snapshot_path

        # Create required directories
        os.makedirs(os.path.dirname(self._status_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)

        # Attempt to restore from checkpoint
        self._load_latest_checkpoint()

        # Export the initial (possibly restored) actor snapshot
        self.export_actor_snapshot(force=True)

        logger.info(
            "LearnerService initialized device=%s step=%s actor_version=%s",
            self._device,
            self._state.global_step,
            self._state.actor_version,
        )

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def train_once(self, stop_event: Any = None) -> dict[str, float] | None:
        """Perform one gradient update step.

        Returns a metrics dict on success, or None when the step is skipped
        (warmup not reached, budget exhausted, frozen, or stop requested).
        """
        stats = self._replay_client.stats()
        progress = self._refresh_progress(stats)
        replay_size = int(progress["replay_size"])

        # 1. Not enough transitions yet
        if replay_size < self._rl_config.warmup_min_size:
            now = time.time()
            if now - self._last_warmup_log_time >= 5.0:
                logger.info(
                    "Waiting for warmup data: replay_size=%s / required=%s",
                    replay_size,
                    self._rl_config.warmup_min_size,
                )
                self._last_warmup_log_time = now
            return None

        # 2. Frozen after warmup
        if self._rl_config.freeze_after_warmup and bool(progress["ready_for_online"]):
            if not self._freeze_logged:
                self.export_actor_snapshot(force=True)
                logger.info(
                    "freeze_after_warmup=True; training disabled at global_step=%s",
                    progress["global_step"],
                )
                self._freeze_logged = True
            return None

        # 3. Update budget exhausted
        if self._pending_update_budget <= 0:
            now = time.time()
            if now - self._last_budget_idle_log_time >= 5.0:
                logger.info(
                    "Learner caught up: replay_size=%s adds_total=%s global_step=%s",
                    replay_size,
                    progress["adds_total"],
                    progress["global_step"],
                )
                self._last_budget_idle_log_time = now
            return None

        if stop_event is not None and stop_event.is_set():
            return None

        # ------------------------------------------------------------------
        # Sample batch and prepare tensors
        # ------------------------------------------------------------------
        batch_np = _ensure_source_chunk(
            self._replay_client.sample_batch(self._learner_config.sample_batch_size),
            self._rl_config.chunk_len,
        )
        if self._adapter is not None:
            batch_np = self._adapter.prepare_training_batch(batch_np)
        batch = self._to_tensors(batch_np)

        bc_weight, q_weight = self._resolve_weights(progress)
        metrics: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Critic update
        # ------------------------------------------------------------------
        self._state.critic.train()
        self._state.actor.eval()
        self._state.critic_optimizer.zero_grad()

        critic_loss, critic_metrics = compute_critic_loss(
            critic=self._state.critic,
            target_actor=self._state.target_actor,
            target_critic=self._state.target_critic,
            z_rl=batch["z_rl"],
            proprio=batch["proprio"],
            action_chunk=batch["action_chunk"],
            rewards=batch["rewards"],
            done=batch["done"],
            next_z_rl=batch["next_z_rl"],
            next_proprio=batch["next_proprio"],
            next_ref_chunk=batch["next_ref_chunk"],
            gamma=self._rl_config.gamma,
        )
        critic_loss.backward()
        self._state.critic_optimizer.step()
        metrics.update({k: float(v.item()) for k, v in critic_metrics.items()})

        # ------------------------------------------------------------------
        # Actor update (every actor_update_period steps)
        # ------------------------------------------------------------------
        did_actor_update = False
        if (self._state.global_step + 1) % self._rl_config.actor_update_period == 0:
            self._state.actor.train()   # enables reference dropout via member method
            self._state.critic.eval()  # critic used for Q-values only, no param update here
            self._state.actor_optimizer.zero_grad()

            # Reference dropout (member method; only active when actor.training=True)
            dropped_ref = self._state.actor.apply_reference_dropout(
                batch["ref_chunk"],
                self._rl_config.reference_dropout_prob,
            )
            sampled_chunk, _ = self._state.actor.sample(
                batch["z_rl"], batch["proprio"], dropped_ref, deterministic=False
            )

            # Source-aware BC target: HUMAN/MIXED → recorded action_chunk; BASE/RL → ref_chunk
            source_chunk = batch["source_chunk"]  # [B, chunk_len] float32
            human_mask = (
                (source_chunk == float(int(TransitionSource.HUMAN)))
                | (source_chunk == float(int(TransitionSource.MIXED)))
            ).float().unsqueeze(-1)  # [B, chunk_len, 1]
            bc_target = human_mask * batch["action_chunk"] + (1.0 - human_mask) * batch["ref_chunk"]

            # BC penalty
            bc_penalty = F.mse_loss(sampled_chunk, bc_target)

            # Q loss (use q1 only; matches reference)
            q1, _ = self._state.critic.q_values(batch["z_rl"], batch["proprio"], sampled_chunk)
            q_loss = -q1.mean()

            # Delta penalty (consecutive-step smoothness in absolute space)
            delta_penalty = torch.tensor(0.0, device=self._device)
            if self._rl_config.delta_weight > 0.0 and self._adapter is not None:
                pred_abs = self._adapter.denormalize_to_abs_chunk_torch(
                    sampled_chunk, batch["proprio"]
                )
                target_abs = self._adapter.denormalize_to_abs_chunk_torch(
                    bc_target, batch["proprio"]
                )
                n = self._rl_config.action_dim - 1  # exclude gripper dim
                pred_step_delta = pred_abs[:, 1:, :n] - pred_abs[:, :-1, :n]
                target_step_delta = target_abs[:, 1:, :n] - target_abs[:, :-1, :n]
                delta_penalty = F.mse_loss(pred_step_delta, target_step_delta)

            actor_loss = (
                bc_weight * bc_penalty
                + q_weight * q_loss
                + self._rl_config.delta_weight * delta_penalty
            )
            actor_loss.backward()
            self._state.actor_optimizer.step()

            # Polyak update both target networks
            polyak_update(self._state.actor, self._state.target_actor, self._rl_config.target_tau)
            polyak_update(self._state.critic, self._state.target_critic, self._rl_config.target_tau)

            self._state.actor_version += 1
            did_actor_update = True

            metrics.update({
                "actor_loss": float(actor_loss.item()),
                "bc_penalty": float(bc_penalty.item()),
                "q_actor": float((-q_loss).item()),
                "delta_penalty": float(delta_penalty.item()),
                "bc_weight": bc_weight,
                "q_weight": q_weight,
                "delta_weight": float(self._rl_config.delta_weight),
            })

        self._state.global_step += 1

        # Refresh progress (updates pending_update_budget) after the step
        progress = self._refresh_progress(stats)
        metrics.update({
            "global_step": float(self._state.global_step),
            "actor_version": float(self._state.actor_version),
            "replay_size": float(progress["replay_size"]),
            "adds_total": float(progress["adds_total"]),
            "pending_update_budget": float(progress["pending_update_budget"]),
            "warmup_required_updates": float(progress["warmup_required_updates"]),
            "ready_for_online": float(progress["ready_for_online"]),
            "did_actor_update": float(did_actor_update),
        })

        self._maybe_save_checkpoint()
        self._maybe_export_actor_snapshot()
        _append_jsonl(self._log_path, metrics)

        step = self._state.global_step
        if step <= 3 or step % 50 == 0:
            logger.info(
                "Learner step=%s actor_version=%s critic_loss=%.4f actor_loss=%.4f "
                "budget=%s bc_w=%.3f q_w=%.3f",
                step,
                self._state.actor_version,
                metrics.get("critic_loss", 0.0),
                metrics.get("actor_loss", 0.0),
                int(progress["pending_update_budget"]),
                metrics.get("bc_weight", 0.0),
                metrics.get("q_weight", 0.0),
            )

        return metrics

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def _to_tensors(self, batch_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Convert numpy batch to float32 tensors on the training device.

        Skips keys in _NON_NUMERIC_KEYS and any array with a string/object dtype.
        """
        result: dict[str, torch.Tensor] = {}
        for key, val in batch_np.items():
            if key in _NON_NUMERIC_KEYS:
                continue
            arr = np.asarray(val)
            if arr.dtype.kind in ("U", "S", "O"):
                continue
            result[key] = torch.from_numpy(arr.astype(np.float32)).to(self._device)
        return result

    # ------------------------------------------------------------------
    # Progress / scheduling
    # ------------------------------------------------------------------

    def _warmup_required_updates(self) -> int:
        """Number of gradient steps required to complete warmup training."""
        if self._warmup_ready_adds_total is None:
            return 0
        if self._rl_config.warmup_post_collect_updates is not None:
            return int(self._rl_config.warmup_post_collect_updates)
        return self._warmup_ready_adds_total * self._rl_config.grad_updates_per_cycle

    def _desired_total_updates(self, replay_size: int, adds_total: int) -> int:
        if replay_size < self._rl_config.warmup_min_size or self._warmup_ready_adds_total is None:
            return 0
        warmup_required_updates = self._warmup_required_updates()
        if self._rl_config.warmup_post_collect_updates is None:
            return adds_total * self._rl_config.grad_updates_per_cycle
        online_adds = max(adds_total - self._warmup_ready_adds_total, 0)
        return warmup_required_updates + online_adds * self._rl_config.grad_updates_per_cycle

    def _refresh_progress(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Recompute pending_update_budget from replay stats; write status JSON."""
        replay_size = int(stats["size"])
        adds_total = int(stats.get("adds_total", replay_size))
        global_step = self._state.global_step
        actor_version = self._state.actor_version

        # Latch warmup-ready timestamp on first crossing
        if replay_size >= self._rl_config.warmup_min_size and self._warmup_ready_adds_total is None:
            self._warmup_ready_adds_total = adds_total
            required = self._warmup_required_updates()
            logger.info(
                "Warmup ready latched: replay_size=%s adds_total=%s required_updates=%s",
                replay_size, adds_total, required,
            )

        warmup_required_updates = self._warmup_required_updates()
        desired_total = self._desired_total_updates(replay_size, adds_total)
        self._pending_update_budget = max(desired_total - global_step, 0)
        ready_for_online = (
            replay_size >= self._rl_config.warmup_min_size
            and global_step >= warmup_required_updates
        )
        progress: dict[str, Any] = {
            "replay_size": replay_size,
            "adds_total": adds_total,
            "global_step": global_step,
            "actor_version": actor_version,
            "pending_update_budget": self._pending_update_budget,
            "warmup_required_updates": warmup_required_updates,
            "ready_for_online": ready_for_online,
        }
        self._write_status(progress)
        return progress

    def _write_status(self, progress: dict[str, Any]) -> None:
        _atomic_write_json(
            self._status_path,
            {
                "replay_size": int(progress["replay_size"]),
                "adds_total": int(progress["adds_total"]),
                "global_step": int(progress["global_step"]),
                "actor_version": int(progress["actor_version"]),
                "pending_update_budget": int(progress["pending_update_budget"]),
                "warmup_ready_adds_total": self._warmup_ready_adds_total,
                "warmup_required_updates": int(progress["warmup_required_updates"]),
                "warmup_post_collect_updates": self._rl_config.warmup_post_collect_updates,
                "training_frozen": bool(
                    self._rl_config.freeze_after_warmup and progress["ready_for_online"]
                ),
                "ready_for_online": bool(progress["ready_for_online"]),
                "update_ratio": int(self._rl_config.grad_updates_per_cycle),
                "timestamp": time.time(),
            },
        )

    def _resolve_weights(self, progress: dict[str, Any]) -> tuple[float, float]:
        """Return (bc_weight, q_weight): warmup values before warmup_required_updates, online after."""
        warmup_required = int(progress["warmup_required_updates"])
        global_step = int(progress["global_step"])
        if warmup_required > 0 and global_step < warmup_required:
            return float(self._rl_config.warmup_bc_weight), float(self._rl_config.warmup_q_weight)
        return float(self._rl_config.online_bc_weight), float(self._rl_config.online_q_weight)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> str:
        """Save full training state (networks + optimizers + progress) as pickle."""
        step = self._state.global_step
        path = os.path.join(self._checkpoint_dir, "latest.pkl")
        payload = {
            "state": {
                "actor_state_dict": self._state.actor.state_dict(),
                "target_actor_state_dict": self._state.target_actor.state_dict(),
                "critic_state_dict": self._state.critic.state_dict(),
                "target_critic_state_dict": self._state.target_critic.state_dict(),
                "actor_optimizer_state_dict": self._state.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self._state.critic_optimizer.state_dict(),
                "global_step": step,
                "actor_version": self._state.actor_version,
            },
            "progress": {
                "warmup_ready_adds_total": self._warmup_ready_adds_total,
            },
        }
        _atomic_write_pickle(path, payload)
        # Also save a step-numbered copy for crash recovery
        step_path = os.path.join(self._checkpoint_dir, f"step_{step}.pkl")
        with open(step_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved checkpoint step=%s path=%s", step, path)
        return path

    def _load_latest_checkpoint(self) -> None:
        """Restore training state from latest.pkl if it exists."""
        path = os.path.join(self._checkpoint_dir, "latest.pkl")
        if not os.path.exists(path):
            logger.debug("No checkpoint found at %s; starting fresh.", path)
            return
        with open(path, "rb") as f:
            payload = pickle.load(f)
        s = payload["state"]
        self._state.actor.load_state_dict(s["actor_state_dict"])
        self._state.target_actor.load_state_dict(s["target_actor_state_dict"])
        self._state.critic.load_state_dict(s["critic_state_dict"])
        self._state.target_critic.load_state_dict(s["target_critic_state_dict"])
        self._state.actor_optimizer.load_state_dict(s["actor_optimizer_state_dict"])
        self._state.critic_optimizer.load_state_dict(s["critic_optimizer_state_dict"])
        self._state.global_step = int(s["global_step"])
        self._state.actor_version = int(s["actor_version"])
        prog = payload.get("progress", {})
        raw_warmup = prog.get("warmup_ready_adds_total")
        self._warmup_ready_adds_total = None if raw_warmup is None else int(raw_warmup)
        logger.info(
            "Restored checkpoint from %s step=%s actor_version=%s",
            path,
            self._state.global_step,
            self._state.actor_version,
        )

    def _maybe_save_checkpoint(self) -> None:
        if self._state.global_step % self._learner_config.checkpoint_interval_steps == 0:
            self.save_checkpoint()

    # ------------------------------------------------------------------
    # Actor snapshot export
    # ------------------------------------------------------------------

    def export_actor_snapshot(self, force: bool = False) -> None:
        """Write actor.state_dict() + version + global_step to snapshot_path atomically."""
        version = self._state.actor_version
        if not force and version == 0:
            return
        payload = {
            "version": version,
            "global_step": self._state.global_step,
            "actor_state_dict": self._state.actor.state_dict(),
        }
        _atomic_write_pickle(self._snapshot_path, payload)
        logger.debug("Exported actor snapshot version=%s path=%s", version, self._snapshot_path)

    def _maybe_export_actor_snapshot(self) -> None:
        if self._state.global_step % self._learner_config.push_actor_interval_steps == 0:
            self.export_actor_snapshot(force=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run_forever(self, stop_event: Any = None) -> None:
        """Loop calling train_once(), sleeping poll_interval_sec when returning None."""
        logger.info("LearnerService entering training loop.")
        while stop_event is None or not stop_event.is_set():
            metrics = self.train_once(stop_event=stop_event)
            if metrics is None:
                time.sleep(self._learner_config.poll_interval_sec)
        logger.info("LearnerService training loop stopped.")

    def flush_artifacts(self) -> None:
        """Save checkpoint and export actor snapshot before shutdown."""
        logger.info("Flushing learner artifacts before shutdown.")
        if self._state.global_step > 0:
            self.save_checkpoint()
        self.export_actor_snapshot(force=True)
