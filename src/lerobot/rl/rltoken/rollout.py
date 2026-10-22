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

"""RLToken robot rollout: environment, episode orchestration, and online RL data collection.

Public API:
    StepResult                — per-step result from SOArmEnv.execute_chunk()
    SOArmEnv                  — rate-controlled chunk executor for SO-ARM hardware
    ChunkFeatures             — Machine A features for one chunk
    RolloutPhaseController    — phase state machine (warmup → online)
    KeyboardSignalBridge      — non-blocking keyboard signal queue
    EnvDriver                 — single-episode orchestrator
    RolloutOrchestrator       — top-level episode loop

Signal constants (keyboard → signal name):
    n → SIG_NEXT_EPISODE, s → SIG_MARK_SUCCESS, f → SIG_MARK_FAILURE
    a → SIG_SET_ACTOR_MODE, b → SIG_SET_BASE_MODE
    t → SIG_TOGGLE_TELEOP,  q → SIG_QUIT
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from lerobot.rl.rltoken.config import EnvDriverConfig, OnlineRLSystemConfig, RLTOnlineRLConfig
from lerobot.rl.rltoken.inference import ActorClient, MachineAFeatureClient, maybe_refine_chunk
from lerobot.rl.rltoken.replay import (
    NullReplayClient,
    ReplayClient,
    build_chunk_transitions_from_episode,
)
from lerobot.rl.rltoken.transition import EpisodeStepRecord, TransitionSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Motor layout
# ---------------------------------------------------------------------------

SO_ARM_MOTOR_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class StepResult:
    """Per-step result from SOArmEnv.execute_chunk().

    The `done` field is always False here; the caller sets it on the terminal
    step after the full episode completes.
    """

    obs: dict               # robot observation at start of step (from get_observation())
    action: np.ndarray      # action vector [action_dim] commanded (from abs_chunk[i])
    actual_action: np.ndarray  # action actually sent after clamping (from send_action() return)
    next_obs: dict          # robot observation after action
    reward: float           # reward signal (0.0 if no reward_fn)
    done: bool              # True only for the terminal step of the episode (set by caller)


# ---------------------------------------------------------------------------
# SOArmEnv
# ---------------------------------------------------------------------------


class SOArmEnv:
    """Rate-controlled chunk executor for SO-ARM hardware.

    Wraps a SOFollower or ThreadSafeRobot and exposes chunk-level execution
    at a configurable control frequency.
    """

    def __init__(self, robot: Any, config: EnvDriverConfig) -> None:
        self._robot = robot
        self._config = config

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def obs_to_state(self, obs: dict) -> np.ndarray:
        """Extract [proprio_dim] float32 state vector from robot obs dict.

        Motor position values are extracted in SO_ARM_MOTOR_NAMES order.
        """
        return np.array(
            [obs[f"{name}.pos"] for name in SO_ARM_MOTOR_NAMES],
            dtype=np.float32,
        )

    def reset(self, reset_pose: np.ndarray | None = None) -> dict:
        """Optionally send reset_pose, sleep 2 s, then return current observation.

        Args:
            reset_pose: [action_dim] absolute joint targets, or None to skip reset.

        Returns:
            Robot observation dict from get_observation().
        """
        if reset_pose is not None:
            action_dict = {
                f"{name}.pos": float(reset_pose[i])
                for i, name in enumerate(SO_ARM_MOTOR_NAMES)
            }
            self._robot.send_action(action_dict)
            time.sleep(2.0)
        return self._robot.get_observation()

    def execute_chunk(
        self,
        abs_chunk: np.ndarray,
        *,
        reward_fn: Any | None = None,
    ) -> list[StepResult]:
        """Execute abs_chunk at config.control_frequency_hz.

        For each step i:
          1. get_observation() → obs
          2. send_action(abs_chunk[i]) → actual_action_dict
          3. sleep remainder of control period
          4. get_observation() → next_obs
          5. compute reward (0.0 if reward_fn is None)
          6. append StepResult(done=False)

        The `done` field is always False here; the caller sets it on the
        terminal step after episode completion.
        """
        dt = 1.0 / self._config.control_frequency_hz
        chunk_len = abs_chunk.shape[0]
        results: list[StepResult] = []

        for i in range(chunk_len):
            t_start = time.monotonic()

            obs = self._robot.get_observation()
            action = abs_chunk[i].astype(np.float32)

            action_dict = {
                f"{name}.pos": float(action[j])
                for j, name in enumerate(SO_ARM_MOTOR_NAMES)
            }
            actual_action_raw = self._robot.send_action(action_dict)

            # Rate-control: wait out the remainder of the control period
            elapsed = time.monotonic() - t_start
            remaining = dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

            next_obs = self._robot.get_observation()
            reward = float(reward_fn(obs, action, next_obs)) if reward_fn is not None else 0.0

            # Unpack returned action dict → array (fall back to commanded action on failure)
            if isinstance(actual_action_raw, dict):
                actual_action = np.array(
                    [
                        actual_action_raw.get(f"{name}.pos", float(action[j]))
                        for j, name in enumerate(SO_ARM_MOTOR_NAMES)
                    ],
                    dtype=np.float32,
                )
            else:
                actual_action = action.copy()

            results.append(
                StepResult(
                    obs=obs,
                    action=action,
                    actual_action=actual_action,
                    next_obs=next_obs,
                    reward=reward,
                    done=False,
                )
            )

        return results


# ---------------------------------------------------------------------------
# ChunkFeatures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class ChunkFeatures:
    """Machine A features for one chunk."""

    z_rl: np.ndarray       # [z_dim]
    proprio: np.ndarray    # [proprio_dim]
    ref_chunk: np.ndarray  # [chunk_len, action_dim]


# ---------------------------------------------------------------------------
# RolloutPhaseController
# ---------------------------------------------------------------------------


class RolloutPhaseController:
    """Polls learner_status.json to implement the warmup → online phase transition.

    Reads the JSON file written by LearnerService at
    ``{run_dir}/metrics/learner_status.json``.  The file is expected to contain
    at least ``{"ready_for_online": bool, ...}``.

    Results are cached; re-poll at most once per ``poll_interval_sec``.
    On file-read failure or missing ``ready_for_online`` key the phase stays
    (or falls back to) ``"warmup"``.
    """

    def __init__(self, status_path: str, *, poll_interval_sec: float = 2.0) -> None:
        self._status_path = status_path
        self._poll_interval_sec = poll_interval_sec
        self._cached_phase: str = "warmup"
        self._cached_stats: dict[str, Any] = {}
        self._last_poll_time: float = 0.0

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self._last_poll_time < self._poll_interval_sec:
            return
        self._last_poll_time = now
        try:
            with open(self._status_path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.loads(f.read())
            self._cached_stats = data
            self._cached_phase = "online" if data.get("ready_for_online", False) else "warmup"
        except Exception:
            # File not yet created, parse error, or IO error — stay in warmup
            self._cached_phase = "warmup"

    def current_phase(self) -> str:
        """Returns ``'warmup'`` or ``'online'``.

        Caches the result and re-polls at most every ``poll_interval_sec``.
        On any read failure returns ``'warmup'``.
        """
        self._refresh()
        return self._cached_phase

    def should_use_actor(self) -> bool:
        """Returns True when ``current_phase() == 'online'``."""
        return self.current_phase() == "online"

    def get_learner_stats(self) -> dict[str, Any]:
        """Returns latest parsed status dict, or ``{}`` if not yet available."""
        self._refresh()
        return self._cached_stats


# ---------------------------------------------------------------------------
# Keyboard signal constants and bridge
# ---------------------------------------------------------------------------

SIG_NEXT_EPISODE: str = "next_episode"
SIG_MARK_SUCCESS: str = "mark_success"
SIG_MARK_FAILURE: str = "mark_failure"
SIG_SET_ACTOR_MODE: str = "set_actor_mode"
SIG_SET_BASE_MODE: str = "set_base_mode"
SIG_TOGGLE_TELEOP: str = "toggle_teleop"
SIG_QUIT: str = "quit"

_KEY_TO_SIGNAL: dict[str, str] = {
    "n": SIG_NEXT_EPISODE,
    "s": SIG_MARK_SUCCESS,
    "f": SIG_MARK_FAILURE,
    "a": SIG_SET_ACTOR_MODE,
    "b": SIG_SET_BASE_MODE,
    "t": SIG_TOGGLE_TELEOP,
    "q": SIG_QUIT,
}

_pynput_warning_logged: bool = False


class KeyboardSignalBridge:
    """Non-blocking keyboard listener backed by pynput.

    Starts a daemon thread listener on construction.  Falls back to a no-op
    (``get_signal()`` always returns ``None``) if pynput is not installed,
    logging a warning exactly once.

    The internal signal queue has ``maxsize=16``; overflow signals are silently
    dropped.
    """

    def __init__(self) -> None:
        global _pynput_warning_logged

        self._queue: queue.Queue[str] = queue.Queue(maxsize=16)
        self._listener: Any | None = None

        try:
            from pynput import keyboard  # type: ignore[import]

            def _on_press(key: Any) -> None:
                try:
                    char = key.char
                    if char in _KEY_TO_SIGNAL:
                        try:
                            self._queue.put_nowait(_KEY_TO_SIGNAL[char])
                        except queue.Full:
                            pass
                except AttributeError:
                    # Special key (shift, ctrl, …) — ignore
                    pass

            listener = keyboard.Listener(on_press=_on_press)
            listener.daemon = True
            listener.start()
            self._listener = listener

        except ImportError:
            if not _pynput_warning_logged:
                logger.warning(
                    "pynput is not installed; keyboard signals are disabled. "
                    "Install with: pip install pynput"
                )
                _pynput_warning_logged = True

    def get_signal(self) -> str | None:
        """Non-blocking; returns the next pending signal or ``None``."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        """Stop the keyboard listener thread."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# EnvDriver
# ---------------------------------------------------------------------------


class EnvDriver:
    """Orchestrates Machine A + Actor + Env + Replay for one episode.

    Handles:
    - Querying Machine A for (z_rl, ref_chunk, proprio) features per chunk
    - Optionally calling ActorClient.infer() to refine the chunk
    - Executing the chunk on the robot via SOArmEnv
    - Post-processing: assigning next_z_rl retroactively per chunk
    - Building RLTTransitions and pushing to replay (unless eval_actor_only)
    """

    def __init__(
        self,
        env: SOArmEnv,
        machine_a_client: MachineAFeatureClient,
        actor_client: ActorClient,
        replay_client: Any,  # ReplayClient | NullReplayClient
        rl_config: RLTOnlineRLConfig,
        env_config: EnvDriverConfig,
        *,
        eval_actor_only: bool = False,
        metrics_path: str | None = None,
    ) -> None:
        self._env = env
        self._machine_a_client = machine_a_client
        self._actor_client = actor_client
        self._replay_client = replay_client
        self._rl_config = rl_config
        self._env_config = env_config
        self._eval_actor_only = eval_actor_only
        self._metrics_path = metrics_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_chunk_features(self, obs: dict) -> ChunkFeatures:
        """Query Machine A and return ChunkFeatures.

        Validates that ``z_rl`` and ``ref_chunk`` are present in the payload.
        Proprio is taken from the Machine A payload if provided; otherwise
        extracted from the robot observation (or from ``obs['state']`` if that
        key is present).
        """
        payload = self._machine_a_client.get_features(obs)

        if "z_rl" not in payload:
            raise KeyError(
                f"Machine A response is missing 'z_rl'. Keys present: {list(payload.keys())}"
            )
        if "ref_chunk" not in payload:
            raise KeyError(
                f"Machine A response is missing 'ref_chunk'. Keys present: {list(payload.keys())}"
            )

        z_rl = np.asarray(payload["z_rl"], dtype=np.float32)
        ref_chunk = np.asarray(payload["ref_chunk"], dtype=np.float32)

        if "proprio" in payload:
            proprio = np.asarray(payload["proprio"], dtype=np.float32)
        elif "state" in obs:
            proprio = np.asarray(
                obs["state"][: self._rl_config.proprio_dim], dtype=np.float32
            )
        else:
            proprio = self._env.obs_to_state(obs)

        return ChunkFeatures(z_rl=z_rl, proprio=proprio, ref_chunk=ref_chunk)

    def _select_action_chunk(
        self,
        features: ChunkFeatures,
        episode_id: int,
        step_id: int,
        use_actor: bool,
    ) -> tuple[np.ndarray, int, int]:
        """Select an action chunk.

        Returns:
            (action_chunk [chunk_len, action_dim], source int, actor_param_version int)

        If use_actor is True, calls maybe_refine_chunk() (falls back to ref_chunk on
        error when env_config.safe_fallback_to_ref is True).
        If use_actor is False, returns ref_chunk with source=BASE and version=-1.
        """
        if use_actor:
            import uuid

            result = maybe_refine_chunk(
                self._actor_client,
                z_rl=features.z_rl,
                proprio=features.proprio,
                ref_chunk=features.ref_chunk,
                request_id=str(uuid.uuid4()),
                episode_id=episode_id,
                step_id=step_id,
                deterministic=self._env_config.actor_deterministic,
                on_error_fallback=self._env_config.safe_fallback_to_ref,
            )
            if result.error:
                logger.warning(
                    "Actor refinement error (episode=%d step=%d): %s",
                    episode_id,
                    step_id,
                    result.error,
                )
            return result.refined_chunk, result.source, result.actor_param_version

        # Base mode: return ref_chunk unchanged
        return (
            np.asarray(features.ref_chunk, dtype=np.float32),
            int(TransitionSource.BASE),
            -1,
        )

    # ------------------------------------------------------------------
    # Main episode method
    # ------------------------------------------------------------------

    def run_episode(
        self,
        episode_id: int,
        *,
        use_actor: bool = False,
        collection_phase: str = "warmup",
        done_signal: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Run one episode and optionally push transitions to the replay buffer.

        Algorithm:
          1. Reset the environment.
          2. Loop chunks until done_signal fires or max_steps is reached.
             For each chunk:
               a. Query Machine A for (z_rl, ref_chunk, proprio).
               b. Select action chunk (actor or ref_chunk).
               c. Execute chunk on robot (rate-controlled).
               d. Check done_signal() after chunk completes.
          3. Query Machine A once more on terminal obs to get next_z_rl for
             the last chunk (post-processing pass).
          4. Build EpisodeStepRecord list (retroactively assigning next_z_rl).
          5. Build RLTTransitions via build_chunk_transitions_from_episode()
             and push to replay (unless eval_actor_only).

        Args:
            episode_id: Integer counter for provenance.
            use_actor:  If True, query ActorClient to refine each chunk.
            collection_phase: ``'warmup'`` or ``'online'``; stored in transitions.
            done_signal: Callable returning True to end the episode early.
                         Called once after each chunk executes.

        Returns:
            Summary dict: episode_id, success, n_steps, transitions,
            collection_phase, use_actor.
        """
        # ------------------------------------------------------------------
        # 1. Reset
        # ------------------------------------------------------------------
        reset_cfg = (
            self._env_config.full_task_reset_action
            or self._env_config.critical_phase_reset_action
        )
        reset_pose = np.array(reset_cfg, dtype=np.float32) if reset_cfg is not None else None
        obs = self._env.reset(reset_pose)

        max_steps = getattr(
            self._env_config,
            "max_episode_steps",
            self._env_config.chunk_exec_horizon * 50,
        )

        # ------------------------------------------------------------------
        # 2. Episode loop — collect one (step_results, features, source) per chunk
        # ------------------------------------------------------------------
        # Each entry: (step_results, features, source, actor_version)
        chunk_data: list[tuple[list[StepResult], ChunkFeatures, int, int]] = []
        step_id = 0
        done = False

        while not done:
            # a. Query Machine A for current obs
            features = self._extract_chunk_features(obs)

            # b. Select action chunk
            action_chunk, source, actor_version = self._select_action_chunk(
                features, episode_id, step_id, use_actor
            )

            # c. Execute chunk (rate-controlled)
            step_results = self._env.execute_chunk(action_chunk)

            # d. Check done signal after chunk completes
            if done_signal is not None and done_signal():
                done = True

            # Check max_steps ceiling
            if step_id + len(step_results) >= max_steps:
                done = True

            chunk_data.append((step_results, features, source, actor_version))
            step_id += len(step_results)
            obs = step_results[-1].next_obs

        # ------------------------------------------------------------------
        # 3. Query Machine A on terminal obs for last chunk's next_z_rl
        # ------------------------------------------------------------------
        terminal_features = self._extract_chunk_features(obs)

        # ------------------------------------------------------------------
        # 4. Post-processing: build EpisodeStepRecord list with correct next_z_rl
        # ------------------------------------------------------------------
        episode_steps: list[EpisodeStepRecord] = []
        global_step_id = 0

        for k, (step_results, features, source, _) in enumerate(chunk_data):
            is_last_chunk = k == len(chunk_data) - 1

            # next_z_rl for all steps in this chunk = z_rl of the NEXT chunk's obs
            # (or terminal_features for the last chunk)
            next_features = chunk_data[k + 1][1] if not is_last_chunk else terminal_features

            ref_chunk = features.ref_chunk
            ref_len = len(ref_chunk)

            for i, sr in enumerate(step_results):
                # Terminal step: last step of last chunk, and episode is done
                is_terminal = is_last_chunk and (i == len(step_results) - 1) and done

                # Ref action for this step (clamp to last element if chunk is shorter)
                ref_action = ref_chunk[i] if i < ref_len else ref_chunk[-1]

                record = EpisodeStepRecord(
                    z_rl=features.z_rl.copy(),
                    proprio=self._env.obs_to_state(sr.obs),
                    ref_action=ref_action.copy(),
                    action=sr.action.copy(),
                    reward=sr.reward,
                    done=is_terminal,
                    next_z_rl=next_features.z_rl.copy(),
                    next_proprio=self._env.obs_to_state(sr.next_obs),
                    source=source,
                    collection_phase=collection_phase,
                    episode_id=episode_id,
                    step_id=global_step_id,
                )
                episode_steps.append(record)
                global_step_id += 1

        # ------------------------------------------------------------------
        # 5. Build transitions and push to replay
        # ------------------------------------------------------------------
        transitions: list = []
        if not self._eval_actor_only and episode_steps:
            transitions = build_chunk_transitions_from_episode(
                episode_steps,
                chunk_len=self._rl_config.chunk_len,
                stride=self._rl_config.chunk_len,
                allow_partial=True,
            )
            try:
                self._replay_client.add_transitions(transitions)
            except Exception as exc:
                logger.warning(
                    "Failed to add transitions to replay buffer (episode=%d): %s",
                    episode_id,
                    exc,
                )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        summary: dict[str, Any] = {
            "episode_id": episode_id,
            "success": False,
            "n_steps": global_step_id,
            "transitions": len(transitions),
            "collection_phase": collection_phase,
            "use_actor": use_actor,
        }

        logger.info(
            "Episode %d complete | n_steps=%d | transitions=%d | phase=%s | actor=%s",
            episode_id,
            global_step_id,
            len(transitions),
            collection_phase,
            use_actor,
        )

        if self._metrics_path is not None:
            self._append_metrics(summary)

        return summary

    def _append_metrics(self, summary: dict[str, Any]) -> None:
        try:
            metrics_dir = os.path.dirname(self._metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            with open(self._metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as exc:
            logger.warning(
                "Failed to write episode metrics to %s: %s", self._metrics_path, exc
            )


# ---------------------------------------------------------------------------
# RolloutOrchestrator
# ---------------------------------------------------------------------------


class RolloutOrchestrator:
    """Top-level orchestrator: assembles all components and drives the episode loop.

    Wires together:
    - SOArmEnv         (robot hardware)
    - MachineAFeatureClient  (VLA features via WebSocket)
    - ActorClient      (RL actor via HTTP)
    - ReplayClient / NullReplayClient  (transition storage)
    - RolloutPhaseController  (warmup → online scheduling)
    - KeyboardSignalBridge    (operator interrupts)
    - EnvDriver        (single-episode logic)
    """

    def __init__(
        self,
        system_config: OnlineRLSystemConfig,
        robot: Any,  # SOFollower | ThreadSafeRobot
        *,
        run_dir: str,
        eval_actor_only: bool = False,
    ) -> None:
        env_config = system_config.env_driver
        rl_config = system_config.rl

        self._env = SOArmEnv(robot, env_config)

        self._machine_a_client = MachineAFeatureClient(
            ws_url=env_config.machine_a_ws_url,
            connect_timeout_sec=env_config.machine_a_connect_timeout_sec,
            recv_timeout_sec=env_config.machine_a_recv_timeout_sec,
            retry_interval_sec=env_config.machine_a_retry_interval_sec,
        )

        self._actor_client = ActorClient(
            base_url=env_config.actor_service_url,
            timeout_sec=env_config.actor_request_timeout_sec,
        )

        if eval_actor_only:
            self._replay_client: Any = NullReplayClient()
        else:
            self._replay_client = ReplayClient(
                base_url=env_config.replay_service_url,
                timeout_sec=env_config.replay_request_timeout_sec,
            )

        status_path = os.path.join(run_dir, "metrics", "learner_status.json")
        self._phase_controller = RolloutPhaseController(status_path)
        self._keyboard = KeyboardSignalBridge()

        metrics_path = os.path.join(run_dir, "metrics", "episode_results.jsonl")
        self._env_driver = EnvDriver(
            env=self._env,
            machine_a_client=self._machine_a_client,
            actor_client=self._actor_client,
            replay_client=self._replay_client,
            rl_config=rl_config,
            env_config=env_config,
            eval_actor_only=eval_actor_only,
            metrics_path=metrics_path,
        )

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        """Main episode loop; runs until stop_event is set or SIG_QUIT received.

        Per-iteration:
          1. Poll keyboard for pre-episode signals (mode overrides, quit).
          2. Determine use_actor and collection_phase.
          3. Run env_driver.run_episode() with an inline done_signal.
          4. Drain remaining keyboard signals to determine episode success.
          5. Log episode result and increment counter.
        """
        episode_id = 0
        # Keyboard-overridden actor mode; None means defer to phase controller
        force_actor: bool | None = None

        try:
            while True:
                # ----------------------------------------------------------
                # Pre-episode: drain keyboard signals for mode changes / quit
                # ----------------------------------------------------------
                while True:
                    sig = self._keyboard.get_signal()
                    if sig is None:
                        break
                    if sig == SIG_QUIT:
                        logger.info("Quit signal received before episode %d. Stopping.", episode_id)
                        if stop_event is not None:
                            stop_event.set()
                        return
                    elif sig == SIG_SET_ACTOR_MODE:
                        force_actor = True
                        logger.info("Keyboard override: actor mode enabled")
                    elif sig == SIG_SET_BASE_MODE:
                        force_actor = False
                        logger.info("Keyboard override: base mode enabled")
                    elif sig == SIG_TOGGLE_TELEOP:
                        logger.info("Toggle teleop signal (unhandled in this orchestrator)")

                if stop_event is not None and stop_event.is_set():
                    break

                use_actor = (
                    force_actor
                    if force_actor is not None
                    else self._phase_controller.should_use_actor()
                )
                collection_phase = self._phase_controller.current_phase()

                # Track which signal (if any) ended the episode
                _last_stop_signal: list[str | None] = [None]

                def _done_signal() -> bool:
                    s = self._keyboard.get_signal()
                    if s in (
                        SIG_NEXT_EPISODE,
                        SIG_MARK_SUCCESS,
                        SIG_MARK_FAILURE,
                        SIG_QUIT,
                    ):
                        _last_stop_signal[0] = s
                        return True
                    return False

                result = self._env_driver.run_episode(
                    episode_id,
                    use_actor=use_actor,
                    collection_phase=collection_phase,
                    done_signal=_done_signal,
                )

                # ----------------------------------------------------------
                # Post-episode: drain remaining signals for success/failure
                # ----------------------------------------------------------
                while True:
                    s = self._keyboard.get_signal()
                    if s is None:
                        break
                    if s in (SIG_MARK_SUCCESS, SIG_MARK_FAILURE):
                        _last_stop_signal[0] = s
                        break
                    if s == SIG_QUIT:
                        _last_stop_signal[0] = s
                        break

                final_sig = _last_stop_signal[0]

                if final_sig == SIG_QUIT:
                    logger.info("Quit signal received after episode %d. Stopping.", episode_id)
                    if stop_event is not None:
                        stop_event.set()
                    break

                success = final_sig == SIG_MARK_SUCCESS

                logger.info(
                    "Episode %d | n_steps=%d | transitions=%d | phase=%s | actor=%s | success=%s",
                    episode_id,
                    result.get("n_steps", 0),
                    result.get("transitions", 0),
                    collection_phase,
                    use_actor,
                    success,
                )

                episode_id += 1

                if stop_event is not None and stop_event.is_set():
                    break

        finally:
            self._keyboard.close()
