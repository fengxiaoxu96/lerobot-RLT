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

"""Replay buffer, journal, HTTP service, and episode-to-transition utilities.

Four public objects are exported:

    build_chunk_transitions_from_episode — converts a flat EpisodeStepRecord list
        into overlapping RLTTransition chunks.
    build_terminal_aligned_chunk_transition — extracts the last full chunk of a
        done episode (ensures the terminal reward is always included).
    ReplayManager — owns the ring buffer + append-only journal + HTTP server.
        This is the B3 process.
    ReplayClient / NullReplayClient — thin HTTP client and no-op variant.

HTTP transport uses pickle over application/octet-stream.  The reference
(openpi-RLT) uses msgpack_numpy from openpi_client, which is not a lerobot
dependency.  Pickle is equally compact for numpy arrays and requires nothing
beyond the standard library.
"""

from __future__ import annotations

import http
import json
import logging
import os
import pickle
import threading
import time
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from lerobot.rl.rltoken.config import ReplayConfig
from lerobot.rl.rltoken.transition import (
    COLLECTION_PHASE_ONLINE,
    COLLECTION_PHASE_WARMUP,
    EpisodeStepRecord,
    RLTTransition,
    TransitionSource,
    _as_array,
)

logger = logging.getLogger(__name__)

ArrayDict = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Episode → chunk transition builders
# ---------------------------------------------------------------------------


def _pad_stack(
    values: list[np.ndarray],
    start: int,
    length: int,
    elem_shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Stack up to `length` elements starting at `start`, zero-padding the tail."""
    out = np.zeros((length, *elem_shape), dtype=dtype)
    for i in range(length):
        idx = start + i
        if idx >= len(values):
            break
        out[i] = values[idx]
    return out


def _pad_scalar_stack(
    values: list[Any],
    start: int,
    length: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Stack scalars (rewards, source ints) with zero-padding."""
    out = np.zeros((length,), dtype=dtype)
    for i in range(length):
        idx = start + i
        if idx >= len(values):
            break
        out[i] = values[idx]
    return out


def _resolve_chunk_source(steps: list[EpisodeStepRecord]) -> tuple[int, bool]:
    """Determine the dominant TransitionSource and intervention flag for a chunk."""
    intervention = any(step.intervention_flag for step in steps)
    source_set = {int(step.source) for step in steps}
    has_human = int(TransitionSource.HUMAN) in source_set
    has_policy = any(
        s in source_set
        for s in (int(TransitionSource.BASE), int(TransitionSource.RL), int(TransitionSource.MIXED))
    )
    if int(TransitionSource.MIXED) in source_set or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(steps[0].source), intervention


def _build_chunk_transition(
    steps: list[EpisodeStepRecord],
    *,
    start: int,
    chunk_len: int,
) -> RLTTransition:
    """Build one RLTTransition from a window of EpisodeStepRecord objects.

    The current state is taken from steps[start]; the next state is taken from
    the last step in the window (or the last available step if the episode is
    shorter than chunk_len).  Rewards and actions are zero-padded to chunk_len.
    """
    current = steps[start]
    end = min(start + chunk_len, len(steps))
    window = steps[start:end]
    last = window[-1]

    action_shape = steps[0].action.shape
    ref_shape = steps[0].ref_action.shape

    ref_actions = [step.ref_action for step in steps]
    actions = [step.action for step in steps]
    rewards = [step.reward for step in steps]
    sources = [step.source for step in steps]

    ref_chunk = _pad_stack(ref_actions, start, chunk_len, ref_shape, np.float32)
    action_chunk = _pad_stack(actions, start, chunk_len, action_shape, np.float32)
    reward_chunk = _pad_scalar_stack(rewards, start, chunk_len, np.float32)
    next_ref_chunk = _pad_stack(ref_actions, start + chunk_len, chunk_len, ref_shape, np.float32)
    source_chunk = _pad_scalar_stack(sources, start, chunk_len, np.uint8)

    source, intervention = _resolve_chunk_source(window)
    done = any(step.done for step in window)

    return RLTTransition(
        z_rl=current.z_rl,
        proprio=current.proprio,
        ref_chunk=ref_chunk,
        action_chunk=action_chunk,
        rewards=reward_chunk,
        done=done,
        next_z_rl=last.next_z_rl,
        next_proprio=last.next_proprio,
        next_ref_chunk=next_ref_chunk,
        source=source,
        source_chunk=source_chunk,
        collection_phase=current.collection_phase,
        success=int(last.success),
        intervention_flag=intervention,
        episode_id=current.episode_id,
        step_id=current.step_id,
    )


def _coerce_step(step: EpisodeStepRecord | dict[str, Any]) -> EpisodeStepRecord:
    """Accept either an EpisodeStepRecord or a plain dict; return EpisodeStepRecord."""
    if isinstance(step, EpisodeStepRecord) or (
        hasattr(step, "z_rl") and hasattr(step, "action") and hasattr(step, "episode_id")
    ):
        return step  # type: ignore[return-value]
    return EpisodeStepRecord.from_mapping(step)


def build_chunk_transitions_from_episode(
    episode_steps: Iterable[EpisodeStepRecord | dict[str, Any]],
    *,
    chunk_len: int,
    stride: int = 2,
    allow_partial: bool = True,
) -> list[RLTTransition]:
    """Convert a flat list of per-step records into overlapping chunk transitions.

    Args:
        episode_steps: Flat sequence of EpisodeStepRecord (or dicts).
        chunk_len: Number of steps per RLTTransition chunk.
        stride: Step between chunk start positions.  stride=1 gives maximum
            overlap; stride=chunk_len gives non-overlapping chunks.
        allow_partial: If True, include chunks that extend past the end of the
            episode (the tail is zero-padded).

    Returns:
        List of RLTTransition objects, one per stride position.
    """
    steps: list[EpisodeStepRecord] = [_coerce_step(step) for step in episode_steps]
    if not steps:
        return []

    transitions: list[RLTTransition] = []
    for start in range(0, len(steps), stride):
        if not allow_partial and start + chunk_len > len(steps):
            break
        transitions.append(_build_chunk_transition(steps, start=start, chunk_len=chunk_len))
    return transitions


def build_terminal_aligned_chunk_transition(
    episode_steps: Iterable[EpisodeStepRecord | dict[str, Any]],
    *,
    chunk_len: int,
) -> RLTTransition | None:
    """Return the last chunk-len steps as a single transition.

    Guarantees the terminal reward is present in the rewards array.
    Returns None if the episode is shorter than chunk_len or did not end with done=True.
    """
    steps: list[EpisodeStepRecord] = [_coerce_step(step) for step in episode_steps]
    if len(steps) < chunk_len or not steps[-1].done:
        return None
    start = len(steps) - chunk_len
    return _build_chunk_transition(steps, start=start, chunk_len=chunk_len)


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class RLTReplayBuffer:
    """Thread-safe CPU ring buffer for RLTTransition objects.

    Storage is a dict of pre-allocated numpy arrays, lazily initialised on the
    first add() call so that field shapes are inferred from the first transition.
    """

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        sample_strategy: str = "uniform",
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._sample_strategy = sample_strategy
        self._recent_episode_window = int(recent_episode_window)
        self._recent_online_ratio = float(recent_online_ratio)
        self._warmup_demo_ratio = float(warmup_demo_ratio)
        self._human_intervention_ratio = float(human_intervention_ratio)
        self._storage: dict[str, np.ndarray] | None = None
        self._position: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def _init_storage(self, record: ArrayDict) -> None:
        self._storage = {
            key: np.empty((self.capacity, *np.asarray(val).shape), dtype=np.asarray(val).dtype)
            for key, val in record.items()
        }

    def add(self, transition: RLTTransition | dict[str, Any]) -> None:
        record = (
            transition
            if isinstance(transition, RLTTransition) or hasattr(transition, "z_rl")
            else RLTTransition.from_mapping(transition)
        )
        record_np = record.to_numpy()
        with self._lock:
            if self._storage is None:
                self._init_storage(record_np)
            assert self._storage is not None
            for key, val in record_np.items():
                expected = self._storage[key].shape[1:]
                if np.asarray(val).shape != expected:
                    raise ValueError(
                        f"Shape mismatch for field '{key}': got {np.asarray(val).shape}, expected {expected}"
                    )
                self._storage[key][self._position] = val
            self._position = (self._position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def extend(self, transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        for t in transitions:
            self.add(t)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ArrayDict:
        with self._lock:
            if self._storage is None or self._size == 0:
                raise RuntimeError("Cannot sample from an empty replay buffer.")
            n = min(batch_size, self._size)
            if self._sample_strategy == "stratified":
                indices = self._stratified_indices(n)
            else:
                indices = self._uniform_indices(n)
            return {key: arr[indices].copy() for key, arr in self._storage.items()}

    def _uniform_indices(self, n: int) -> np.ndarray:
        return self._rng.integers(0, self._size, size=n)

    def _stratified_indices(self, batch_size: int) -> np.ndarray:
        assert self._storage is not None
        phase = self._storage["collection_phase_id"][: self._size]
        episode_id = self._storage["episode_id"][: self._size]
        source = self._storage["source"][: self._size]
        source_chunk = self._storage["source_chunk"][: self._size]
        intervention = self._storage["intervention_flag"][: self._size]
        all_idx = np.arange(self._size, dtype=np.int64)

        max_ep = int(np.max(episode_id)) if episode_id.size else -1
        recent_start = max_ep - max(self._recent_episode_window, 1) + 1

        recent_pool = np.flatnonzero(
            (phase == COLLECTION_PHASE_ONLINE) & (episode_id >= recent_start)
        )
        warmup_pool = np.flatnonzero(phase == COLLECTION_PHASE_WARMUP)
        human_pool = np.flatnonzero(
            intervention
            | (source == int(TransitionSource.HUMAN))
            | (source == int(TransitionSource.MIXED))
            | np.any(source_chunk == int(TransitionSource.HUMAN), axis=1)
            | np.any(source_chunk == int(TransitionSource.MIXED), axis=1)
        )

        n_recent = int(round(batch_size * self._recent_online_ratio))
        n_warmup = int(round(batch_size * self._warmup_demo_ratio))
        n_human = int(round(batch_size * self._human_intervention_ratio))
        n_uniform = max(batch_size - n_recent - n_warmup - n_human, 0)

        parts = [
            self._draw_from_pool(recent_pool, n_recent),
            self._draw_from_pool(warmup_pool, n_warmup),
            self._draw_from_pool(human_pool, n_human),
            self._draw_from_pool(all_idx, n_uniform),
        ]
        result = np.concatenate([p for p in parts if p.size > 0]) if any(p.size > 0 for p in parts) else np.empty(
            (0,), dtype=np.int64
        )
        if result.size < batch_size:
            result = np.concatenate(
                [result, self._draw_from_pool(all_idx, batch_size - result.size)]
            )
        self._rng.shuffle(result)
        return result[:batch_size]

    def _draw_from_pool(self, pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.empty((0,), dtype=np.int64)
        return self._rng.choice(pool, size=count, replace=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "size": self._size,
                "position": self._position,
                "sample_strategy": self._sample_strategy,
                "recent_episode_window": self._recent_episode_window,
                "recent_online_ratio": self._recent_online_ratio,
                "warmup_demo_ratio": self._warmup_demo_ratio,
                "human_intervention_ratio": self._human_intervention_ratio,
            }


# ---------------------------------------------------------------------------
# Replay manager (B3 process)
# ---------------------------------------------------------------------------


class ReplayManager:
    """Owns the RLTReplayBuffer, an append-only journal, and an HTTP service.

    The journal is a sequential pickle file: each record is one
    `transition.to_journal_record()` dict appended with pickle.dump().
    On startup, the journal is replayed to restore the buffer state after a crash.

    HTTP routes (pickle-over-octet-stream):
        GET  /healthz  → b"OK"
        GET  /stats    → JSON dict
        POST /add      → add one RLTTransition, returns {"ok": True}
        POST /extend   → add list of RLTTransitions, returns {"ok": True}
        POST /sample   → {"batch_size": N} → ArrayDict
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config
        self._buffer = RLTReplayBuffer(
            capacity=config.capacity,
            seed=config.seed,
            sample_strategy=config.sample_strategy,
            recent_episode_window=config.recent_episode_window,
            recent_online_ratio=config.recent_online_ratio,
            warmup_demo_ratio=config.warmup_demo_ratio,
            human_intervention_ratio=config.human_intervention_ratio,
        )
        self._journal_path = config.journal_path
        self._lock = threading.Lock()
        self._adds_total: int = 0
        self._max_episode_id: int = -1
        os.makedirs(os.path.dirname(self._journal_path) or ".", exist_ok=True)
        self._restore_from_journal()
        logger.info(
            "ReplayManager ready  capacity=%d  journal=%s", config.capacity, config.journal_path
        )

    # ------------------------------------------------------------------
    # Mutating API (called by ReplayClient or directly in-process)
    # ------------------------------------------------------------------

    def add_transition(self, transition: RLTTransition | dict[str, Any]) -> None:
        record = (
            transition
            if isinstance(transition, RLTTransition)
            else RLTTransition.from_mapping(transition)
        )
        with self._lock:
            self._buffer.add(record)
            self._journal_append_many([record])
            self._adds_total += 1
            self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
            if self._adds_total <= 5 or self._adds_total % 50 == 0:
                logger.info(
                    "ReplayManager size=%d/%d adds_total=%d",
                    self._buffer._size,
                    self._buffer.capacity,
                    self._adds_total,
                )

    def add_transitions(self, transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        records = [
            t if isinstance(t, RLTTransition) else RLTTransition.from_mapping(t)
            for t in transitions
        ]
        if not records:
            return
        with self._lock:
            for record in records:
                self._buffer.add(record)
                self._adds_total += 1
                self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
            self._journal_append_many(records)
            logger.info(
                "ReplayManager size=%d/%d adds_total=%d batch=%d",
                self._buffer._size,
                self._buffer.capacity,
                self._adds_total,
                len(records),
            )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def sample_batch(self, batch_size: int) -> ArrayDict:
        return self._buffer.sample(batch_size)

    def stats(self) -> dict[str, Any]:
        buf_stats = self._buffer.stats()
        return {
            **buf_stats,
            "adds_total": self._adds_total,
            "journal_path": self._journal_path,
            "max_episode_id": self._max_episode_id,
        }

    # ------------------------------------------------------------------
    # Journal helpers
    # ------------------------------------------------------------------

    def _journal_append_many(self, records: list[RLTTransition]) -> None:
        with open(self._journal_path, "ab") as f:
            for record in records:
                pickle.dump(record.to_journal_record(), f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

    def _restore_from_journal(self) -> None:
        if not os.path.exists(self._journal_path):
            logger.info("No journal found at %s; starting empty.", self._journal_path)
            return
        restored = 0
        with open(self._journal_path, "rb") as f:
            while True:
                try:
                    raw = pickle.load(f)
                except EOFError:
                    break
                record = RLTTransition.from_mapping(raw)
                self._buffer.add(record)
                self._max_episode_id = max(self._max_episode_id, int(record.episode_id))
                restored += 1
        self._adds_total = restored
        logger.info("Restored %d transitions from journal %s", restored, self._journal_path)

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def serve_forever(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Start the blocking HTTP server.  Call from a dedicated thread or process."""
        bind_host = host or self._config.bind_host
        bind_port = port or self._config.port
        manager = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self._send_text(b"OK\n")
                elif self.path == "/stats":
                    payload = json.dumps(manager.stats()).encode()
                    self._send_bytes(payload, content_type="application/json")
                else:
                    self.send_error(http.HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/add":
                    payload = pickle.loads(body)
                    manager.add_transition(payload)
                    self._send_pickle({"ok": True})
                elif self.path == "/extend":
                    payload = pickle.loads(body)
                    manager.add_transitions(payload["transitions"])
                    self._send_pickle({"ok": True})
                elif self.path == "/sample":
                    payload = pickle.loads(body)
                    batch = manager.sample_batch(int(payload["batch_size"]))
                    self._send_pickle(batch)
                else:
                    self.send_error(http.HTTPStatus.NOT_FOUND)

            def log_message(self, _fmt: str, *_args: Any) -> None:
                return  # suppress access logs

            def _send_text(self, data: bytes) -> None:
                self.send_response(http.HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(data)

            def _send_bytes(self, data: bytes, *, content_type: str = "application/octet-stream") -> None:
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_pickle(self, obj: Any) -> None:
                self._send_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        server = ThreadingHTTPServer((bind_host, bind_port), _Handler)
        server.timeout = 0.5
        logger.info("ReplayManager listening on http://%s:%d", bind_host, bind_port)
        try:
            if stop_event is None:
                server.serve_forever()
            else:
                while not stop_event.is_set():
                    server.handle_request()
        finally:
            server.server_close()
            logger.info("ReplayManager HTTP server stopped.")


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class ReplayClient:
    """HTTP client for the ReplayManager service.

    Sends pickle-encoded payloads; decodes pickle responses.  Used by the
    learner (sample_batch) and the robot rollout process (add_transitions).
    """

    def __init__(self, base_url: str, *, timeout_sec: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec

    def add_transition(self, transition: RLTTransition | dict[str, Any]) -> None:
        payload = (
            transition.to_journal_record()
            if isinstance(transition, RLTTransition)
            else transition
        )
        self._post("/add", payload)

    def add_transitions(self, transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        records = [
            t.to_journal_record() if isinstance(t, RLTTransition) else t for t in transitions
        ]
        self._post("/extend", {"transitions": records})

    def sample_batch(self, batch_size: int) -> ArrayDict:
        return self._post("/sample", {"batch_size": batch_size})

    def stats(self) -> dict[str, Any]:
        req = urllib_request.Request(f"{self._base_url}/stats", method="GET")
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                return json.loads(resp.read().decode())
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Replay GET /stats failed: {exc}") from exc

    def healthz(self) -> bool:
        try:
            req = urllib_request.Request(f"{self._base_url}/healthz", method="GET")
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_until_healthy(self, *, timeout_sec: float = 30.0, poll_sec: float = 0.5) -> None:
        """Block until the server responds to /healthz or raise RuntimeError."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.healthz():
                return
            time.sleep(poll_sec)
        raise RuntimeError(f"Replay service at {self._base_url} did not become healthy within {timeout_sec}s")

    def _post(self, path: str, payload: Any) -> Any:
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        req = urllib_request.Request(
            f"{self._base_url}{path}",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                return pickle.loads(resp.read())
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Replay POST {path} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# No-op client for eval-only rollout
# ---------------------------------------------------------------------------


class NullReplayClient:
    """Drop-in replacement for ReplayClient that discards all writes.

    Used in eval-only mode (launch_actor_eval_rlt.py) where transitions
    should not be collected.
    """

    def add_transition(self, _transition: RLTTransition | dict[str, Any]) -> None:
        return

    def add_transitions(self, _transitions: Iterable[RLTTransition | dict[str, Any]]) -> None:
        return

    def sample_batch(self, _batch_size: int) -> ArrayDict:
        raise RuntimeError("NullReplayClient cannot sample — no buffer exists.")

    def stats(self) -> dict[str, Any]:
        return {
            "capacity": 0,
            "size": 0,
            "position": 0,
            "adds_total": 0,
            "journal_path": None,
            "max_episode_id": -1,
        }

    def healthz(self) -> bool:
        return True

    def wait_until_healthy(self, **_kwargs: Any) -> None:
        return
