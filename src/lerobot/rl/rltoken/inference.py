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

"""ActorService (B1 process), ActorClient, and MachineAFeatureClient for the RLToken pipeline.

Public API:
    ActorRequest          — typed request dataclass.
    ActorResponse         — typed response dataclass.
    RefinementResult      — result of maybe_refine_chunk().
    ActorService          — HTTP inference service backed by ChunkActor; polls snapshot file.
    ActorClient           — HTTP client for ActorService.
    maybe_refine_chunk    — convenience wrapper with error fallback.
    MachineAFeatureClient — WebSocket client for Machine A (z_rl / ref_chunk features).

Transport: pickle over application/octet-stream (HTTP) and raw bytes (WebSocket).
"""

from __future__ import annotations

import dataclasses
import http
import logging
import os
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import torch

from lerobot.rl.rltoken.action_representation import ActionRepresentationAdapter
from lerobot.rl.rltoken.config import ActorServiceConfig, RLTOnlineRLConfig
from lerobot.rl.rltoken.networks import ChunkActor
from lerobot.rl.rltoken.transition import TransitionSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class ActorRequest:
    """Single inference request sent to ActorService."""

    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    request_id: str
    episode_id: int
    step_id: int
    deterministic: bool = False
    timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "z_rl": np.asarray(self.z_rl, dtype=np.float32),
            "proprio": np.asarray(self.proprio, dtype=np.float32),
            "ref_chunk": np.asarray(self.ref_chunk, dtype=np.float32),
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "deterministic": self.deterministic,
            "timestamp": self.timestamp,
        }


@dataclasses.dataclass(slots=True)
class ActorResponse:
    """Inference response from ActorService."""

    refined_chunk: np.ndarray
    actor_param_version: int
    request_id: str
    source: int = int(TransitionSource.RL)
    timestamp: float | None = None


@dataclasses.dataclass(slots=True)
class RefinementResult:
    """Outcome of maybe_refine_chunk()."""

    refined_chunk: np.ndarray
    source: int
    actor_param_version: int
    used_fallback: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _healthz_url_from_ws_url(ws_url: str) -> str:
    """Derive the HTTP healthz URL from a WebSocket URL (ws:// → http://, wss:// → https://)."""
    parsed = urllib_parse.urlsplit(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urllib_parse.urlunsplit((scheme, parsed.netloc, "/healthz", "", ""))


# ---------------------------------------------------------------------------
# ActorService (B1 process)
# ---------------------------------------------------------------------------


class ActorService:
    """B1 process: ChunkActor HTTP inference server with background snapshot polling.

    The actor always runs in eval mode with torch.no_grad().
    A background thread polls the snapshot file every pull_params_interval_sec
    and hot-swaps actor weights when a newer version is available.
    All accesses to actor weights are serialised through a single threading.Lock.
    """

    def __init__(self, rl_config: RLTOnlineRLConfig, service_config: ActorServiceConfig) -> None:
        self._rl_config = rl_config
        self._service_config = service_config

        # Actor network in eval mode (inference only)
        self._actor = ChunkActor(rl_config)
        self._actor.eval()

        # Action representation adapter (may be None)
        self._adapter: ActionRepresentationAdapter | None = ActionRepresentationAdapter.from_config(
            rl_config
        )

        # Snapshot state — protected by _lock
        self._actor_version: int = -1
        self._params_loaded: bool = False
        self._lock = threading.Lock()

        # Background poller lifecycle
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._start_param_poller()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, request: ActorRequest) -> ActorResponse:
        """Thread-safe single-request inference.

        If no snapshot has been loaded yet, returns ref_chunk as-is with source=BASE.
        Otherwise normalizes ref_chunk (if adapter), runs the actor forward pass,
        denormalizes the output, and returns source=RL.
        """
        # Normalize outside the lock — adapter is frozen/immutable
        model_ref_chunk = np.asarray(request.ref_chunk, dtype=np.float32)
        if self._adapter is not None:
            model_ref_chunk = self._adapter.normalize_ref_chunk(model_ref_chunk, request.proprio)

        with self._lock:
            if not self._params_loaded:
                return ActorResponse(
                    refined_chunk=np.asarray(request.ref_chunk, dtype=np.float32),
                    actor_param_version=self._actor_version,
                    request_id=request.request_id,
                    source=int(TransitionSource.BASE),
                    timestamp=time.time(),
                )
            with torch.no_grad():
                z_t = torch.from_numpy(np.asarray(request.z_rl, dtype=np.float32)).unsqueeze(0)
                p_t = torch.from_numpy(np.asarray(request.proprio, dtype=np.float32)).unsqueeze(0)
                r_t = torch.from_numpy(model_ref_chunk).unsqueeze(0)
                chunk_t, _ = self._actor.sample(z_t, p_t, r_t, deterministic=request.deterministic)
                chunk_np = chunk_t.squeeze(0).cpu().numpy().astype(np.float32)
            actor_version = self._actor_version

        # Denormalize outside the lock — adapter is stateless
        if self._adapter is not None:
            chunk_np = self._adapter.denormalize_to_abs_chunk(chunk_np, request.proprio)

        return ActorResponse(
            refined_chunk=chunk_np.astype(np.float32),
            actor_param_version=actor_version,
            request_id=request.request_id,
            source=int(TransitionSource.RL),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Snapshot polling
    # ------------------------------------------------------------------

    @property
    def actor_param_version(self) -> int:
        with self._lock:
            return self._actor_version

    def _start_param_poller(self) -> None:
        self._poll_thread = threading.Thread(target=self._poll_snapshot_loop, daemon=True)
        self._poll_thread.start()

    def _poll_snapshot_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._try_reload_snapshot()
            except Exception as exc:
                logger.warning("Snapshot reload error: %s", exc)
            self._stop_event.wait(self._service_config.pull_params_interval_sec)

    def _try_reload_snapshot(self) -> None:
        snapshot_path = self._service_config.snapshot_path
        if not os.path.exists(snapshot_path):
            return
        try:
            with open(snapshot_path, "rb") as f:
                payload = pickle.load(f)
        except (EOFError, FileNotFoundError, pickle.UnpicklingError) as exc:
            logger.debug("Could not read snapshot file: %s", exc)
            return

        version = int(payload.get("version", -1))

        # Check without the lock first (cheap path)
        with self._lock:
            if version <= self._actor_version:
                return

        # Load and warm up under the lock so inference is not racing against a partial load
        state_dict = payload["actor_state_dict"]
        with self._lock:
            if version <= self._actor_version:
                return
            self._actor.load_state_dict(state_dict)
            self._warmup_inference()
            self._actor_version = version
            self._params_loaded = True
        logger.info("Loaded actor snapshot version=%s", version)

    def _warmup_inference(self) -> None:
        """Run one dummy forward pass to warm up any lazy initialisation."""
        with torch.no_grad():
            z = torch.zeros(1, self._rl_config.z_dim)
            p = torch.zeros(1, self._rl_config.proprio_dim)
            r = torch.zeros(1, self._rl_config.chunk_len, self._rl_config.action_dim)
            self._actor.sample(z, p, r, deterministic=True)

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def serve_forever(self, stop_event: threading.Event | None = None) -> None:
        """Start the blocking HTTP server.

        Routes:
            GET  /healthz → b"OK\\n"
            GET  /version → pickle {"actor_param_version": N}
            POST /infer   → pickle ActorResponse fields
        """
        service = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self.send_response(http.HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(b"OK\n")
                elif self.path == "/version":
                    body = pickle.dumps(
                        {"actor_param_version": service.actor_param_version},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    self._send_pickle_body(body)
                else:
                    self.send_error(http.HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/infer":
                    self.send_error(http.HTTPStatus.NOT_FOUND)
                    return
                content_len = int(self.headers.get("Content-Length", "0"))
                payload = pickle.loads(self.rfile.read(content_len))
                response = service.infer(
                    ActorRequest(
                        z_rl=np.asarray(payload["z_rl"], dtype=np.float32),
                        proprio=np.asarray(payload["proprio"], dtype=np.float32),
                        ref_chunk=np.asarray(payload["ref_chunk"], dtype=np.float32),
                        request_id=str(payload["request_id"]),
                        episode_id=int(payload["episode_id"]),
                        step_id=int(payload["step_id"]),
                        deterministic=bool(payload.get("deterministic", False)),
                        timestamp=payload.get("timestamp"),
                    )
                )
                body = pickle.dumps(
                    {
                        "refined_chunk": response.refined_chunk,
                        "actor_param_version": response.actor_param_version,
                        "request_id": response.request_id,
                        "timestamp": response.timestamp,
                        "source": response.source,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                try:
                    self._send_pickle_body(body)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("Actor client disconnected before inference response was written.")

            def _send_pickle_body(self, body: bytes) -> None:
                self.send_response(http.HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _fmt: str, *_args: Any) -> None:
                return  # suppress per-request access logs

        server = ThreadingHTTPServer(
            (self._service_config.bind_host, self._service_config.port), _Handler
        )
        server.timeout = 0.5
        logger.info(
            "ActorService listening on http://%s:%d",
            self._service_config.bind_host,
            self._service_config.port,
        )
        try:
            if stop_event is None:
                server.serve_forever()
            else:
                while not stop_event.is_set():
                    server.handle_request()
        finally:
            self._stop_event.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=1.0)
            server.server_close()
            logger.info("ActorService stopped.")


# ---------------------------------------------------------------------------
# ActorClient
# ---------------------------------------------------------------------------


class ActorClient:
    """HTTP client for ActorService.

    Uses pickle transport to match ActorService.
    """

    def __init__(self, base_url: str, *, timeout_sec: float = 1.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec

    def infer(self, request: ActorRequest) -> ActorResponse:
        body = pickle.dumps(request.to_payload(), protocol=pickle.HIGHEST_PROTOCOL)
        req = urllib_request.Request(
            f"{self._base_url}/infer",
            method="POST",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                payload = pickle.loads(resp.read())
            return ActorResponse(
                refined_chunk=np.asarray(payload["refined_chunk"], dtype=np.float32),
                actor_param_version=int(payload["actor_param_version"]),
                request_id=str(payload["request_id"]),
                source=int(payload.get("source", int(TransitionSource.RL))),
                timestamp=payload.get("timestamp"),
            )
        except (urllib_error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"ActorClient /infer failed: {exc}") from exc

    def get_version(self) -> int:
        req = urllib_request.Request(f"{self._base_url}/version", method="GET")
        try:
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                payload = pickle.loads(resp.read())
            return int(payload["actor_param_version"])
        except (urllib_error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"ActorClient /version failed: {exc}") from exc

    def healthz(self) -> bool:
        try:
            req = urllib_request.Request(f"{self._base_url}/healthz", method="GET")
            with urllib_request.urlopen(req, timeout=self._timeout_sec) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_until_healthy(self, timeout_sec: float = 30.0, poll_sec: float = 0.5) -> None:
        """Block until /healthz returns 200 or raise RuntimeError on timeout."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.healthz():
                return
            time.sleep(poll_sec)
        raise RuntimeError(
            f"ActorService at {self._base_url} did not become healthy within {timeout_sec}s"
        )


# ---------------------------------------------------------------------------
# maybe_refine_chunk
# ---------------------------------------------------------------------------


def maybe_refine_chunk(
    actor_client: ActorClient,
    *,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    ref_chunk: np.ndarray,
    request_id: str,
    episode_id: int,
    step_id: int,
    deterministic: bool = False,
    on_error_fallback: bool = True,
) -> RefinementResult:
    """Call actor_client.infer() and return a RefinementResult.

    Args:
        on_error_fallback: If True, return ref_chunk with source=BASE on error.
                           If False, re-raise the RuntimeError.
    """
    try:
        response = actor_client.infer(
            ActorRequest(
                z_rl=z_rl,
                proprio=proprio,
                ref_chunk=ref_chunk,
                request_id=request_id,
                episode_id=episode_id,
                step_id=step_id,
                deterministic=deterministic,
                timestamp=time.time(),
            )
        )
        return RefinementResult(
            refined_chunk=response.refined_chunk,
            source=int(response.source),
            actor_param_version=response.actor_param_version,
            used_fallback=False,
        )
    except RuntimeError as exc:
        if not on_error_fallback:
            raise
        return RefinementResult(
            refined_chunk=np.asarray(ref_chunk, dtype=np.float32),
            source=int(TransitionSource.BASE),
            actor_param_version=-1,
            used_fallback=True,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# MachineAFeatureClient
# ---------------------------------------------------------------------------


class MachineAFeatureClient:
    """WebSocket client to Machine A for z_rl / ref_chunk / proprio features.

    Uses pickle for both request and response payloads.
    Automatically reconnects after send/recv errors.
    """

    def __init__(
        self,
        ws_url: str,
        connect_timeout_sec: float = 10.0,
        recv_timeout_sec: float = 5.0,
        retry_interval_sec: float = 0.5,
    ) -> None:
        self._ws_url = ws_url
        self._healthz_url = _healthz_url_from_ws_url(ws_url)
        self._connect_timeout_sec = connect_timeout_sec
        self._recv_timeout_sec = recv_timeout_sec
        self._retry_interval_sec = retry_interval_sec
        # Connect on construction — blocks until server is healthy
        self._ws = self._wait_for_server()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send a single observation, return feature dict."""
        return self._infer(observation)

    def get_features_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch feature request: send many observations in one round-trip.

        Sends {"batch": [obs1, obs2, ...]}, expects {"batch_results": [r1, r2, ...]}.
        Falls back to sequential get_features() if the batch request fails.
        """
        if not observations:
            return []
        try:
            response = self._infer({"batch": observations})
            if isinstance(response, dict) and "batch_results" in response:
                results = response["batch_results"]
                if len(results) != len(observations):
                    raise ValueError(
                        f"Batch response length mismatch: sent {len(observations)}, got {len(results)}"
                    )
                return list(results)
            raise ValueError(f"Unexpected batch response format: {type(response)}")
        except Exception as exc:
            logger.warning("Batch feature request failed (%s); falling back to sequential.", exc)
            return [self.get_features(obs) for obs in observations]

    def close(self) -> None:
        """Close the WebSocket connection."""
        try:
            self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> Any:
        import websockets.sync.client

        return websockets.sync.client.connect(
            self._ws_url,
            compression=None,
            max_size=None,
            open_timeout=self._connect_timeout_sec,
        )

    def _wait_for_server(self) -> Any:
        """Poll healthz URL and open WebSocket; block until successful."""
        logger.debug("Waiting for Machine A feature server at %s", self._ws_url)
        while True:
            ws = None
            try:
                # Step 1: HTTP health check
                req = urllib_request.Request(self._healthz_url, method="GET")
                with urllib_request.urlopen(req, timeout=self._connect_timeout_sec) as resp:
                    if resp.status != http.HTTPStatus.OK:
                        raise RuntimeError(f"unexpected healthz status={resp.status}")
                # Step 2: Open WebSocket and receive metadata handshake
                ws = self._connect()
                meta_bytes = ws.recv(timeout=self._recv_timeout_sec)
                metadata = pickle.loads(meta_bytes)
                if not isinstance(metadata, dict):
                    raise RuntimeError(
                        f"Machine A metadata must be a mapping, got {type(metadata).__name__}."
                    )
                logger.debug("Connected to Machine A at %s", self._ws_url)
                return ws
            except Exception as exc:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                logger.warning("Machine A not ready at %s: %s", self._ws_url, exc)
                time.sleep(self._retry_interval_sec)

    def _reconnect(self) -> None:
        """Close current connection and wait for a new one."""
        self.close()
        self._ws = self._wait_for_server()

    def _infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send a request and receive a response; reconnects once on failure."""
        import websockets.exceptions as websocket_exceptions

        payload = pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL)
        response: bytes | str | None = None
        for attempt in range(2):
            try:
                self._ws.send(payload)
                response = self._ws.recv(timeout=self._recv_timeout_sec)
                break
            except (OSError, TimeoutError, websocket_exceptions.ConnectionClosed) as exc:
                if attempt == 1:
                    raise RuntimeError(f"Machine A feature request failed: {exc}") from exc
                logger.warning(
                    "Machine A WebSocket failed; reconnecting to %s: %s", self._ws_url, exc
                )
                self._reconnect()

        if isinstance(response, str):
            raise RuntimeError(f"Machine A feature service error:\n{response}")
        unpacked = pickle.loads(response)  # type: ignore[arg-type]
        if not isinstance(unpacked, dict):
            raise RuntimeError(
                f"Machine A feature response must be a mapping, got {type(unpacked).__name__}."
            )
        return unpacked
