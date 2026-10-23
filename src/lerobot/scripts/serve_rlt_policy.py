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

"""Machine A WebSocket server — serves VLA features (z_rl, ref_chunk, proprio).

Loads a trained RLTokenPolicy checkpoint and answers inference requests from
the SO-ARM rollout process (MachineAFeatureClient in inference.py).

Protocol (pickle over WebSocket):
  Client → server: {"images": {cam_key: HxWxC uint8 numpy}, "state": [state_dim,]}
  Server → client: {"z_rl": [z_dim,], "ref_chunk": [chunk_len, action_dim],
                    "proprio": [proprio_dim,], "infer_ms": float}

Batch protocol:
  Client → server: {"batch": [obs1, obs2, ...]}
  Server → client: {"batch_results": [res1, res2, ...], "batch_size": N}

On connect, the server sends a metadata handshake packet (pickle dict).

Usage::

    python serve_rlt_policy.py \\
        --checkpoint-path /path/to/rlt_checkpoint \\
        --chunk-len 10 \\
        --action-dim 6 \\
        --proprio-dim 6 \\
        --z-dim 2048 \\
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import pickle
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8000
DEFAULT_CHUNK_LEN = 10
DEFAULT_ACTION_DIM = 6
DEFAULT_PROPRIO_DIM = 6
DEFAULT_Z_DIM = 2048


# ---------------------------------------------------------------------------
# Batch preprocessing helpers
# ---------------------------------------------------------------------------


def _prepare_batch(
    obs: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a raw observation dict to a model-ready batch dict.

    Image arrays (HxWxC uint8) are converted to [1, C, H, W] float32 [0, 1].
    State is converted to [1, state_dim] float32.
    Both are placed on `device`.
    """
    batch: dict[str, torch.Tensor] = {}
    for cam_key, img in obs.get("images", {}).items():
        img_np = np.asarray(img)
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        else:
            img_np = img_np.astype(np.float32)
        if img_np.ndim == 3:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        else:
            img_t = torch.from_numpy(img_np).unsqueeze(0)
        batch[f"observation.images.{cam_key}"] = img_t.to(device)

    state = np.asarray(obs.get("state", []), dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    if "prompt" in obs:
        batch["observation.language_instruction"] = [obs["prompt"]]

    return batch


def _prepare_batch_stacked(
    obs_list: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack a list of observations into a single batched dict."""
    single_batches = [_prepare_batch(obs, device=device) for obs in obs_list]
    stacked: dict[str, torch.Tensor] = {}
    for key in single_batches[0]:
        if key == "observation.language_instruction":
            stacked[key] = [b[key][0] for b in single_batches]
        else:
            stacked[key] = torch.cat([b[key] for b in single_batches], dim=0)
    return stacked


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------


class RLTInferenceServer:
    """Wraps a loaded RLTokenPolicy and provides feature extraction for Machine A.

    Features returned per observation:
        z_rl      — [z_dim] float32, flattened RL state token
        ref_chunk — [chunk_len, action_dim] float32, VLA reference actions
        proprio   — [proprio_dim] float32, robot joint positions
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        chunk_len: int = DEFAULT_CHUNK_LEN,
        action_dim: int = DEFAULT_ACTION_DIM,
        proprio_dim: int = DEFAULT_PROPRIO_DIM,
        z_dim: int = DEFAULT_Z_DIM,
        device: str | None = None,
    ) -> None:
        self._chunk_len = chunk_len
        self._action_dim = action_dim
        self._proprio_dim = proprio_dim
        self._z_dim = z_dim

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Loading RLTokenPolicy from %s on device %s", checkpoint_path, self._device)

        from lerobot.policies.rltoken.modeling_rltoken import RLTokenPolicy

        self._policy = RLTokenPolicy.from_pretrained(checkpoint_path).to(self._device)
        self._policy.eval()
        logger.info("RLTokenPolicy loaded successfully")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Extract z_rl, ref_chunk, and proprio for one observation."""
        t0 = time.monotonic()
        batch = _prepare_batch(obs, device=self._device)
        z_rl, ref_chunk = self._forward_once(batch)
        state = np.asarray(obs.get("state", []), dtype=np.float32)
        proprio = state[: self._proprio_dim].astype(np.float32)
        infer_ms = (time.monotonic() - t0) * 1000.0
        return {
            "z_rl": z_rl,
            "ref_chunk": ref_chunk,
            "proprio": proprio,
            "infer_ms": infer_ms,
        }

    @torch.no_grad()
    def extract_features_batch(self, obs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch feature extraction — one model forward pass for the whole batch."""
        t0 = time.monotonic()
        batch = _prepare_batch_stacked(obs_list, device=self._device)
        z_rl_batch, ref_chunk_batch = self._forward_batch(batch, batch_size=len(obs_list))
        infer_ms = (time.monotonic() - t0) * 1000.0

        results = []
        for i, obs in enumerate(obs_list):
            state = np.asarray(obs.get("state", []), dtype=np.float32)
            proprio = state[: self._proprio_dim].astype(np.float32)
            results.append({
                "z_rl": z_rl_batch[i],
                "ref_chunk": ref_chunk_batch[i],
                "proprio": proprio,
                "infer_ms": infer_ms / len(obs_list),
            })
        return results

    def _forward_once(self, batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """Run both policy.encode() and policy.predict_action_chunk() on a single-element batch."""
        z_rl_t = self._policy.encode(batch)           # [1, num_tokens, embed_dim]
        ref_t = self._policy.predict_action_chunk(batch)  # [1, action_horizon, action_dim]
        z_rl = z_rl_t[0].float().cpu().numpy().reshape(-1)
        ref_chunk = ref_t[0, : self._chunk_len, : self._action_dim].float().cpu().numpy()
        return z_rl.astype(np.float32), ref_chunk.astype(np.float32)

    def _forward_batch(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        z_rl_t = self._policy.encode(batch)           # [B, num_tokens, embed_dim]
        ref_t = self._policy.predict_action_chunk(batch)  # [B, action_horizon, action_dim]
        z_rls = [
            z_rl_t[i].float().cpu().numpy().reshape(-1).astype(np.float32)
            for i in range(batch_size)
        ]
        ref_chunks = [
            ref_t[i, : self._chunk_len, : self._action_dim].float().cpu().numpy().astype(np.float32)
            for i in range(batch_size)
        ]
        return z_rls, ref_chunks

    def warmup(self, n_cameras: int = 1, image_hw: tuple[int, int] = (224, 224)) -> None:
        """Run one dummy forward pass to trigger JIT/GPU graph compilation."""
        logger.info("Warming up RLT inference server…")
        H, W = image_hw
        fake_images = {f"cam{i}": np.zeros((H, W, 3), dtype=np.uint8) for i in range(n_cameras)}
        fake_state = np.zeros(self._proprio_dim, dtype=np.float32)
        fake_obs = {"images": fake_images, "state": fake_state}
        self.extract_features(fake_obs)
        logger.info("Warmup complete (z_dim=%d, chunk_len=%d, action_dim=%d)",
                    self._z_dim, self._chunk_len, self._action_dim)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "z_dim": self._z_dim,
            "chunk_len": self._chunk_len,
            "action_dim": self._action_dim,
            "proprio_dim": self._proprio_dim,
            "device": str(self._device),
            "supports_batch": True,
        }


# ---------------------------------------------------------------------------
# HTTP healthz sidecar (used by MachineAFeatureClient._wait_for_server)
# ---------------------------------------------------------------------------


def _start_healthz_server(port: int) -> ThreadingHTTPServer:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self.send_response(http.HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(b"OK\n")
            else:
                self.send_error(http.HTTPStatus.NOT_FOUND)

        def log_message(self, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def _ws_handler(
    websocket: Any,
    inference_server: RLTInferenceServer,
) -> None:
    """Handle one WebSocket client connection."""
    # Send metadata handshake
    meta_bytes = pickle.dumps(inference_server.metadata, protocol=pickle.HIGHEST_PROTOCOL)
    await websocket.send(meta_bytes)

    async for raw in websocket:
        try:
            obs = pickle.loads(raw)
        except Exception as exc:
            await websocket.send(f"ERROR: failed to decode request: {exc}")
            continue

        try:
            if isinstance(obs, dict) and "batch" in obs:
                results = inference_server.extract_features_batch(obs["batch"])
                response = {"batch_results": results, "batch_size": len(results)}
            else:
                response = inference_server.extract_features(obs)
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            await websocket.send(f"ERROR: inference failed: {exc}")
            continue

        resp_bytes = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        await websocket.send(resp_bytes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve RLToken VLA features over WebSocket.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to RLTokenPolicy checkpoint dir")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="WebSocket server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--healthz-port", type=int, default=None,
                        help="HTTP healthz port (default: ws_port)")
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--proprio-dim", type=int, default=DEFAULT_PROPRIO_DIM)
    parser.add_argument("--z-dim", type=int, default=DEFAULT_Z_DIM)
    parser.add_argument("--device", default=None, help="torch device override (cpu / cuda)")
    return parser.parse_args()


async def _serve(args: argparse.Namespace) -> None:
    import websockets  # type: ignore[import]

    inference_server = RLTInferenceServer(
        args.checkpoint_path,
        chunk_len=args.chunk_len,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        z_dim=args.z_dim,
        device=args.device,
    )
    inference_server.warmup()

    healthz_port = args.healthz_port if args.healthz_port is not None else args.port
    _start_healthz_server(healthz_port)
    logger.info("Healthz HTTP server listening on port %d", healthz_port)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(
        "RLT WebSocket server listening on ws://%s:%d (host=%s, ip=%s)",
        args.host, args.port, hostname, local_ip,
    )

    async def _handler(ws: Any) -> None:
        await _ws_handler(ws, inference_server)

    async with websockets.serve(_handler, args.host, args.port, max_size=None, compression=None):
        await asyncio.Future()  # run forever


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        logger.info("serve_rlt_policy stopped by KeyboardInterrupt.")


if __name__ == "__main__":
    main()
