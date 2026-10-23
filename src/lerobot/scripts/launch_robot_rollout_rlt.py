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

"""Launch the SO-ARM robot rollout for the RLToken online RL pipeline.

Waits for both the Actor Service (Machine B) and the VLA server (Machine A)
to become healthy, then starts the RolloutOrchestrator.

Usage::

    python launch_robot_rollout_rlt.py \\
        --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml

    # With explicit service URLs:
    python launch_robot_rollout_rlt.py \\
        --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml \\
        --actor-url http://192.168.1.10:9101 \\
        --replay-url http://192.168.1.10:9102 \\
        --machine-a-ws-url ws://192.168.1.20:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import requests

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 2.0
_HEALTH_TIMEOUT = 300.0  # 5 minutes max to wait for services


def _wait_http_healthy(
    url: str, name: str, timeout: float = _HEALTH_TIMEOUT
) -> None:
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=3.0)
            if r.status_code == 200:
                logger.info("%s is healthy at %s", name, url)
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:
            last_err = str(exc)
        logger.info("Waiting for %s (%s)… %s", name, url, last_err)
        time.sleep(_HEALTH_POLL_INTERVAL)
    raise TimeoutError(f"{name} at {url} did not become healthy within {timeout}s")


def _wait_ws_healthy(ws_url: str, timeout: float = _HEALTH_TIMEOUT) -> None:
    # Derive the HTTP healthz URL from the ws:// URL
    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    if not http_url.endswith("/healthz"):
        http_url = http_url.rstrip("/") + "/healthz"
    _wait_http_healthy(http_url, "Machine A VLA server", timeout=timeout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RLToken robot rollout.")
    parser.add_argument("--config", required=True, help="Path to OnlineRLSystemConfig YAML.")
    parser.add_argument("--actor-url", default=None, help="Override actor service URL.")
    parser.add_argument("--replay-url", default=None, help="Override replay service URL.")
    parser.add_argument("--machine-a-ws-url", default=None, help="Override Machine A WS URL.")
    parser.add_argument("--run-dir", default=None, help="Override run directory.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip health-check wait and start rollout immediately.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()

    from lerobot.rl.rltoken.config import load_system_config_yaml

    cfg = load_system_config_yaml(args.config)
    env_cfg = cfg.env_driver

    # Apply CLI overrides
    actor_url = args.actor_url or env_cfg.actor_service_url
    replay_url = args.replay_url or env_cfg.replay_service_url
    machine_a_ws_url = args.machine_a_ws_url or env_cfg.machine_a_ws_url

    run_dir = args.run_dir
    if run_dir is None:
        import os
        run_dir = os.path.dirname(os.path.abspath(cfg.learner_service.checkpoint_dir))

    if not args.no_wait:
        logger.info("Waiting for services to become healthy…")
        _wait_http_healthy(f"{actor_url}/healthz", "ActorService")
        _wait_http_healthy(f"{replay_url}/healthz", "ReplayManager")
        _wait_ws_healthy(machine_a_ws_url)
        logger.info("All services healthy. Starting rollout.")
    else:
        logger.info("Skipping health-check wait (--no-wait).")

    from lerobot.rl.rltoken.rollout import RolloutOrchestrator

    orchestrator = RolloutOrchestrator(
        rl_config=cfg.rl,
        env_driver_config=env_cfg,
        run_dir=run_dir,
    )
    try:
        orchestrator.run_forever()
    except KeyboardInterrupt:
        logger.info("Rollout stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
