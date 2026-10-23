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

"""Eval-only rollout for a trained RLToken actor (no learner or replay).

Loads an actor snapshot and starts the RolloutOrchestrator with the actor
service set to deterministic mode and no data collection.  Useful for:
  - Evaluating a saved checkpoint after training.
  - Interactive demos at a controlled frequency.
  - Regression tests without spinning up Machine B services.

The actor is served locally via ActorService (running in a background thread).
Machine A (VLA inference) still needs to be running.

Usage::

    python launch_actor_eval_rlt.py \\
        --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml \\
        --snapshot /path/to/actor_snapshot.pkl \\
        --machine-a-ws-url ws://192.168.1.20:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only RLToken actor rollout.")
    parser.add_argument("--config", required=True, help="Path to OnlineRLSystemConfig YAML.")
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to actor_snapshot.pkl to evaluate.",
    )
    parser.add_argument("--machine-a-ws-url", default=None, help="Override Machine A WS URL.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic (mean) actor actions (default: True).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic actor actions.",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        default=True,
        help="Do not push transitions to replay (eval only, default: True).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()

    from lerobot.rl.rltoken.config import load_system_config_yaml
    from lerobot.rl.rltoken.inference import ActorService

    cfg = load_system_config_yaml(args.config)

    # Override snapshot path with the specified checkpoint
    import dataclasses
    actor_service_cfg = dataclasses.replace(
        cfg.actor_service,
        snapshot_path=args.snapshot,
    )

    # Start ActorService in a daemon thread (no learner writes new snapshots)
    logger.info("Starting local ActorService for eval from snapshot: %s", args.snapshot)
    actor_service = ActorService(cfg.rl, actor_service_cfg)
    t = threading.Thread(target=actor_service.serve_forever, daemon=True)
    t.start()

    # Wait until the actor is healthy
    import requests
    actor_url = f"http://{actor_service_cfg.bind_host}:{actor_service_cfg.port}"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{actor_url}/healthz", timeout=2.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        logger.error("ActorService did not become healthy within 30s.")
        sys.exit(1)

    logger.info("ActorService healthy at %s", actor_url)

    # Apply CLI overrides to env driver config
    env_cfg = cfg.env_driver
    if args.machine_a_ws_url:
        env_cfg = dataclasses.replace(env_cfg, machine_a_ws_url=args.machine_a_ws_url)
    env_cfg = dataclasses.replace(
        env_cfg,
        actor_service_url=actor_url,
        actor_deterministic=args.deterministic,
    )

    import os
    run_dir = "/tmp/rlt_eval_run"
    os.makedirs(run_dir, exist_ok=True)

    from lerobot.rl.rltoken.rollout import RolloutOrchestrator

    orchestrator = RolloutOrchestrator(
        rl_config=cfg.rl,
        env_driver_config=env_cfg,
        run_dir=run_dir,
        eval_only=True,
    )
    try:
        orchestrator.run_forever()
    except KeyboardInterrupt:
        logger.info("Eval rollout stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
