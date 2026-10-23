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

"""Role dispatcher for Machine B RLToken services.

Called by launch_machine_b_rlt.py as a subprocess; each process runs exactly
one service role.

Usage::

    python run_rlt_role.py --role replay_manager --config /path/to/config.yaml
    python run_rlt_role.py --role learner_service --config /path/to/config.yaml
    python run_rlt_role.py --role actor_service  --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single RLToken Machine B service role.")
    parser.add_argument(
        "--role",
        required=True,
        choices=["replay_manager", "learner_service", "actor_service"],
        help="Which service to run in this process.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to OnlineRLSystemConfig YAML.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override for the run directory (default: derived from config checkpoint_dir).",
    )
    return parser.parse_args()


def _run_replay_manager(config_path: str) -> None:
    from lerobot.rl.rltoken.config import load_system_config_yaml
    from lerobot.rl.rltoken.replay import ReplayManager

    cfg = load_system_config_yaml(config_path)
    logger.info("Starting ReplayManager on %s:%d", cfg.replay.bind_host, cfg.replay.port)
    manager = ReplayManager(cfg.replay)
    manager.serve_forever()


def _run_learner_service(config_path: str, run_dir: str | None) -> None:
    from lerobot.rl.rltoken.config import load_system_config_yaml
    from lerobot.rl.rltoken.replay import ReplayClient
    from lerobot.rl.rltoken.trainer import LearnerService

    cfg = load_system_config_yaml(config_path)

    # Derive run_dir from checkpoint_dir when not explicitly provided.
    if run_dir is None:
        run_dir = os.path.dirname(os.path.abspath(cfg.learner_service.checkpoint_dir))
    os.makedirs(run_dir, exist_ok=True)

    replay_client = ReplayClient(cfg.learner_service.replay_url)
    logger.info("Starting LearnerService (run_dir=%s)", run_dir)
    learner = LearnerService(cfg.rl, cfg.learner_service, replay_client, run_dir=run_dir)
    learner.run_forever()


def _run_actor_service(config_path: str) -> None:
    from lerobot.rl.rltoken.config import load_system_config_yaml
    from lerobot.rl.rltoken.inference import ActorService

    cfg = load_system_config_yaml(config_path)
    logger.info(
        "Starting ActorService on %s:%d", cfg.actor_service.bind_host, cfg.actor_service.port
    )
    actor = ActorService(cfg.rl, cfg.actor_service)
    actor.serve_forever()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    try:
        if args.role == "replay_manager":
            _run_replay_manager(args.config)
        elif args.role == "learner_service":
            _run_learner_service(args.config, args.run_dir)
        elif args.role == "actor_service":
            _run_actor_service(args.config)
    except KeyboardInterrupt:
        logger.info("Role %s interrupted.", args.role)
        sys.exit(0)


if __name__ == "__main__":
    main()
