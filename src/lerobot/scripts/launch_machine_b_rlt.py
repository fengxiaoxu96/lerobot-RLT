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

"""Launch all Machine B services for the RLToken online RL pipeline.

Spawns three service processes (replay_manager, learner_service, actor_service)
via run_rlt_role.py and monitors them for crashes.  Terminates all processes
cleanly on Ctrl-C or if any service exits unexpectedly.

Optionally starts a Weights & Biases monitor process to stream learner metrics.

Usage::

    python launch_machine_b_rlt.py --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml

    # With W&B:
    python launch_machine_b_rlt.py \\
        --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml \\
        --wandb-project rlt-online-rl \\
        --wandb-run-name so-arm-test

    # Resume from a checkpoint:
    python launch_machine_b_rlt.py \\
        --config configs/rltoken/tasks/so_arm_ethernet/online_rl.yaml \\
        --run-dir /path/to/previous/run
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import NamedTuple

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 1.0
_STARTUP_GRACE_SEC = 5.0  # time to let processes start before first crash check


class _Proc(NamedTuple):
    name: str
    proc: subprocess.Popen


def _spawn_role(
    role: str,
    config_path: str,
    *,
    run_dir: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn `run_rlt_role.py --role <role> --config <config>` as a subprocess."""
    script = os.path.join(os.path.dirname(__file__), "run_rlt_role.py")
    cmd = [sys.executable, script, "--role", role, "--config", config_path]
    if run_dir:
        cmd += ["--run-dir", run_dir]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    logger.info("Spawning %s: %s", role, " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _spawn_wandb_monitor(
    run_dir: str,
    project: str,
    run_name: str,
    entity: str | None,
) -> subprocess.Popen:
    """Spawn a simple W&B uploader that tails learner_status.json metrics."""
    script = os.path.join(os.path.dirname(__file__), "_rlt_wandb_monitor.py")
    if not os.path.exists(script):
        logger.warning("W&B monitor script not found at %s; skipping.", script)
        return None
    cmd = [
        sys.executable, script,
        "--run-dir", run_dir,
        "--project", project,
        "--run-name", run_name,
    ]
    if entity:
        cmd += ["--entity", entity]
    return subprocess.Popen(cmd)


def _terminate_all(procs: list[_Proc]) -> None:
    logger.info("Terminating all services…")
    for p in procs:
        if p.proc.poll() is None:
            logger.info("Sending SIGTERM to %s (pid=%d)", p.name, p.proc.pid)
            p.proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    for p in procs:
        remaining = deadline - time.monotonic()
        if remaining > 0 and p.proc.poll() is None:
            try:
                p.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("%s did not exit cleanly; sending SIGKILL", p.name)
                p.proc.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Machine B RLToken services.")
    parser.add_argument("--config", required=True, help="Path to OnlineRLSystemConfig YAML.")
    parser.add_argument("--run-dir", default=None, help="Override run directory.")
    parser.add_argument("--wandb-project", default=None, help="Enable W&B monitoring.")
    parser.add_argument("--wandb-run-name", default="rlt-online-rl")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()

    procs: list[_Proc] = []

    def _shutdown(signum=None, frame=None):
        logger.info("Received shutdown signal; stopping all services.")
        _terminate_all(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Spawn all three B services
    for role in ("replay_manager", "learner_service", "actor_service"):
        p = _spawn_role(role, args.config, run_dir=args.run_dir)
        procs.append(_Proc(name=role, proc=p))

    # Optional W&B monitor
    wandb_proc = None
    if args.wandb_project and args.run_dir:
        wandb_proc = _spawn_wandb_monitor(
            args.run_dir, args.wandb_project, args.wandb_run_name, args.wandb_entity
        )

    logger.info("All Machine B services started. Monitoring for crashes…")
    time.sleep(_STARTUP_GRACE_SEC)

    # Main monitoring loop
    while True:
        time.sleep(_POLL_INTERVAL_SEC)
        for p in procs:
            rc = p.proc.poll()
            if rc is not None:
                logger.error(
                    "Service %s (pid=%d) exited with returncode=%d. Shutting down.",
                    p.name, p.proc.pid, rc,
                )
                if wandb_proc and wandb_proc.poll() is None:
                    wandb_proc.terminate()
                _terminate_all(procs)
                sys.exit(1)


if __name__ == "__main__":
    main()
