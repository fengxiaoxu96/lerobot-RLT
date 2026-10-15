#!/usr/bin/env python
"""Stage 1 RLToken training: offline VLA + RLToken encoder-decoder training.

Trains ``RLTokenPolicy`` on a ``LeRobotDataset`` of robot demonstrations.
The VLA backbone (pi0.5) is frozen by default (``policy.alpha == 0``); set
``policy.alpha > 0`` for joint VLA fine-tuning.

Usage — command-line overrides:
    python scripts/train_rlt.py \
        --policy.pi05_pretrained_path lerobot/pi05 \
        --dataset.repo_id my_org/so_arm_demos \
        --output_dir ./runs/rlt_stage1

Usage — YAML config file:
    python scripts/train_rlt.py \
        --config configs/rltoken/tasks/so_arm_ethernet/stage1.yaml

Outputs (inside ``output_dir``):
    checkpoint_NNNNNN/   — policy.save_pretrained() every save_every_n_steps
    train_log.jsonl      — one JSON record per log step
"""

import json
import logging
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

from lerobot.policies.rltoken.configuration_rltoken import RLTokenPolicyConfig
from lerobot.policies.rltoken.modeling_rltoken import RLTokenPolicy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLTTrainConfig:
    """Top-level config for Stage 1 RLToken training.

    Mirrors the structure of lerobot's ``TrainPipelineConfig`` but is
    self-contained: it does not require the full distributed-training stack.
    """

    # ---- core ----
    policy: RLTokenPolicyConfig = field(default_factory=RLTokenPolicyConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id=""))

    # ---- training ----
    steps: int = 5_000
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    # ---- logging / checkpointing ----
    log_every_n_steps: int = 100
    save_every_n_steps: int = 1_000
    output_dir: str = "./runs/rlt_stage1"

    # ---- optional W&B ----
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _resolve_policy_features(policy_cfg: RLTokenPolicyConfig, ds_meta: LeRobotDatasetMetadata) -> None:
    """Populate policy_cfg.input_features / output_features from dataset metadata.

    Mirrors the logic in ``lerobot.policies.factory.make_policy`` so that
    ``RLTokenPolicyConfig.validate_features()`` can succeed without going
    through the full make_policy factory.
    """
    from lerobot.configs.types import FeatureType

    all_features = dataset_to_policy_features(ds_meta.features)
    policy_cfg.output_features = {
        k: ft for k, ft in all_features.items() if ft.type is FeatureType.ACTION
    }
    if not policy_cfg.input_features:
        policy_cfg.input_features = {
            k: ft for k, ft in all_features.items() if k not in policy_cfg.output_features
        }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(cfg: RLTTrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    log_file = output_dir / "train_log.jsonl"

    # ---- device ----------------------------------------------------------------
    if cfg.policy.device:
        device = torch.device(cfg.policy.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # ---- dataset metadata ------------------------------------------------------
    if not cfg.dataset.repo_id:
        raise ValueError("dataset.repo_id must be set")
    ds_meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    logger.info(f"Dataset: {cfg.dataset.repo_id}  ({ds_meta.total_frames} frames, {ds_meta.total_episodes} episodes)")

    # ---- resolve policy features from dataset ----------------------------------
    _resolve_policy_features(cfg.policy, ds_meta)

    # ---- delta_timestamps for action chunk loading -----------------------------
    # RLTokenPolicyConfig.action_delta_indices = range(action_horizon), so
    # resolve_delta_timestamps converts those to float offsets at ds fps.
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    # ---- dataset ---------------------------------------------------------------
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        episodes=cfg.dataset.episodes,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lerobot_collate_fn,
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # ---- policy ----------------------------------------------------------------
    policy = RLTokenPolicy(cfg.policy)
    policy = policy.to(device)
    policy.train()

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy: {n_trainable:,} trainable / {n_total:,} total params")

    # ---- preprocessor (pi05 tokenizer + normalizer) ----------------------------
    # We reuse the VLA's preprocessor (pi05) to get language tokens and
    # normalisation statistics.  RLTokenPolicy does not register its own processor
    # in lerobot's factory, so we delegate to the inner VLA config.
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.vla.config,
        dataset_meta=ds_meta,
    )

    # ---- optimizer / scheduler -------------------------------------------------
    optimizer_cfg = cfg.policy.get_optimizer_preset()
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optimizer_cfg.build(trainable_params)

    scheduler_cfg = cfg.policy.get_scheduler_preset()
    lr_scheduler = scheduler_cfg.build(optimizer, cfg.steps)

    # ---- W&B -------------------------------------------------------------------
    wandb_run = None
    if cfg.wandb.enable:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                notes=cfg.wandb.notes,
                mode=cfg.wandb.mode,
                config={
                    "steps": cfg.steps,
                    "batch_size": cfg.batch_size,
                    "alpha": cfg.policy.alpha,
                    "num_rl_tokens": cfg.policy.num_rl_tokens,
                    "dataset": cfg.dataset.repo_id,
                },
            )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")

    # ---- training loop ---------------------------------------------------------
    logger.info(f"Starting Stage 1 training for {cfg.steps} steps")
    t0 = time.time()

    for step in range(cfg.steps):
        batch = next(dl_iter)

        # Language tokenisation, state/action normalisation.
        batch = preprocessor(batch)
        batch = _to_device(batch, device)

        optimizer.zero_grad()

        loss, info = policy.forward(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            cfg.policy.optimizer_grad_clip_norm,
        )

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # ---- logging -----------------------------------------------------------
        if step % cfg.log_every_n_steps == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            log_entry = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": float(grad_norm),
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
                **{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in info.items()},
            }
            logger.info(
                "step=%d/%d  loss=%.4f  grad_norm=%.3f  lr=%.2e  t=%.0fs",
                step, cfg.steps, log_entry["loss"], log_entry["grad_norm"], lr, elapsed,
            )
            with log_file.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            if wandb_run is not None:
                wandb_run.log(log_entry, step=step)

        # ---- checkpointing -----------------------------------------------------
        if (step + 1) % cfg.save_every_n_steps == 0 or step == cfg.steps - 1:
            ckpt_dir = output_dir / f"checkpoint_{step + 1:06d}"
            policy.save_pretrained(ckpt_dir)
            logger.info("Saved checkpoint → %s", ckpt_dir)

    if wandb_run is not None:
        wandb_run.finish()

    total_time = time.time() - t0
    logger.info("Training complete in %.0fs", total_time)
    logger.info("Final checkpoint: %s", output_dir / f"checkpoint_{cfg.steps:06d}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: RLTTrainConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
