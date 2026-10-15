"""Stage 1 RLToken policy: VLA + RLToken encoder-decoder.

``RLTokenPolicy`` owns:
  - ``vla``       (PI05RLT)   — frozen or jointly-trained VLA backbone
  - ``rlt_model`` (RLTokenModel) — RLToken encoder + decoder

Training (Stage 1):
    loss, info = policy.forward(batch)   # MSE recon ± VLA flow-matching

Machine A serving:
    z_rl      = policy.encode(batch)              # [B, num_rl_tokens, embed_dim]
    ref_chunk = policy.predict_action_chunk(batch) # [B, action_horizon, action_dim]
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.import_utils import require_package

from ..pi05.configuration_pi05 import PI05Config
from ..pretrained import PreTrainedPolicy
from ..pi05_rlt.modeling_pi05_rlt import PI05RLT
from .configuration_rltoken import RLTokenPolicyConfig
from .rl_token import RLTokenConfig, RLTokenModel


# ---------------------------------------------------------------------------
# Helper: build a minimal PI05Config from RLTokenPolicyConfig
# ---------------------------------------------------------------------------


def _build_pi05_config(config: RLTokenPolicyConfig) -> PI05Config:
    """Construct a fresh PI05Config from RLTokenPolicyConfig for scratch init.

    Used when ``pi05_pretrained_path`` is empty (testing or ablations).
    For production, always load from a pretrained pi0.5 checkpoint.
    """
    pi05 = PI05Config(
        chunk_size=config.action_horizon,
        n_action_steps=config.action_horizon,
        # pi0.5 pads actions/states to the nearest power of 2 (≥32).
        max_state_dim=max(config.action_dim, 32),
        max_action_dim=max(config.action_dim, 32),
        image_resolution=config.image_resolution,
        # Copy feature dicts so the two configs are independent.
        input_features=dict(config.input_features) if config.input_features else {},
        output_features=dict(config.output_features) if config.output_features else {},
        n_obs_steps=config.n_obs_steps,
    )
    pi05.validate_features()
    return pi05


# ---------------------------------------------------------------------------
# RLTokenPolicy
# ---------------------------------------------------------------------------


class RLTokenPolicy(PreTrainedPolicy):
    """Stage 1 policy: trains the RLToken encoder-decoder on top of a frozen (or
    jointly fine-tuned) pi0.5 VLA.

    Class variables required by PreTrainedPolicy:
        config_class  — the draccus config registered as "rltoken"
        name          — used as the policy identifier in run configs
    """

    config_class = RLTokenPolicyConfig
    name = "rltoken"

    def __init__(self, config: RLTokenPolicyConfig) -> None:
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()

        # --- VLA ----------------------------------------------------------------
        # Load pretrained weights when a path is given; otherwise fresh-init for
        # unit tests / ablations.  The pretrained loader handles openpi key-remapping.
        if config.pi05_pretrained_path:
            self.vla: PI05RLT = PI05RLT.from_pretrained(config.pi05_pretrained_path)
        else:
            pi05_config = _build_pi05_config(config)
            self.vla = PI05RLT(pi05_config)

        # --- RLToken model -------------------------------------------------------
        rlt_arch = RLTokenConfig(
            num_rl_tokens=config.num_rl_tokens,
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            input_dim=config.input_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout_rate=config.dropout_rate,
            max_seq_len=config.max_seq_len,
        )
        self.rlt_model = RLTokenModel(rlt_arch)

        # --- Gradient routing ----------------------------------------------------
        # When alpha == 0 the VLA is fully frozen; only rlt_model trains.
        if config.alpha == 0.0:
            self._freeze_vla()

        self.reset()

    # ------------------------------------------------------------------
    # VLA freeze / unfreeze helpers (also called by train_rlt.py)
    # ------------------------------------------------------------------

    def _freeze_vla(self) -> None:
        for param in self.vla.parameters():
            param.requires_grad_(False)

    def freeze_vla(self) -> None:
        """Freeze VLA parameters (idempotent)."""
        self._freeze_vla()

    def unfreeze_vla(self) -> None:
        """Restore VLA gradients for joint training."""
        for param in self.vla.parameters():
            param.requires_grad_(True)

    # ------------------------------------------------------------------
    # PreTrainedPolicy interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state between episodes (delegates to VLA queue)."""
        self.vla.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Stage 1 loss for one training step.

        ``alpha == 0`` (VLA frozen, default):
            total = rlt_loss   (MSE reconstruction only)

        ``alpha > 0`` (joint training):
            total = rlt_loss + alpha * vla_loss

        Args:
            batch: Standard lerobot batch dict from LeRobotDataset.

        Returns:
            (scalar_loss, log_dict)  — log_dict contains at minimum {"mse": float}.
        """
        if self.config.alpha == 0.0:
            # VLA is frozen: run embedding under no_grad to avoid storing
            # unnecessary activations; rlt_model.loss already expects detached input.
            with torch.no_grad():
                prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
                    batch, image_only=self.config.image_only_prefix
                )
            rlt_loss, info = self.rlt_model.loss(prefix_embs, prefix_mask)
            return rlt_loss, info
        else:
            # Joint training: VLA gradients come from vla_loss; prefix_embs
            # are detached inside rlt_model.loss so no VLA grad via RLT path.
            vla_loss, prefix_embs, prefix_mask = self.vla.compute_loss_with_prefix(
                batch, image_only=self.config.image_only_prefix
            )
            rlt_loss, rlt_info = self.rlt_model.loss(prefix_embs, prefix_mask)
            total = rlt_loss + self.config.alpha * vla_loss
            return total, {
                "rlt_loss": rlt_loss.item(),
                "vla_loss": vla_loss.item(),
                **{k: v.item() if isinstance(v, Tensor) else v for k, v in rlt_info.items()},
            }

    # ------------------------------------------------------------------
    # Inference (Machine A serving)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """Return RL-state token z_rl for Machine A serving.

        Returns:
            z_rl: [B, num_rl_tokens, embed_dim]
                Caller (serve_rlt_policy.py) may flatten to [B, num_rl_tokens * embed_dim].
        """
        prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
            batch, image_only=self.config.image_only_prefix
        )
        return self.rlt_model.encode(prefix_embs, prefix_mask)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return the VLA reference chunk for Machine A serving.

        Returns:
            ref_chunk: [B, action_horizon, action_dim]
                Serving script slices to [B, chunk_len, action_dim] before sending.
        """
        return self.vla.predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action per environment step (delegates to VLA queue)."""
        return self.vla.select_action(batch, **kwargs)
