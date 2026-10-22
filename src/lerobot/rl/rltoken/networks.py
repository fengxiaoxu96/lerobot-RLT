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

"""PyTorch actor-critic networks for the RLToken online RL pipeline.

Three network classes:
    ChunkActor  — MLP that refines a VLA ref_chunk into a sampled action chunk.
    QNetwork    — MLP that estimates Q(z_rl, proprio, action_chunk).
    TwinCritic  — Clipped double-Q: two QNetwork instances.

Two loss functions and a TD-target helper:
    build_td_target        — discounted chunk rewards + bootstrap value.
    compute_critic_loss    — MSE between current Q and TD target for both heads.
    compute_actor_loss     — -Q + bc_weight * BC_penalty; source-aware BC target.

One utility:
    apply_reference_dropout — zero entire ref_chunk for a batch element with
                              probability `dropout_prob`.

Architecture mirrors the JAX reference in openpi-RLT / networks.py.  The
non-affine LayerNorm (elementwise_affine=False) matches the reference's manual
_layer_norm which has no learnable scale/shift.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.rl.rltoken.config import RLTOnlineRLConfig
from lerobot.rl.rltoken.transition import TransitionSource


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------


def _build_mlp(in_dim: int, hidden_dim: int, num_layers: int, out_dim: int) -> nn.Sequential:
    """Build a MLP with LayerNorm+GELU hidden layers and a linear output layer.

    Matches the JAX reference _mlp_forward:
      - Each hidden layer: Linear → LayerNorm (no affine) → GELU
      - Output layer: Linear only (no activation, no norm)
    """
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(cur, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6))
        layers.append(nn.GELU())
        cur = hidden_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# ChunkActor
# ---------------------------------------------------------------------------


class ChunkActor(nn.Module):
    """Refines a VLA reference chunk into a sampled action chunk.

    Inputs  (all float32):
        z_rl        [B, z_dim]                 compressed VLA state token
        proprio     [B, proprio_dim]           joint positions at chunk start
        ref_chunk   [B, chunk_len, action_dim] VLA reference actions (normalised)

    Forward returns the action mean [B, chunk_len, action_dim].
    Fixed-std Gaussian noise is added in `sample()`.

    Feature encoding (mirrors JAX reference):
        z_feat     = LN( Linear(z_rl) )                      dim 256
        proprio_feat = tanh( LN( Linear(proprio) ) )         dim 64
        ref_feat   = tanh( LN( Linear(ref_flat) ) )          dim 256
        concat → [B, 576] → trunk MLP → [B, chunk_len * action_dim]
    """

    def __init__(self, config: RLTOnlineRLConfig) -> None:
        super().__init__()
        z_dim = config.z_dim
        proprio_dim = config.proprio_dim
        chunk_len = config.chunk_len
        action_dim = config.action_dim
        hidden_dim = config.actor_hidden_dim
        num_layers = config.actor_num_layers

        self.chunk_len = chunk_len
        self.action_dim = action_dim

        self.z_proj = nn.Linear(z_dim, 256)
        self.proprio_proj = nn.Linear(proprio_dim, 64)
        self.ref_proj = nn.Linear(chunk_len * action_dim, 256)

        self.z_ln = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)
        self.proprio_ln = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.ref_ln = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)

        trunk_in = 256 + 64 + 256
        self.trunk = _build_mlp(trunk_in, hidden_dim, num_layers, chunk_len * action_dim)

        # fixed_std registered as a buffer so it travels with .to(device/dtype)
        self.register_buffer("fixed_std", torch.tensor(config.fixed_std, dtype=torch.float32))

    def _encode(self, z_rl: Tensor, proprio: Tensor, ref_chunk: Tensor) -> Tensor:
        B = z_rl.shape[0]
        ref_flat = ref_chunk.reshape(B, self.chunk_len * self.action_dim)
        z_feat = self.z_ln(self.z_proj(z_rl))
        proprio_feat = torch.tanh(self.proprio_ln(self.proprio_proj(proprio)))
        ref_feat = torch.tanh(self.ref_ln(self.ref_proj(ref_flat)))
        return torch.cat([z_feat, proprio_feat, ref_feat], dim=-1)

    def forward(self, z_rl: Tensor, proprio: Tensor, ref_chunk: Tensor) -> Tensor:
        """Return action mean [B, chunk_len, action_dim]."""
        B = z_rl.shape[0]
        features = self._encode(z_rl, proprio, ref_chunk)
        return self.trunk(features).reshape(B, self.chunk_len, self.action_dim)

    def sample(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Sample an action chunk and return (chunk, summed_log_prob).

        log_prob is the sum over all chunk steps and action dims of the
        Gaussian log-density N(action | mu, fixed_std).
        """
        mu = self.forward(z_rl, proprio, ref_chunk)
        std: Tensor = self.fixed_std  # scalar buffer
        if deterministic:
            chunk = mu
        else:
            chunk = mu + std * torch.randn_like(mu)
        # sum log-prob over all dims (chunk_len × action_dim)
        log_prob = (
            -0.5 * ((chunk - mu) / std).pow(2)
            - std.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(dim=(-2, -1))
        return chunk, log_prob

    def apply_reference_dropout(self, ref_chunk: Tensor, prob: float) -> Tensor:
        """Zero the entire ref_chunk for a batch element with probability `prob`.

        Whole-chunk dropout: either the full chunk is preserved or fully zeroed
        (not per-step).  This encourages the actor to work without VLA guidance.
        """
        if prob <= 0.0 or not self.training:
            return ref_chunk
        # Bernoulli mask [B, 1, 1] — True means keep
        keep = torch.bernoulli(
            torch.full((ref_chunk.shape[0], 1, 1), 1.0 - prob, device=ref_chunk.device)
        ).bool()
        return ref_chunk * keep.to(ref_chunk.dtype)


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Single Q-value head: Q(z_rl, proprio, action_chunk) → [B].

    Same projection structure as ChunkActor but takes action_chunk (not
    ref_chunk) and outputs a scalar Q-value per batch element.
    """

    def __init__(self, config: RLTOnlineRLConfig) -> None:
        super().__init__()
        z_dim = config.z_dim
        proprio_dim = config.proprio_dim
        chunk_len = config.chunk_len
        action_dim = config.action_dim
        hidden_dim = config.critic_hidden_dim
        num_layers = config.critic_num_layers

        self.chunk_len = chunk_len
        self.action_dim = action_dim

        self.z_proj = nn.Linear(z_dim, 256)
        self.proprio_proj = nn.Linear(proprio_dim, 64)
        self.action_proj = nn.Linear(chunk_len * action_dim, 256)

        self.z_ln = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)
        self.proprio_ln = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.action_ln = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)

        trunk_in = 256 + 64 + 256
        self.trunk = _build_mlp(trunk_in, hidden_dim, num_layers, 1)

    def forward(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        """Return Q-value [B]."""
        B = z_rl.shape[0]
        action_flat = action_chunk.reshape(B, self.chunk_len * self.action_dim)
        z_feat = self.z_ln(self.z_proj(z_rl))
        proprio_feat = torch.tanh(self.proprio_ln(self.proprio_proj(proprio)))
        action_feat = torch.tanh(self.action_ln(self.action_proj(action_flat)))
        features = torch.cat([z_feat, proprio_feat, action_feat], dim=-1)
        return self.trunk(features).squeeze(-1)


# ---------------------------------------------------------------------------
# TwinCritic
# ---------------------------------------------------------------------------


class TwinCritic(nn.Module):
    """Clipped double-Q critic: two independent QNetwork instances (q1, q2)."""

    def __init__(self, config: RLTOnlineRLConfig) -> None:
        super().__init__()
        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def q_values(
        self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (q1, q2) each of shape [B]."""
        return self.q1(z_rl, proprio, action_chunk), self.q2(z_rl, proprio, action_chunk)

    def min_q(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        """Return element-wise min(q1, q2), shape [B]."""
        q1, q2 = self.q_values(z_rl, proprio, action_chunk)
        return torch.minimum(q1, q2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def apply_reference_dropout(ref_chunk: Tensor, prob: float) -> Tensor:
    """Standalone (non-member) reference dropout for use outside of training mode.

    Useful in inference.py where the actor is in eval mode and the member
    method skips dropout.
    """
    if prob <= 0.0:
        return ref_chunk
    keep = torch.bernoulli(
        torch.full((ref_chunk.shape[0], 1, 1), 1.0 - prob, device=ref_chunk.device)
    ).bool()
    return ref_chunk * keep.to(ref_chunk.dtype)


def _discounted_chunk_rewards(rewards: Tensor, gamma: float) -> Tensor:
    """Sum per-step rewards with geometric discounting: sum_t r_t * gamma^t."""
    T = rewards.shape[-1]
    discounts = torch.pow(
        torch.tensor(gamma, dtype=rewards.dtype, device=rewards.device),
        torch.arange(T, dtype=rewards.dtype, device=rewards.device),
    )
    return (rewards * discounts.unsqueeze(0)).sum(dim=-1)


# ---------------------------------------------------------------------------
# TD target
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_td_target(
    next_z_rl: Tensor,
    next_proprio: Tensor,
    next_ref_chunk: Tensor,
    rewards: Tensor,
    done: Tensor,
    target_actor: ChunkActor,
    target_critic: TwinCritic,
    gamma: float,
) -> Tensor:
    """Compute the TD target for critic training.

    Uses the target actor to sample a stochastic next action (matching the
    reference), then takes min(q1, q2) from the target critic.

    Args:
        rewards : [B, chunk_len]
        done    : [B]  bool or float32

    Returns:
        TD target [B]
    """
    chunk_len = rewards.shape[-1]
    # Stochastic target action (matches reference; fixed_std is tiny so minimal effect)
    next_action, _ = target_actor.sample(next_z_rl, next_proprio, next_ref_chunk, deterministic=False)
    next_q = target_critic.min_q(next_z_rl, next_proprio, next_action)
    not_done = 1.0 - done.float()
    bootstrap = not_done * (gamma**chunk_len) * next_q
    return _discounted_chunk_rewards(rewards, gamma) + bootstrap


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def compute_critic_loss(
    critic: TwinCritic,
    target_actor: ChunkActor,
    target_critic: TwinCritic,
    z_rl: Tensor,
    proprio: Tensor,
    action_chunk: Tensor,
    rewards: Tensor,
    done: Tensor,
    next_z_rl: Tensor,
    next_proprio: Tensor,
    next_ref_chunk: Tensor,
    gamma: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the clipped double-Q critic loss.

    Loss = MSE(q1, target) + MSE(q2, target).

    Returns:
        (critic_loss scalar, metrics dict)
    """
    td_target = build_td_target(
        next_z_rl, next_proprio, next_ref_chunk, rewards, done,
        target_actor, target_critic, gamma,
    )
    q1, q2 = critic.q_values(z_rl, proprio, action_chunk)
    loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
    return loss, {
        "critic_loss": loss.detach(),
        "q1_mean": q1.detach().mean(),
        "q2_mean": q2.detach().mean(),
        "target_q_mean": td_target.mean(),
    }


def compute_actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    z_rl: Tensor,
    proprio: Tensor,
    ref_chunk: Tensor,
    action_chunk: Tensor,
    source_chunk: Tensor,
    bc_weight: float,
    q_weight: float,
    reference_dropout_prob: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the actor loss: -Q + bc_weight * BC_penalty.

    BC target is source-aware:
        - HUMAN / MIXED steps → use recorded action_chunk (human demonstration target)
        - BASE / RL steps     → use ref_chunk (VLA reference target)

    Reference dropout is applied to ref_chunk before the actor forward pass to
    encourage robustness to missing VLA guidance.

    Args:
        source_chunk : [B, chunk_len] uint8 TransitionSource per step

    Returns:
        (actor_loss scalar, metrics dict)
    """
    # Apply reference dropout (whole-chunk, per batch element)
    dropped_ref = actor.apply_reference_dropout(ref_chunk, reference_dropout_prob)
    sampled_chunk, _ = actor.sample(z_rl, proprio, dropped_ref)

    # Source-aware BC target
    human_mask = (
        (source_chunk == int(TransitionSource.HUMAN))
        | (source_chunk == int(TransitionSource.MIXED))
    ).float().unsqueeze(-1)  # [B, chunk_len, 1]
    bc_target = human_mask * action_chunk + (1.0 - human_mask) * ref_chunk

    bc_penalty = F.mse_loss(sampled_chunk, bc_target)

    # Q-value uses non-dropped ref (actor receives the clean ref for Q evaluation)
    q1, _ = critic.q_values(z_rl, proprio, sampled_chunk)
    q_loss = -q1.mean()

    actor_loss = bc_weight * bc_penalty + q_weight * q_loss
    return actor_loss, {
        "actor_loss": actor_loss.detach(),
        "bc_penalty": bc_penalty.detach(),
        "q_actor": q1.detach().mean(),
    }
