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

"""PyTorch implementation of the RLToken encoder-decoder bottleneck.

Ported from the JAX/Flax reference in openpi-RLT:
    src/openpi/models/rl_token.py

The three public classes are:
    RLTokenConfig   — architecture hyper-parameters (plain frozen dataclass)
    RLTokenEncoder  — compresses VLA prefix embeddings → RL tokens
    RLTokenDecoder  — reconstructs prefix embeddings from RL tokens
    RLTokenModel    — wraps encoder + decoder; exposes encode() and loss()

Both encoder and decoder stack num_layers of _CrossAttentionBlock, which uses
the same GeGLU-based MLP as the pi0 / Gemma transformer family.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sinusoidal_pe(seq_len: int, d_embed: int) -> Tensor:
    """Sinusoidal positional encoding, shape [seq_len, d_embed].

    Matches the JAX sinusoidal_pe_init used in the reference implementation:
        pe[:, 0::2] = sin(pos / 10000^(2i/d))
        pe[:, 1::2] = cos(pos / 10000^(2i/d))
    The JAX version concatenates sin and cos halves; the result is identical
    to the interleaved form when d_embed is even.
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [seq, 1]
    half = d_embed // 2
    div_term = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / d_embed))
    pe = torch.empty(seq_len, d_embed)
    pe[:, :half] = torch.sin(position * div_term)
    pe[:, half:] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Building blocks (private — not exported)
# ---------------------------------------------------------------------------


class _GeGLUGate(nn.Module):
    """GeGLU gating unit: Linear(in → in*2) → split → x * gelu(gate).

    The JAX GeGLU is defined identically: it receives a tensor of some dim,
    projects to 2×dim, splits, and returns dim-sized output.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        x2 = self.proj(x)
        x_val, gate = x2.chunk(2, dim=-1)
        return x_val * F.gelu(gate)


class _CrossAttentionBlock(nn.Module):
    """One transformer block: self-attention → cross-attention → GeGLU MLP.

    Mirrors CrossAttentionLayer from openpi's fsq_tokenizer.py, translated to
    PyTorch nn.MultiheadAttention (batch_first=True).

    Args:
        embed_dim:    model width.
        num_heads:    attention heads.
        mlp_ratio:    MLP hidden dim = embed_dim * mlp_ratio.
        dropout_rate: applied inside attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        d_ff = int(embed_dim * mlp_ratio)

        # Self-attention
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.sa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

        # Cross-attention (Q from x, K/V from kv)
        self.ca_norm = nn.LayerNorm(embed_dim)
        self.ca = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

        # MLP: expand → drop → GeGLU gate → project back
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp_fc1 = nn.Linear(embed_dim, d_ff)
        self.mlp_drop = nn.Dropout(dropout_rate)
        self.mlp_gate = _GeGLUGate(d_ff)
        self.mlp_fc2 = nn.Linear(d_ff, embed_dim)

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        kv_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:               [B, q_len, embed_dim]
            kv:              [B, kv_len, embed_dim]
            kv_padding_mask: [B, kv_len] bool — True means *ignore* (PyTorch convention).
        Returns:
            [B, q_len, embed_dim]
        """
        # Self-attention
        skip = x
        x = self.sa_norm(x)
        x, _ = self.sa(x, x, x, need_weights=False)
        x = skip + x

        # Cross-attention
        skip = x
        x = self.ca_norm(x)
        x, _ = self.ca(
            x, kv, kv,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        x = skip + x

        # GeGLU MLP
        skip = x
        x = self.mlp_norm(x)
        x = self.mlp_fc1(x)
        x = self.mlp_drop(x)
        x = self.mlp_gate(x)
        x = self.mlp_fc2(x)
        return skip + x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """Architecture hyper-parameters for RLTokenEncoder / RLTokenDecoder.

    Note: this is a plain Python frozen dataclass, not a draccus ChoiceRegistry
    config.  The draccus-registered policy config lives in
    configuration_rltoken.py and embeds these fields.
    """

    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 2048       # Internal dimension; matches PaliGemma 2B hidden size
    input_dim: int = 2048       # VLA prefix embedding dimension (Gemma 2B = 2048)
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    # Maximum prefix sequence length to pre-allocate positional encodings for.
    # Actual seq_len is determined at runtime and must be <= max_seq_len.
    # PaliGemma with 3×224×224 images at patch-14 produces ~768 image tokens.
    max_seq_len: int = 1024


class RLTokenEncoder(nn.Module):
    """Compresses VLA prefix embeddings into a compact set of RL tokens.

    Architecture:
        Input  : prefix_embs  [B, seq_len, input_dim]
        Queries: learned       [1, num_rl_tokens, embed_dim]  (sinusoidal init)
        KV     : prefix_embs + learned positional encoding [B, seq_len, embed_dim]
        Output : rl_tokens    [B, num_rl_tokens, embed_dim]

    Each of the num_layers _CrossAttentionBlock layers refines the query
    embeddings by attending over the (position-encoded) prefix.
    """

    def __init__(self, config: RLTokenConfig) -> None:
        super().__init__()
        cfg = config

        # Optional input projection when input_dim != embed_dim
        self.input_proj: nn.Linear | None = (
            nn.Linear(cfg.input_dim, cfg.embed_dim) if cfg.input_dim != cfg.embed_dim else None
        )

        # Learnable query embeddings — these ARE the RL tokens before refinement.
        # Shape [1, num_rl_tokens, embed_dim]; broadcast over batch at forward time.
        self.q_embed = nn.Parameter(_sinusoidal_pe(cfg.num_rl_tokens, cfg.embed_dim).unsqueeze(0))

        # Learnable KV positional encoding added to the prefix before cross-attn.
        # Pre-allocated for max_seq_len; sliced to actual seq_len at forward time.
        self.kv_pos_enc = nn.Parameter(_sinusoidal_pe(cfg.max_seq_len, cfg.embed_dim).unsqueeze(0))

        self.layers = nn.ModuleList(
            [
                _CrossAttentionBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout_rate)
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(
        self,
        prefix_embs: Tensor,
        prefix_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            prefix_embs:  [B, seq_len, input_dim]  VLA prefix hidden states.
            prefix_mask:  [B, seq_len] bool — True means *valid* position
                          (JAX convention; inverted before passing to PyTorch attn).
        Returns:
            rl_tokens: [B, num_rl_tokens, embed_dim]
        """
        B, seq_len, _ = prefix_embs.shape

        if self.input_proj is not None:
            prefix_embs = self.input_proj(prefix_embs)

        # Add positional encoding to the key-value input
        kv = prefix_embs + self.kv_pos_enc[:, :seq_len, :]  # [B, seq_len, embed_dim]

        # Broadcast learnable queries over the batch
        x = self.q_embed.expand(B, -1, -1)  # [B, num_rl_tokens, embed_dim]

        # Convert mask to PyTorch convention (True = ignore)
        kv_padding_mask = ~prefix_mask if prefix_mask is not None else None

        for layer in self.layers:
            x = layer(x, kv, kv_padding_mask=kv_padding_mask)

        return x  # [B, num_rl_tokens, embed_dim]


class RLTokenDecoder(nn.Module):
    """Reconstructs prefix embeddings from RL tokens (used only in Stage 1 training).

    Architecture:
        Input  : rl_tokens      [B, num_rl_tokens, embed_dim]
        Queries: learned         [1, target_seq_len, embed_dim]  (sinusoidal init, sliced)
        KV     : rl_tokens + learned positional encoding [B, num_rl_tokens, embed_dim]
        Output : reconstructed  [B, target_seq_len, embed_dim (or input_dim if projected)]
    """

    def __init__(self, config: RLTokenConfig) -> None:
        super().__init__()
        cfg = config

        # Learnable query embeddings for each reconstruction position.
        # Pre-allocated for max_seq_len; sliced at forward time.
        self.q_embed = nn.Parameter(_sinusoidal_pe(cfg.max_seq_len, cfg.embed_dim).unsqueeze(0))

        # Learnable positional encoding for the (fixed-length) RL token KV
        self.kv_pos_enc = nn.Parameter(_sinusoidal_pe(cfg.num_rl_tokens, cfg.embed_dim).unsqueeze(0))

        self.layers = nn.ModuleList(
            [
                _CrossAttentionBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout_rate)
                for _ in range(cfg.num_layers)
            ]
        )

        # Optional output projection back to input_dim
        self.output_proj: nn.Linear | None = (
            nn.Linear(cfg.embed_dim, cfg.input_dim) if cfg.input_dim != cfg.embed_dim else None
        )

    def forward(self, rl_tokens: Tensor, target_seq_len: int) -> Tensor:
        """
        Args:
            rl_tokens:       [B, num_rl_tokens, embed_dim]
            target_seq_len:  number of prefix positions to reconstruct
                             (must be <= max_seq_len).
        Returns:
            [B, target_seq_len, embed_dim]  (or input_dim if projected)
        """
        B = rl_tokens.shape[0]

        # Add positional encoding to RL token KV
        kv = rl_tokens + self.kv_pos_enc  # [B, num_rl_tokens, embed_dim]

        # Slice learnable queries for the target sequence length
        x = self.q_embed[:, :target_seq_len, :].expand(B, -1, -1)  # [B, target_seq_len, embed_dim]

        for layer in self.layers:
            x = layer(x, kv)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x  # [B, target_seq_len, embed_dim (or input_dim)]


class RLTokenModel(nn.Module):
    """Full encoder-decoder model used for Stage 1 training.

    Wraps RLTokenEncoder and RLTokenDecoder.  The encoder is also used
    standalone during Machine A serving (encode() only, no decoder).

    Usage in Stage 1:
        mse, info = model.loss(prefix_embs.detach(), prefix_mask)
        mse.backward()

    Usage in Machine A serving:
        z_rl = model.encode(prefix_embs, prefix_mask)  # [B, num_rl_tokens, embed_dim]
    """

    def __init__(self, config: RLTokenConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def encode(
        self,
        prefix_embs: Tensor,
        prefix_mask: Tensor | None = None,
    ) -> Tensor:
        """Return rl_tokens [B, num_rl_tokens, embed_dim]."""
        return self.encoder(prefix_embs, prefix_mask)

    def loss(
        self,
        prefix_embs: Tensor,
        prefix_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """MSE reconstruction loss.

        Args:
            prefix_embs: [B, seq_len, input_dim] — should already be detached
                         from the VLA computation graph (stop_gradient in JAX).
            prefix_mask: [B, seq_len] bool — True = valid position.
        Returns:
            (mse_scalar, {"mse": mse_scalar})
        """
        rl_tokens = self.encoder(prefix_embs, prefix_mask)
        target_seq_len = prefix_embs.shape[1]
        reconstructed = self.decoder(rl_tokens, target_seq_len)

        # stop_gradient equivalent — caller should already detach, but double-guard
        target = prefix_embs.detach()
        sq_error = (reconstructed - target).pow(2)  # [B, seq_len, input_dim]

        if prefix_mask is not None:
            # Average only over valid (unmasked) token positions
            mask_f = prefix_mask.unsqueeze(-1).float()   # [B, seq_len, 1]
            num_valid = mask_f.sum() * prefix_embs.shape[-1]  # scalar
            mse = (sq_error * mask_f).sum() / num_valid.clamp(min=1.0)
        else:
            mse = sq_error.mean()

        return mse, {"mse": mse.detach()}
