from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    pass  # transformers is required; guard keeps mypy happy

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..pi05.modeling_pi05 import PI05Policy


class PI05RLT(PI05Policy):
    """PI05Policy extended with prefix-embedding extraction for RLToken Stage 1 training.

    Adds two methods on top of the unmodified parent:

    - ``extract_prefix_embeddings``: runs only the vision encoder (+ optionally the
      language token embedder), returning the VLA prefix representations without any
      action expert or flow-matching step.  Used by ``RLTokenPolicy`` when
      ``alpha == 0`` (frozen VLA).

    - ``compute_loss_with_prefix``: runs the full VLA flow-matching forward to obtain
      the per-sample loss, then separately extracts prefix embeddings under
      ``torch.no_grad``.  Used when ``alpha > 0`` (joint VLA + RLToken training).
      The two passes share the preprocessed batch tensors; only the vision tower is
      run twice.  This is acceptable because prefix embeddings are always detached
      before entering the RLToken encoder, so no gradient need flow through the
      second pass.

    No existing files are modified.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_prefix_embeddings(
        self,
        batch: dict[str, Tensor],
        image_only: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Extract VLA prefix embeddings from a preprocessed batch.

        Args:
            batch: Standard lerobot batch dict (images + language tokens + state).
            image_only: If True (default) return only image-token embeddings; if
                False concatenate language-token embeddings as well.

        Returns:
            prefix_embs: ``[B, seq_len, hidden_dim]`` float tensor.
            prefix_mask: ``[B, seq_len]`` bool tensor — True where the token is
                valid (not a padding position).
        """
        images, img_masks = self._preprocess_images(batch)

        embs: list[Tensor] = []
        masks: list[Tensor] = []

        # _embed_images handles per-camera embedding + gradient checkpointing.
        # The batch_first fused path is used at inference; per-camera checkpointing
        # path is used during training — same as PI05Pytorch.embed_prefix.
        for img_emb, img_mask in zip(
            self.model._embed_images(images, img_masks), img_masks, strict=True
        ):
            bsize, n_tok = img_emb.shape[:2]
            embs.append(img_emb)
            # img_mask shape: [B] for single-frame, [B, T] for MEM video.
            # Use the last frame's mask as the per-token validity flag.
            cur_mask = img_mask[:, -1] if img_mask.ndim == 2 else img_mask
            masks.append(cur_mask[:, None].expand(bsize, n_tok))

        if not image_only:
            tokens = batch[OBS_LANGUAGE_TOKENS]
            attn_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
            lang_emb = self.model.paligemma_with_expert.embed_language_tokens(tokens)
            embs.append(lang_emb)
            masks.append(attn_mask.bool())

        prefix_embs = torch.cat(embs, dim=1)   # [B, seq_len, hidden_dim]
        prefix_mask = torch.cat(masks, dim=1)   # [B, seq_len]
        return prefix_embs, prefix_mask

    def compute_loss_with_prefix(
        self,
        batch: dict[str, Tensor],
        image_only: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full VLA forward that also returns prefix embeddings.

        Runs the flow-matching forward pass (with VLA gradients enabled) to get
        the per-sample VLA loss, then computes prefix embeddings in a separate
        ``torch.no_grad`` call.

        Used when ``RLTokenPolicyConfig.alpha > 0`` (joint training).

        Args:
            batch: Standard lerobot batch dict.
            image_only: Passed through to ``extract_prefix_embeddings``.

        Returns:
            vla_loss:     Scalar flow-matching loss tensor (differentiable w.r.t. VLA params).
            prefix_embs:  ``[B, seq_len, hidden_dim]`` — detached (no_grad).
            prefix_mask:  ``[B, seq_len]`` bool.
        """
        # VLA flow-matching loss — gradients flow through VLA parameters.
        vla_loss, _ = super().forward(batch)

        # Prefix embeddings for the RLToken encoder.
        # Always extracted under no_grad: the caller does prefix_embs.detach() before
        # passing to RLTokenModel.loss(), so no gradient is needed here.
        with torch.no_grad():
            prefix_embs, prefix_mask = self.extract_prefix_embeddings(
                batch, image_only=image_only
            )

        return vla_loss, prefix_embs, prefix_mask
