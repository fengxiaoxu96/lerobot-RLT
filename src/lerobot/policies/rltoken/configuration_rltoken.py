from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("rltoken")
@dataclass
class RLTokenPolicyConfig(PreTrainedConfig):
    """draccus config for the RLTokenPolicy (Stage 1: offline VLA + RLToken training).

    Naming note: this class is ``RLTokenPolicyConfig`` — not ``RLTokenConfig`` — to
    avoid a collision with the plain architecture dataclass of the same name in
    ``rl_token.py``.  Both can coexist in ``modeling_rltoken.py`` imports.
    """

    # -------------------------------------------------------------------------
    # VLA backbone
    # -------------------------------------------------------------------------
    # HuggingFace Hub repo ID or local path to a pi0.5 checkpoint directory.
    # Required before training; may be left empty when loading a full RLToken
    # checkpoint that already embeds the VLA weights.
    pi05_pretrained_path: str = ""

    # Joint training coefficient.  0 = VLA is fully frozen (only RLToken trains);
    # 1 = VLA gradients are added at full weight.  Intermediate values are valid.
    alpha: float = 0.0

    # If True, only image-token prefix embeddings are passed to the RLToken encoder
    # (language/text tokens are dropped).  Matches the reference openpi-RLT setup.
    image_only_prefix: bool = True

    # -------------------------------------------------------------------------
    # RLToken architecture  (superset of rl_token.RLTokenConfig fields)
    # -------------------------------------------------------------------------
    num_rl_tokens: int = 1      # number of compressed RL-state tokens z_rl
    num_layers: int = 2         # cross-attention layers in encoder & decoder
    embed_dim: int = 2048       # internal dimension; matches PaliGemma 2B hidden size
    input_dim: int = 2048       # VLA prefix hidden size
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    # Pre-allocated PE length; actual prefix seq_len must be <= this.
    max_seq_len: int = 1024

    # -------------------------------------------------------------------------
    # Action & observation space (SO-ARM defaults)
    # -------------------------------------------------------------------------
    action_dim: int = 6         # SO-ARM 6-DOF
    # Full VLA action chunk length (number of flow-matching output steps).
    action_horizon: int = 50
    # RL execution chunk length (steps actually sent to the robot per chunk).
    chunk_len: int = 10

    # pi0.5 image resolution (height, width).
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # -------------------------------------------------------------------------
    # Optimizer / scheduler (Stage 1 training)
    # -------------------------------------------------------------------------
    # Learning rate applies only to RLToken parameters when alpha=0; to both
    # RLToken + VLA when alpha > 0 (though you may want separate LRs — extend
    # as needed via ``get_optimizer_preset``).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 5_000
    scheduler_decay_lr: float = 1e-5

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------
    n_obs_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

        if self.chunk_len > self.action_horizon:
            raise ValueError(
                f"chunk_len ({self.chunk_len}) cannot exceed action_horizon ({self.action_horizon})"
            )

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "input_features and output_features must be resolved from the dataset "
                "before calling validate_features()"
            )

        # Ensure at least one visual feature is present (Stage 1 requires images).
        has_visual = any(
            f.type == FeatureType.VISUAL for f in self.input_features.values()
        )
        if not has_visual:
            raise ValueError(
                "RLTokenPolicyConfig requires at least one VISUAL input feature. "
                "Check that the dataset provides camera observations."
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.action_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
