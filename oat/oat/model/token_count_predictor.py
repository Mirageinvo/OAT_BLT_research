"""
Token-count predictor for adaptive-budget OATPolicy.

Maps the fused observation embedding (output of `OATPolicy.obs_encoder`,
shape [B, To, d]) to a predicted min-k -- the number of action tokens the
autoregressive policy should generate for this observation.

Trained offline on labels from `scripts/collect_min_k_dataset.py`
(min_k = fewest tokens whose detokenized action lands within eps of the
demonstration). At inference the same `features` tensor the policy already
computes is reused, so the predictor adds only a small MLP on top.

Feature standardization (z-score) stats are stored as buffers, so a loaded
predictor is self-contained: pass raw `features` and it normalizes internally.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class TokenCountPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,            # per-step feature dim d (e.g. 138)
        n_obs_steps: int,       # To (e.g. 2)
        num_classes: int,       # number of output classes
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        class_values: Optional[List[int]] = None,  # k value per class; default [1..num_classes].
                                                    # For pow2 OAT use [1,2,4,8] (only trained budgets).
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_obs_steps = n_obs_steps
        self.num_classes = num_classes
        self.flat_dim = in_dim * n_obs_steps

        if class_values is None:
            class_values = list(range(1, num_classes + 1))
        assert len(class_values) == num_classes, "class_values must have length num_classes"
        self.register_buffer("class_values", torch.tensor(class_values, dtype=torch.long))

        # feature standardization (set via set_feature_stats); identity by default
        self.register_buffer("feat_mean", torch.zeros(self.flat_dim))
        self.register_buffer("feat_std", torch.ones(self.flat_dim))

        layers: List[nn.Module] = []
        prev = self.flat_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        """Store flattened (To*d,) standardization stats computed on the training split."""
        self.feat_mean.copy_(mean.reshape(-1))
        self.feat_std.copy_(std.reshape(-1).clamp_min(eps))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, To, d] -> logits [B, num_classes]."""
        x = features.reshape(features.shape[0], -1)         # [B, To*d]
        x = (x - self.feat_mean) / self.feat_std
        return self.net(x)

    @torch.inference_mode()
    def predict_k(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, To, d] -> predicted k (mapped through class_values), long tensor [B]."""
        idx = self.forward(features).argmax(dim=-1)
        return self.class_values[idx]

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "TokenCountPredictor":
        """Load a predictor saved by scripts/train_token_count_predictor.py."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model
