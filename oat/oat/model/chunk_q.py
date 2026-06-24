"""
Chunk-Q critic for value-guided best-of-N selection in OATPolicy (our Q-chunking analog, QC).

Scores a candidate action CHUNK given the (frozen) fused observation features:
    Q(features [B,To,d], chunk_norm [B,R,D]) -> success-value logit [B].

Trained offline (scripts/train_chunk_q.py) on collect_awr_dataset rollouts: each executed
chunk is labeled by its EPISODE success (Monte-Carlo return, gamma=1, sparse terminal reward),
so Q approximates P(success | state, chunk) under the data-collection policy. At inference we
sample N candidate chunks at a state (vision amortized) and EXECUTE argmax_chunk Q -- the
value-guided version of the verifier-free `vote` consensus selector (Q-chunking QC variant).
TEST: does value-BoN beat the vote-BoN plateau (~0.72)? If yes, Q captures chunk quality; if
~=vote, chunk choice given the state rarely matters (a diagnosis confirmation).

Chunks are passed in NORMALIZER space (action_tokenizer.normalizer['action']), matching the
geometry of OATPolicy._bon_select. Feature z-score stats are stored as buffers (self-contained
inference), exactly like TokenCountPredictor.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ChunkQ(nn.Module):
    def __init__(
        self,
        in_dim: int,            # per-step feature dim d (e.g. 138)
        n_obs_steps: int,       # To (e.g. 2)
        horizon: int,           # R scored steps of the chunk (e.g. 16 = n_action_steps)
        action_dim: int,        # D (e.g. 7)
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.action_dim = action_dim
        self.feat_flat = in_dim * n_obs_steps
        self.act_flat = horizon * action_dim
        self.flat_dim = self.feat_flat + self.act_flat

        # feature standardization (set via set_feature_stats); identity by default
        self.register_buffer("feat_mean", torch.zeros(self.feat_flat))
        self.register_buffer("feat_std", torch.ones(self.feat_flat))

        layers: List[nn.Module] = []
        prev = self.flat_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        """Store flattened (To*d,) standardization stats computed on the training split."""
        self.feat_mean.copy_(mean.reshape(-1))
        self.feat_std.copy_(std.reshape(-1).clamp_min(eps))

    def _prep_chunk(self, chunk_norm: torch.Tensor) -> torch.Tensor:
        """[B,R,D] (normalizer space) -> [B, horizon*D], padding/cropping to self.horizon."""
        a = chunk_norm[:, :self.horizon]
        if a.shape[1] < self.horizon:                       # pad short chunks with last step
            pad = a[:, -1:].expand(-1, self.horizon - a.shape[1], -1)
            a = torch.cat([a, pad], dim=1)
        return a.reshape(a.shape[0], -1)

    def forward(self, features: torch.Tensor, chunk_norm: torch.Tensor) -> torch.Tensor:
        """features [B,To,d], chunk_norm [B,R,D] (normalizer space) -> Q logit [B]."""
        B = features.shape[0]
        f = (features.reshape(B, -1) - self.feat_mean) / self.feat_std
        x = torch.cat([f, self._prep_chunk(chunk_norm)], dim=-1)
        return self.net(x).squeeze(-1)

    @torch.inference_mode()
    def score(self, features: torch.Tensor, chunk_norm: torch.Tensor) -> torch.Tensor:
        """-> success-prob value [B] in [0,1] (sigmoid of the logit)."""
        return torch.sigmoid(self.forward(features, chunk_norm))

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "ChunkQ":
        """Load a critic saved by scripts/train_chunk_q.py."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model
