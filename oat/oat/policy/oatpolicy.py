import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
# from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class OATPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
    ):
        super().__init__()
        
        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # create AR model
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        model = AutoregressiveModel(
            vocab_size=codebook_size + 1,  # +1 for <BOS>
            max_seq_len=latent_horizon + 1,
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size  # last token id for <BOS>

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        # optional learned token-count predictor (attached post-hoc, not a submodule
        # so it is excluded from the policy's own checkpoint / training)
        self.token_predictor = None
        # optional obs-agnostic budget mixture (baseline for the adaptivity gate):
        # per-sample k ~ Categorical(agnostic_k_probs) over k=1..max_seq_len
        self.agnostic_k_probs = None

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'oatpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        # self.action_tokenizer.set_normalizer(normalizer)

    def set_token_predictor(self, predictor):
        """Attach a frozen TokenCountPredictor. When set, `predict_action_adaptive`
        uses it (predicts per-observation token budget) instead of the entropy
        threshold. Pass None to revert to entropy-threshold mode."""
        self.token_predictor = predictor
        if predictor is not None:
            predictor.to(self.device).eval()
            for p in predictor.parameters():
                p.requires_grad_(False)

    def set_agnostic_mix(self, k_probs):
        """Attach an obs-agnostic budget mixture (adaptivity-gate baseline). When set,
        `predict_action_adaptive` samples each sample's budget k ~ Categorical(k_probs),
        ignoring the observation, instead of predicting it. Takes precedence over the
        token predictor. Pass None to disable.
        k_probs: dict {k: prob} or list/tensor over k=1..max_seq_len (renormalized)."""
        if k_probs is None:
            self.agnostic_k_probs = None
            return
        probs = torch.zeros(self.max_seq_len)
        if isinstance(k_probs, dict):
            for k, p in k_probs.items():
                probs[int(k) - 1] = float(p)
        else:
            vals = torch.as_tensor(k_probs, dtype=torch.float)
            probs[:len(vals)] = vals
        self.agnostic_k_probs = (probs / probs.sum()).to(self.device)

    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        encoder_decay_params = []
        encoder_nodecay_params = []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                encoder_decay_params.append(param)
            else:
                encoder_nodecay_params.append(param)

        policy_decay_params = []
        policy_nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                policy_decay_params.append(param)
            else:
                policy_nodecay_params.append(param)
        
        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # autoregressive generation
        action_tokens = torch.full( # [B, 1] seq: [<BOS>,]
            (B, 1), self.bos_id, 
            dtype=torch.long, device=self.device
        )
        action_tokens = self.model.generate(
            action_tokens,
            cond=features,
            max_new_tokens=use_k_tokens,
            temperature=temperature,
            top_k=topk,
        )[:, 1:]    # [B, max_seq_len], drop <BOS>

        # decode action tokens
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
            )

        # receeding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def predict_action_adaptive(self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        entropy_threshold: float = 2.75,
        adaptive_r: Optional[str] = None,
        r_coarse_k: int = 4,
        r_min: int = 8,
        r_max: Optional[int] = None,
        r_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        # variable-R execution (GATE 1): hold K fixed (= use_k_tokens, default full
        # budget) and adapt the executed chunk length R per observation. Distinct axis
        # from the K-budget modes below, so it takes precedence when requested.
        if adaptive_r is not None:
            return self.predict_action_variable_r(
                obs_dict, use_k_tokens=use_k_tokens,
                temperature=temperature, topk=topk,
                adaptive_r=adaptive_r, r_coarse_k=r_coarse_k,
                r_min=r_min, r_max=r_max, r_threshold=r_threshold,
            )
        # obs-agnostic budget mixture (gate baseline) takes precedence if attached
        if getattr(self, 'agnostic_k_probs', None) is not None:
            return self.predict_action_agnostic(
                obs_dict, use_k_tokens=use_k_tokens,
                temperature=temperature, topk=topk,
            )
        # if a learned token-count predictor is attached, use it instead of the
        # entropy threshold (no per-step entropy check, KV-cache fast generation)
        if getattr(self, 'token_predictor', None) is not None:
            return self.predict_action_predictor(
                obs_dict, use_k_tokens=use_k_tokens,
                temperature=temperature, topk=topk,
            )

        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        action_tokens = torch.full(
            (B, 1), self.bos_id,
            dtype=torch.long, device=self.device,
        )
        entropies = []
        for step in range(use_k_tokens):
            logits = self.model(action_tokens, cond=features)        # [B, T, vocab]
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)  # [B, vocab]

            probs = F.softmax(next_logits, dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(-1).mean()
            entropy_val = entropy.item()
            entropies.append(entropy_val)

            if topk is not None:
                v, _ = torch.topk(next_logits, min(topk, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float('-inf'))
                probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)     # [B, 1]
            action_tokens = torch.cat([action_tokens, next_token], dim=1)

            if step >= 1 and entropy_val < entropy_threshold:
                break

        action_tokens = action_tokens[:, 1:]   # drop <BOS>; detokenize pads to latent_horizon
        n_tokens = action_tokens.shape[1]

        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(tokens=action_tokens)

        action = action_pred[:, :self.n_action_steps]

        return {
            'action': action,
            'action_pred': action_pred,
            'n_tokens': n_tokens,
            'entropies': entropies,
        }

    @torch.inference_mode()
    def _generate_at_budgets(self, features, k_pred, temperature, topk):
        """Given per-sample budgets k_pred [B], generate max(k_pred) tokens for the batch
        (KV-cache) and decode each sample at its own k. Returns (action, action_pred)."""
        B = features.shape[0]
        K = int(k_pred.max().item())
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
        action_tokens = self.model.generate(
            bos, cond=features, max_new_tokens=K,
            temperature=temperature, top_k=topk,
        )[:, 1:]    # [B, K], drop <BOS>
        action_pred = self.action_tokenizer.detokenize(
            tokens=action_tokens,
            eval_keep_k=k_pred.tolist(),
        )
        action = action_pred[:, :self.n_action_steps]
        return action, action_pred

    @torch.inference_mode()
    def predict_action_predictor(self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Adaptive generation driven by the attached `token_predictor`:
        predict a per-observation budget k, generate that many tokens (KV-cache),
        and decode each sample at its own k. `use_k_tokens` optionally caps the budget."""
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk
        cap = self.max_seq_len if use_k_tokens is None else min(use_k_tokens, self.max_seq_len)

        features = self.obs_encoder(obs_dict)   # [B, To, d]
        k_pred = self.token_predictor.predict_k(features).clamp(min=1, max=cap)  # [B]
        action, action_pred = self._generate_at_budgets(features, k_pred, temperature, topk)

        return {
            'action': action,
            'action_pred': action_pred,
            'n_tokens': float(k_pred.float().mean().item()),  # mean budget over batch
            'k_pred': k_pred,
        }

    @torch.inference_mode()
    def predict_action_agnostic(self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Obs-agnostic budget mixture (adaptivity-gate baseline): per-sample
        k ~ Categorical(self.agnostic_k_probs), ignoring the observation. Same marginal
        budget distribution as the predictor but no obs-conditioning."""
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk
        cap = self.max_seq_len if use_k_tokens is None else min(use_k_tokens, self.max_seq_len)

        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]
        # sample budgets i.i.d. from the fixed categorical (k = index + 1)
        k_pred = (torch.multinomial(self.agnostic_k_probs, B, replacement=True) + 1).clamp(min=1, max=cap)
        action, action_pred = self._generate_at_budgets(features, k_pred, temperature, topk)

        return {
            'action': action,
            'action_pred': action_pred,
            'n_tokens': float(k_pred.float().mean().item()),
            'k_pred': k_pred,
        }

    @torch.inference_mode()
    def predict_action_variable_r(self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        adaptive_r: str = 'convergence',
        r_coarse_k: int = 4,
        r_min: int = 8,
        r_max: Optional[int] = None,
        r_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Variable executed-chunk-length (R) generation for the R-axis gate (GATE 1).

        Generates a single full-budget chunk (K = use_k_tokens, default max_seq_len) and
        picks, per observation, how many leading actions R to execute open-loop before the
        next replan. The token budget K is held FIXED -- this isolates the R axis so SR-vs-R
        is directly comparable to the fixed-R sweep.

        R-signal modes:
          'convergence' : decode the SAME tokens at a coarse budget (r_coarse_k) and at K;
                          R = length of the leading prefix where the coarse and fine action
                          plans agree (per-timestep L2 divergence in normalizer space below
                          r_threshold). Stable plan -> long R; early divergence -> short R.
                          Generation-aware (reads the decoded plan, not the obs).
          'random'      : R ~ Uniform{r_min..r_max} (control: variance in R uncorrelated with
                          obs; must NOT beat fixed-R at matched mean if obs-conditioning is the
                          source of any gain).
          'fixed'       : R = r_max for every sample (sanity == fixed-R sweep at R=r_max).

        Returns the FULL r_max-length chunk plus per-sample `r_exec`; the runner NaN-pads
        beyond each sample's R so MultiStepWrapper executes only that prefix.
        """
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk
        K = self.max_seq_len if use_k_tokens is None else min(use_k_tokens, self.max_seq_len)
        H = self.action_tokenizer.latent_horizon
        if r_max is None:
            r_max = H
        r_max = min(int(r_max), H)
        r_min = max(1, min(int(r_min), r_max))

        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # full-budget generation (KV-cache), then prefix-decode for the R signal
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
        tokens = self.model.generate(
            bos, cond=features, max_new_tokens=K,
            temperature=temperature, top_k=topk,
        )[:, 1:]    # [B, K], drop <BOS>
        action_pred = self.action_tokenizer.detokenize(tokens, eval_keep_k=[K] * B)  # [B,H,D] raw

        div_mean = torch.zeros(B, device=self.device)
        if adaptive_r == 'fixed':
            r_exec = torch.full((B,), r_max, dtype=torch.long, device=self.device)
        elif adaptive_r == 'random':
            r_exec = torch.randint(r_min, r_max + 1, (B,), device=self.device)
        elif adaptive_r == 'convergence':
            k_coarse = max(1, min(int(r_coarse_k), K))
            action_coarse = self.action_tokenizer.detokenize(tokens, eval_keep_k=[k_coarse] * B)
            norm = self.action_tokenizer.normalizer['action']
            d = (norm.normalize(action_pred) - norm.normalize(action_coarse)).norm(dim=-1)  # [B,H]
            d = d[:, :r_max]
            exceed = d > r_threshold                       # [B, r_max]
            first = exceed.float().argmax(dim=1)           # first exceed idx (0 if none)
            r_exec = torch.where(
                exceed.any(dim=1), first,
                torch.full_like(first, r_max),
            ).clamp(min=r_min, max=r_max).long()
            div_mean = d.mean(dim=1)
        else:
            raise ValueError(f"unknown adaptive_r mode: {adaptive_r}")

        action = action_pred[:, :r_max]
        return {
            'action': action,
            'action_pred': action_pred,
            'n_tokens': float(K),
            'r_exec': r_exec,
            'div_mean': float(div_mean.mean().item()),
        }

    def forward(self, batch) -> torch.Tensor:
        # tokenize trajectory
        with torch.inference_mode():
            action_tokens = self.action_tokenizer.tokenize(batch['action'])

        B = batch['action'].shape[0]
        device = batch['action'].device

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id, 
                dtype=torch.long, device=device
            ),
            action_tokens
        ], dim=1)

        # forward model
        logits = self.model(action_tokens[:, :-1], cond=features)

        # compute loss
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),     # (B*T, vocab_size)
            action_tokens[:, 1:].reshape(-1)    # (B*T,)
        )
        return loss
