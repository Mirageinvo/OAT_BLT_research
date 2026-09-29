import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.policy.kdpe import kdpe_endpoint_scores
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

    def set_chunk_q(self, critic):
        """Attach a frozen ChunkQ critic (Q-chunking QC analog). When set and
        `bon_signal='value'`, verifier-free best-of-N ranks the N candidate chunks by
        Q(features, chunk) and executes argmax_chunk Q instead of the consensus vote.
        Pass None to disable."""
        self.chunk_q = critic
        if critic is not None:
            critic.to(self.device).eval()
            for p in critic.parameters():
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

    @torch.inference_mode()
    def _bon_select(self, cand_b: torch.Tensor, R: int, bon_signal: str,
                    features_b: Optional[torch.Tensor] = None,
                    seq_logprob_b: Optional[torch.Tensor] = None,
                    sel_generator: Optional[torch.Generator] = None,
                    kdpe_bandwidth: float = 0.05) -> int:
        """Pick the best candidate index from [N, H, D] by a ranking signal over the executed
        prefix R (normalizer space).
        - 'vote' = mode-seeking KDE density (CS / paper primary)
        - 'medoid' = min sum of distances to the others
        - 'max_likelihood' = argmax sum token log-prob under sampling distribution
        - 'random' = Uniform{0..N-1} via sel_generator (does not touch global RNG)
        - 'kdpe' = KDPE-style endpoint KDE (OAT adaptation): density of the LAST executed
          action cand_b[:, R-1] only, raw (de-normalized) 7-D actions, SO(3)-aware kernel
        - 'value' = argmax ChunkQ (needs features_b)
        """
        N = cand_b.shape[0]
        if N == 1:
            return 0
        if bon_signal == 'random':
            if sel_generator is None:
                return int(torch.randint(0, N, (1,)).item())
            return int(torch.randint(0, N, (1,), generator=sel_generator).item())
        if bon_signal == 'max_likelihood':
            if seq_logprob_b is None:
                raise ValueError("max_likelihood requires seq_logprob_b [N]")
            # deterministic tie-break: lowest index
            return int(seq_logprob_b.argmax().item())
        if bon_signal == 'kdpe':
            # cand_b is already raw env actions (detokenize un-normalizes): no normalizer here.
            scores = kdpe_endpoint_scores(cand_b, execution_horizon=R, bandwidth=kdpe_bandwidth)
            return int(scores.argmax().item())
        norm = self.action_tokenizer.normalizer['action']
        a = norm.normalize(cand_b[:, :R])             # [N, R, D]
        if bon_signal == 'value' and getattr(self, 'chunk_q', None) is not None:
            feat = features_b.unsqueeze(0).expand(a.shape[0], -1, -1)   # [N, To, d]
            return int(self.chunk_q.score(feat, a).argmax())
        flat = a.reshape(cand_b.shape[0], -1)         # [N, R*D]
        dist = torch.cdist(flat, flat)                # [N, N]
        if bon_signal == 'medoid':
            # min sum L2 to others; diagonal 0 does not change argmin; tie -> min index
            return int(dist.sum(dim=1).argmin())
        off = dist[dist > 0]                          # 'vote' = mode-seeking KDE density
        sigma = (off.median() if off.numel() > 0 else dist.new_tensor(1.0)) + 1e-6
        dens = torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(dim=1)
        return int(dens.argmax())

    def _bon_sample(self, cond, n_new, temperature, topk, first_temp,
                    return_logprobs: bool = False):
        """Sample n_new tokens for each row of cond. If first_temp>0 (idea #3), draw the
        FIRST token (the 'mode' token) at first_temp to inject mode diversity, then continue
        the tail at the base temperature (clean refinement). OAT-unique: only an ordered
        token code has a 'first token' to target.
        Returns [B, n_new] or ([B, n_new], logprobs [B, n_new]) if return_logprobs."""
        B = cond.shape[0]
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
        if not (first_temp and first_temp > 0) or n_new < 1:
            out = self.model.generate(
                bos, cond=cond, max_new_tokens=n_new,
                temperature=temperature, top_k=topk, return_logprobs=return_logprobs,
            )
            if return_logprobs:
                tokens, lp = out
                return tokens[:, 1:], lp
            return out[:, 1:]
        # first-token temperature path: only used for exploration ablations; logprobs
        # under mixed T are not the paper max_likelihood score — refuse if requested.
        if return_logprobs:
            raise ValueError("return_logprobs unsupported with bon_first_temp > 0")
        t1 = self.model.generate(bos, cond=cond, max_new_tokens=1,
                                 temperature=first_temp, top_k=topk)        # [B, 2]
        if n_new == 1:
            return t1[:, 1:]
        out = self.model.generate(t1, cond=cond, max_new_tokens=n_new - 1,
                                  temperature=temperature, top_k=topk)      # [B, 1+n_new]
        return out[:, 1:]

    def predict_action_bon_free(self,
        obs_dict: Dict[str, torch.Tensor],
        bon_n: int = 8,
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        bon_signal: str = 'vote',
        bon_prefix_k: int = 0,
        bon_first_temp: float = 0.0,
        selector_seed: Optional[int] = None,
        kdpe_bandwidth: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Verifier-free best-of-N. Per env: encode vision ONCE (amortized), sample bon_n
        candidate plans from the cheap AR head, pick by a FREE consensus signal (see
        `_bon_select`). No trained verifier, no oracle -> ranks by consensus, a PROXY for
        quality.

        Coarse-to-fine (idea #2): if 0 < bon_prefix_k < use_k_tokens, sample + select on the
        cheap `bon_prefix_k`-token PREFIX (each prefix-decodes to the full chunk via
        prefix-decodability), then AR-refine the WINNER's tail up to use_k_tokens (one
        continuation, not N). Exploits k2~=k8: the mode lives in the first tokens, so spend
        the N-sample budget there and refine only the chosen mode. bon_prefix_k=0 -> flat BoN
        (sample full use_k_tokens for all N, original behaviour).
        """
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        coarse_to_fine = 0 < bon_prefix_k < use_k_tokens
        if bon_signal == 'kdpe' and (coarse_to_fine or bon_first_temp > 0):
            raise ValueError(
                "bon_signal='kdpe' is defined only for flat BoN: need bon_prefix_k=0 and "
                "bon_first_temp=0")
        gen_k = bon_prefix_k if coarse_to_fine else use_k_tokens
        need_lp = (bon_signal == 'max_likelihood')

        features = self.obs_encoder(obs_dict)        # [B, To, d]  (vision computed once)
        B = features.shape[0]
        feat_rep = features.repeat_interleave(bon_n, dim=0)   # [B*N, To, d]
        sampled = self._bon_sample(
            feat_rep, gen_k, temperature, topk, bon_first_temp,
            return_logprobs=need_lp,
        )
        if need_lp:
            tokens, tok_lp = sampled                     # [B*N, gen_k], [B*N, gen_k]
            seq_lp = tok_lp.sum(dim=-1).reshape(B, bon_n)  # [B, N]
        else:
            tokens = sampled
            seq_lp = None
        cand = self.action_tokenizer.detokenize(tokens, eval_keep_k=[gen_k] * (B * bon_n))
        H, Dd = cand.shape[1], cand.shape[2]
        cand = cand.reshape(B, bon_n, H, Dd)          # [B, N, H, D] (repeat_interleave layout)
        tokens = tokens.reshape(B, bon_n, gen_k)

        R = min(self.n_action_steps, H)               # rank over the executed prefix only
        sel_gen = None
        if bon_signal == 'random':
            # CPU generator — isolated from CUDA sampling RNG / torch default RNG
            sel_gen = torch.Generator(device='cpu')
            seed = 0 if selector_seed is None else int(selector_seed)
            # per-call uniqueness without touching global RNG: hash seed with batch id
            sel_gen.manual_seed(seed)

        best_idx = torch.tensor(
            [self._bon_select(
                cand[b], R, bon_signal, features[b],
                seq_logprob_b=(None if seq_lp is None else seq_lp[b]),
                sel_generator=sel_gen,
                kdpe_bandwidth=kdpe_bandwidth,
            ) for b in range(B)],
            device=self.device, dtype=torch.long)
        ar = torch.arange(B, device=self.device)
        chosen_tokens = tokens[ar, best_idx]          # [B, gen_k]

        if coarse_to_fine:
            # refine the winner only: AR-continue its prefix to the full budget (KV-cache
            # fills from the multi-token prefix), then decode at full fidelity
            bos1 = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
            prefix = torch.cat([bos1, chosen_tokens], dim=1)            # [B, 1+gen_k]
            full = self.model.generate(
                prefix, cond=features, max_new_tokens=use_k_tokens - gen_k,
                temperature=temperature, top_k=topk,
            )[:, 1:]                                                    # [B, use_k_tokens]
            chosen = self.action_tokenizer.detokenize(full, eval_keep_k=[use_k_tokens] * B)
        else:
            chosen = cand[ar, best_idx]                                 # [B, H, D]

        action = chosen[:, :self.n_action_steps]
        out = {'action': action, 'action_pred': chosen,
               'n_tokens': float(use_k_tokens), 'bon_n': float(bon_n),
               'bon_prefix_k': float(gen_k),
               'selected_idx': best_idx.detach().float().mean()}
        if seq_lp is not None:
            out['seq_logprob_selected'] = float(seq_lp[ar, best_idx].mean().item())
            out['seq_logprob_mean'] = float(seq_lp.mean().item())
            out['seq_logprob_std'] = float(seq_lp.std().item())
            out['seq_logprob_max'] = float(seq_lp.max().item())
        return out

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
        bon_free: int = 0,
        bon_signal: str = 'vote',
        bon_prefix_k: int = 0,
        bon_first_temp: float = 0.0,
        selector_seed: Optional[int] = None,
        kdpe_bandwidth: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        # verifier-free best-of-N: sample bon_free candidate plans (vision encoded ONCE,
        # amortized), pick by a free intrinsic signal (mode-seeking consensus / likelihood).
        # No trained verifier, no oracle. Takes precedence (it's an inference-time wrapper).
        if bon_free and bon_free > 1:
            return self.predict_action_bon_free(
                obs_dict, bon_n=bon_free, use_k_tokens=use_k_tokens,
                temperature=temperature, topk=topk, bon_signal=bon_signal,
                bon_prefix_k=bon_prefix_k, bon_first_temp=bon_first_temp,
                selector_seed=selector_seed, kdpe_bandwidth=kdpe_bandwidth,
            )
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

        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # full-budget generation (KV-cache), then prefix-decode for the R signal
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
        tokens = self.model.generate(
            bos, cond=features, max_new_tokens=K,
            temperature=temperature, top_k=topk,
        )[:, 1:]    # [B, K], drop <BOS>
        action_pred = self.action_tokenizer.detokenize(tokens, eval_keep_k=[K] * B)  # [B,Ta,D] raw

        # R is an executed-action-step count, bounded by the DECODED action horizon
        # (action_pred.shape[1], e.g. 32) -- NOT latent_horizon (=num token registers, 8).
        H = action_pred.shape[1]
        if r_max is None:
            r_max = H
        r_max = min(int(r_max), H)
        r_min = max(1, min(int(r_min), r_max))

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
        elif adaptive_r in ('pace', 'pace_raw'):
            # phase-aware (PACE, arXiv 2606.00537): replan at the first PROMINENT low-speed
            # valley of the decoded plan's speed profile. A valley = the plan slowing = a phase
            # transition (contact/grasp) = natural replan boundary. Reads the PLAN, not obs
            # (dodges the obs-wall). r_threshold is reused as the valley PROMINENCE threshold.
            #   'pace'     : speed = ||EE-delta||_{:6} in normalizer space (uniform per-dim scale,
            #                the analog of PACE's joint-space speed).
            #   'pace_raw' : speed = ||raw EE-translation||_{:3} (no normalizer) -- robustness
            #                variant ruling out the normalizer-space speed choice as the reason.
            from scipy.signal import find_peaks
            if adaptive_r == 'pace':
                norm = self.action_tokenizer.normalizer['action']
                speed = norm.normalize(action_pred)[:, :r_max, :6].norm(dim=-1)   # [B, r_max]
            else:
                speed = action_pred[:, :r_max, :3].norm(dim=-1)                   # raw translation
            sp = speed
            if r_max >= 3:                                   # smooth (window 3, edge-replicate)
                kernel = torch.ones(1, 1, 3, device=self.device) / 3.0
                sp = F.conv1d(F.pad(speed[:, None, :], (1, 1), mode='replicate'), kernel)[:, 0, :]
            sp_np = sp.detach().cpu().numpy()
            r_list = []
            for b in range(B):
                valleys, _ = find_peaks(-sp_np[b], prominence=float(r_threshold))
                r_list.append(int(valleys[0]) + 1 if len(valleys) else r_max)
            r_exec = torch.tensor(r_list, device=self.device, dtype=torch.long
                                  ).clamp(min=r_min, max=r_max)
            div_mean = speed.mean(dim=1)
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
