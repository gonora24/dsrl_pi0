"""
AutoregressiveActorTransformer in JAX/Flax
"""
import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag

class CachedSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    n_embd: int
    n_head: int
    use_bias: bool
    dropout: float
    # residual_std: float  # init std for output projection: 0.02 / sqrt(2 * n_layer)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        kv_cache: tuple,
        cache_len: jnp.ndarray,
        training: bool = False,
        mask=None,
    ) -> jnp.ndarray:
        T = x.shape[-2]
        C = self.n_embd
        head_dim = C // self.n_head
        assert C % self.n_head == 0

        # Fused QKV projection: [..., T, C] -> [..., T, 3*C]
        qkv = nn.Dense(
            3 * C, use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(0.02),
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [..., T, C]

        # [..., T, C] -> [..., n_head, T, head_dim]
        def split_heads(t: jnp.ndarray) -> jnp.ndarray:
            return t.reshape(t.shape[:-1] + (self.n_head, head_dim)).swapaxes(-3, -2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        cache_k, cache_v = kv_cache
        update_idx = (0, 0, cache_len, 0)
        k_cache = jax.lax.dynamic_update_slice(cache_k, k, update_idx)
        v_cache = jax.lax.dynamic_update_slice(cache_v, v, update_idx)
        max_len = k_cache.shape[-2]

        # Scaled dot-product attention over full padded cache; mask invalid slots.
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("...qd,...kd->...qk", q, k_cache) * scale
        valid_mask = jnp.arange(max_len)[None, None, None, :] <= cache_len
        attn_weights = jnp.where(valid_mask, attn_weights, jnp.finfo(jnp.float32).min)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(jnp.float32).min)
        elif T > 1:
            causal_mask = jnp.tril(jnp.ones((T, max_len), dtype=jnp.bool_))
            attn_weights = jnp.where(causal_mask, attn_weights, jnp.finfo(jnp.float32).min)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=not training)

        y = jnp.einsum("...qk,...kd->...qd", attn_weights, v_cache)

        # [..., n_head, T, head_dim] -> [..., T, C]
        y = y.swapaxes(-3, -2).reshape(y.shape[:-3] + (T, C))

        y = nn.Dense(C, use_bias=self.use_bias)(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not training)
        return y, (k_cache, v_cache)


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
        kv_cache: tuple = None,
        cache_len: jnp.ndarray = None,
        mask=None,
    ) -> jnp.ndarray:
        residual = x
        x, kv = CachedSelfAttention(
            n_embd=self.d_model,
            n_head=self.n_heads,
            use_bias=True,
            dropout=self.dropout,
        )(x, kv_cache, cache_len, training=training, mask=mask)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)

        residual = x
        x = nn.Dense(4 * self.d_model)(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)
        return x, kv


# Distribution wrapper

class AutoregressiveDistribution:
    """Returned by AutoregressiveActorTransformer.__call__.

    Satisfies the interface expected by actor_updater.py and critic_updater.py:
      - sample_and_log_prob(seed) -> (actions [B,T,A], log_probs [B])
      - compute_marginalized_logprobs(means, log_stds, key) -> (actions [B,T,A], log_probs [B,T])
      - distribution._loc        -> [B, action_dim]  (first-token forward, logging only)
      - distribution._scale_diag -> [B, action_dim]  (first-token forward, logging only)

    ``means``/``log_stds`` from ``__call__`` are first-token predictions for logging.
    ``compute_marginalized_logprobs`` runs the full AR rollout and returns per-step log probs.
    """

    def __init__(self, loc, scale_diag, context, variables, module, training: bool = False):
        # Expose a fake .distribution so actor_updater can log mean/std.
        class _FakeDist:
            pass
        self.distribution = _FakeDist()
        self.distribution._loc = loc
        self.distribution._scale_diag = scale_diag

        self._context = context
        self._variables = variables
        self._module = module
        self._training = training
        self.low = module.low
        self.high = module.high
        self.action_dim = module.action_dim
        self.action_horizon = module.chunk_size

    def sample_and_log_prob(self, *, seed):
        return self._module.ar_sample(
            self._variables, self._context, seed, training=self._training
        )

    def sample_and_log_prob_diff(self, *, seed):
        return self._module.ar_sample_diff(
            self._variables, self._context, seed, training=self._training
        )

    def sample_and_log_prob_diff_mean(self, *, seed):
        return self._module.ar_sample_diff_mean(
            self._variables, self._context, seed, training=self._training
        )

    def sample(self, *, seed):
        actions, _ = self._module.ar_sample(
            self._variables, self._context, seed, training=self._training
        )
        return actions

    def compute_marginalized_logprobs(self, means, log_stds, key):
        """AR rollout with per-timestep log probs (not summed over the chunk)."""
        del means, log_stds  # per-step params come from the AR forward pass
        return self._module.ar_sample(
            self._variables,
            self._context,
            key,
            training=self._training,
            per_step_log_probs=True,
        )

class AutoregressiveActorTransformer(nn.Module):
    state_dim: int
    image_dim: int
    action_dim: int
    chunk_size: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    low: Optional[float] = None
    high: Optional[float] = None
    use_actor_diff_mean: bool = False
    residual_bound: float = 1.0
    residual_mean_bound: float = 0.3
    residual_log_std_bound: float = 2.0

    def setup(self):
        self.context_proj = nn.Dense(self.d_model, use_bias=False)
        # action_proj bleibt nn.Dense, wird aber als Modul-Methode
        # korrekt über einen dedizierten Wrapper aufgerufen (siehe unten).
        self.action_proj = nn.Dense(self.d_model)
        self.pos_embed = nn.Embed(self.chunk_size + 1, self.d_model)
        self.blocks = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        ]
        self.out = nn.Dense(self.action_dim * 2)
        self.residual_out = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        self.residual_mean_std_out = nn.Dense(self.action_dim * 2, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)

    def _empty_kv_cache(self, batch_size: int, dtype) -> tuple:
        """Pre-allocated per-layer (k, v) caches: [B, n_heads, chunk_size+1, head_dim]."""
        head_dim = self.d_model // self.n_heads
        max_seq_len = self.chunk_size + 1
        return tuple(
            (
                jnp.zeros((batch_size, self.n_heads, max_seq_len, head_dim), dtype=dtype),
                jnp.zeros((batch_size, self.n_heads, max_seq_len, head_dim), dtype=dtype),
            )
            for _ in range(self.n_layers)
        )

    def _forward_tokens(
        self,
        tokens: jnp.ndarray,
        training: bool = False,
        mask=None,
        kv_cache: tuple = None,
        cache_len: jnp.ndarray = None,
    ) -> jnp.ndarray:
        x = tokens
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, training=training, kv_cache=kv_cache[i], cache_len=cache_len, mask=mask)
            new_caches.append(kv)
        return x, tuple(new_caches), cache_len + 1

    # Dedizierter Wrapper, damit self.apply(..., method=...) korrekt
    # auf action_proj als Sub-Modul zugreift.
    def _embed_action(self, action: jnp.ndarray) -> jnp.ndarray:
        return self.action_proj(action)

    def _head(self, h: jnp.ndarray):
        out = self.out(h)
        mu, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std, jnp.exp(log_std)
    
    def _residual_head(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.residual_bound*jnp.tanh(self.residual_out(h))   # [B, action_dim]

    def _residual_mean_std_head(self, h: jnp.ndarray) -> jnp.ndarray:
        out = self.residual_mean_std_out(h)
        residual_mean, residual_log_std = jnp.split(out, 2, axis=-1)
        residual_mean = jnp.clip(residual_mean, -self.residual_mean_bound, self.residual_mean_bound)
        residual_log_std = jnp.clip(residual_log_std, -self.residual_log_std_bound, self.residual_log_std_bound)
        return residual_mean, residual_log_std   # [B, action_dim]

    def _pos_embed(self, idx: jnp.ndarray) -> jnp.ndarray:
        """Look up positional embedding(s). idx: integer or [K] array."""
        return self.pos_embed(idx)

    def __call__(self, observations, training: bool = False):
        image_features = observations['pixels']
        state = observations['state']
        state_features = state.reshape(state.shape[0], -1)

        features = jnp.concatenate([state_features, image_features], axis=-1)
        context = self.context_proj(features)[:, None, :]
        if self.use_actor_diff_mean:
            dummy_action = jnp.zeros(
                (context.shape[0], self.action_dim * 2),
                dtype=context.dtype
            )
        else:
            dummy_action = jnp.zeros(
                (context.shape[0], self.action_dim),
                dtype=context.dtype
            )
        _ = self._embed_action(dummy_action)

        # seq_embed = jnp.concatenate([context, dummy_action], axis=1)

        # Positional embeddings
        pos_indices = jnp.arange(context.shape[1])
        pos_embed = self.pos_embed(pos_indices)

        # Add positional embeddings
        x = context + pos_embed
        
        kv_cache = self._empty_kv_cache(context.shape[0], context.dtype)
        h, _, _ = self._forward_tokens(
            x, training=training, kv_cache=kv_cache, cache_len=jnp.int32(0)
        )
        residual = self._residual_head(h[:, -1])
        residual_mean, residual_std = self._residual_mean_std_head(h[:, -1])
        loc, log_std, scale = self._head(h[:, -1])

        variables = self.variables
        dist = AutoregressiveDistribution(loc, scale, x, variables, self, training=training)
        return dist, loc, log_std

    def ar_sample(
        self,
        variables,
        context,
        rng,
        training: bool = False,
        per_step_log_probs: bool = False,
    ):
        """Autoregressive sampling via jax.lax.scan."""
        B = context.shape[0]
        T = self.chunk_size

        buf = jnp.zeros((B, T + 1, self.d_model))
        buf = buf.at[:, :1, :].set(context)
        kv_cache = self._empty_kv_cache(B, context.dtype)
        cache_len = jnp.int32(0)

        def step(carry, t):
            buf, kv_cache, cache_len, rng = carry
            rng, dropout_rng, action_rng = jax.random.split(rng, 3)
            mask = jnp.arange(T + 1)[None, None, None, :] <= t
            mask = jnp.broadcast_to(mask, (B, 1, 1, T + 1))
            token = jax.lax.dynamic_slice(
                buf,
                start_indices=(0, t, 0),
                slice_sizes=(B, 1, self.d_model),
            )
            h, kv_cache, cache_len = self.apply(
                variables,
                tokens=token,
                method=self._forward_tokens,
                kv_cache=kv_cache,
                cache_len=cache_len,
                training=training,
                mask=mask,
                rngs={'dropout': dropout_rng},
            )
            h_last = h[:, -1, :]
            mu, _, std = self.apply(variables, h_last, method=self._head)

            dist = TanhMultivariateNormalDiag(
                loc=mu,
                scale_diag=std,
                low=jnp.array(self.low),
                high=jnp.array(self.high),
            )
            action, log_prob = dist.sample_and_log_prob(seed=action_rng)               # [B, action_dim]
            # log_prob = dist.log_prob(action)                     # [B]
            # Korrekte Methode für action embedding
            new_token = self.apply(
                variables, action, method=self._embed_action
            )       
            pos_embed = self.apply(variables, jnp.array([t + 1]), method=self._pos_embed)
            new_token = new_token + pos_embed[0]

            # start_indices als jnp.array für JAX-Tracer-Kompatibilität
            buf_new = jax.lax.dynamic_update_slice(
                buf,
                new_token[:, None, :],
                jnp.array([0, t + 1, 0]),                       # ← FIX
            )

            return (buf_new, kv_cache, cache_len, rng), (action, log_prob)

        (_, _, _, _), (actions, log_probs) = jax.lax.scan(
            step,
            (buf, kv_cache, cache_len, rng),
            jnp.arange(T),
        )
        actions = jnp.transpose(actions, (1, 0, 2))             # [B, T, action_dim]
        if per_step_log_probs:
            log_probs = jnp.transpose(log_probs, (1, 0))         # [B, T]
        else:
            log_probs = log_probs.sum(axis=0)                    # [B]
        return actions, log_probs

    def ar_sample_diff(
        self,
        variables,
        context,
        rng,
        training: bool = False,
        per_step_log_probs: bool = False,
    ):
        """Autoregressive difference predictor via jax.lax.scan.

        Step 0: sample a_0 ~ TanhNormal(_head(h)); record log_prob.
        Steps 1..T-1: residual r_t = _residual_head(h);
                      cumulative += r_t; log_prob contribution = 0.
        """
        B = context.shape[0]
        T = self.chunk_size

        buf = jnp.zeros((B, T + 1, self.d_model))
        buf = buf.at[:, :1, :].set(context)
        kv_cache = self._empty_kv_cache(B, context.dtype)
        cache_len = jnp.int32(0)
        cumulative_action = jnp.zeros((B, self.action_dim), dtype=context.dtype)

        def step(carry, t):
            buf, kv_cache, cache_len, rng, cumulative_action = carry
            rng, dropout_rng, action_rng = jax.random.split(rng, 3)

            mask = jnp.arange(T + 1)[None, None, None, :] <= t
            mask = jnp.broadcast_to(mask, (B, 1, 1, T + 1))
            token = jax.lax.dynamic_slice(
                buf,
                start_indices=(0, t, 0),
                slice_sizes=(B, 1, self.d_model),
            )
            h, kv_cache, cache_len = self.apply(
                variables,
                tokens=token,
                method=self._forward_tokens,
                kv_cache=kv_cache,
                cache_len=cache_len,
                training=training,
                mask=mask,
                rngs={'dropout': dropout_rng},
            )
            h_last = h[:, -1, :]

            # first step: full distribution sample + log_prob
            def first_step(_):
                mu, _, std = self.apply(variables, h_last, method=self._head)
                dist = TanhMultivariateNormalDiag(
                    loc=mu,
                    scale_diag=std,
                    low=jnp.array(self.low),
                    high=jnp.array(self.high),
                )
                action, lp = dist.sample_and_log_prob(seed=action_rng)
                return action, lp

            # subsequent steps: deterministic residual, zero log_prob
            def residual_step(_):
                residual = self.apply(variables, h_last, method=self._residual_head)
                new_cum = jnp.clip(cumulative_action + residual, self.low, self.high)
                return new_cum, jnp.zeros((B,), dtype=context.dtype)

            new_cumulative, step_log_prob = jax.lax.cond(
                t == 0,
                first_step,
                residual_step,
                None,
            )

            new_token = self.apply(variables, new_cumulative, method=self._embed_action)
            pos_embed = self.apply(variables, jnp.array([t + 1]), method=self._pos_embed)
            new_token = new_token + pos_embed[0]

            buf_new = jax.lax.dynamic_update_slice(
                buf,
                new_token[:, None, :],
                jnp.array([0, t + 1, 0]),
            )

            return (buf_new, kv_cache, cache_len, rng, new_cumulative), (new_cumulative, step_log_prob)

        (_, _, _, _, _), (actions, log_probs) = jax.lax.scan(
            step,
            (buf, kv_cache, cache_len, rng, cumulative_action),
            jnp.arange(T),
        )
        actions = jnp.transpose(actions, (1, 0, 2))         # [B, T, action_dim]
        if per_step_log_probs:
            log_probs = jnp.transpose(log_probs, (1, 0))   # [B, T]
        else:
            log_probs = log_probs.sum(axis=0)               # [B], only step-0 non-zero
        return actions, log_probs

    def ar_sample_diff_mean(
        self,
        variables,
        context,
        rng,
        training: bool = False,
        per_step_log_probs: bool = False,
    ):
        """Autoregressive mean predictor via jax.lax.scan.

        Step 0: sample a_0 ~ TanhNormal(_head(h)); record log_prob.
        Steps 1..T-1: residual r_t = _residual_head(h);
                      cumulative += r_t; log_prob contribution = 0.
        """
        B = context.shape[0]
        T = self.chunk_size

        buf = jnp.zeros((B, T + 1, self.d_model))
        buf = buf.at[:, :1, :].set(context)
        kv_cache = self._empty_kv_cache(B, context.dtype)
        cache_len = jnp.int32(0)
        new_params = jnp.zeros((B, self.action_dim*2), dtype=context.dtype)

        def step(carry, t):
            buf, kv_cache, cache_len, rng, new_params = carry
            rng, dropout_rng = jax.random.split(rng, 2)

            mask = jnp.arange(T + 1)[None, None, None, :] <= t
            mask = jnp.broadcast_to(mask, (B, 1, 1, T + 1))
            token = jax.lax.dynamic_slice(
                buf,
                start_indices=(0, t, 0),
                slice_sizes=(B, 1, self.d_model),
            )

            h, kv_cache, cache_len = self.apply(
                variables,
                tokens=token,
                method=self._forward_tokens,
                kv_cache=kv_cache,
                cache_len=cache_len,
                training=training,
                mask=mask,
                rngs={'dropout': dropout_rng},
            )
            h_last = h[:, -1, :]

            # first step: get mean and std
            def first_step(_):
                mu, log_std, _ = self.apply(variables, h_last, method=self._head)
                return jnp.concatenate([mu, log_std], axis=-1)

            # subsequent steps: get mean and std of residual
            def residual_step(_):
                residual_mean, residual_log_std = self.apply(variables, h_last, method=self._residual_mean_std_head)
                mu, log_std = jnp.split(new_params, 2, axis=-1)
                mu = mu + residual_mean
                log_std = log_std + residual_log_std
                log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
                return jnp.concatenate([mu, log_std], axis=-1)

            new_params = jax.lax.cond(
                t == 0,
                first_step,
                residual_step,
                None,
            )
            new_token = self.apply(variables, new_params, method=self._embed_action)

            pos_embed = self.apply(variables, jnp.array([t + 1]), method=self._pos_embed)
            new_token = new_token + pos_embed[0]

            buf_new = jax.lax.dynamic_update_slice(
                buf,
                new_token[:, None, :],
                jnp.array([0, t + 1, 0]),
            )

            return (buf_new, kv_cache, cache_len, rng, new_params), (new_params)

        (_, _, _, _, _), (params) = jax.lax.scan(
            step,
            (buf, kv_cache, cache_len, rng, new_params),
            jnp.arange(T),
        )
        params = jnp.split(params, 2, axis=-1)
        dist = TanhMultivariateNormalDiag(
            loc=params[0],
            scale_diag=jnp.exp(params[1]),
            low=jnp.array(self.low),
            high=jnp.array(self.high),
        )
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        actions = jnp.transpose(actions, (1, 0, 2))         # [B, T, action_dim]
        if per_step_log_probs:
            log_probs = jnp.transpose(log_probs, (1, 0))   # [B, T]
        else:
            log_probs = log_probs.sum(axis=0)               # [B], only step-0 non-zero
        return actions, log_probs, params[0], params[1]