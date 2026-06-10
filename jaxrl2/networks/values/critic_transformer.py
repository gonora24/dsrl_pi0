"""
CriticGPT in JAX/Flax — ported from RLinf's critic_transformer.py (PyTorch).

Adapted from the NanoGPT implementation (https://github.com/karpathy/nanoGPT).
Main adaptations:
1. Translate from PyTorch to Flax linen
2. Replace language-model input with (state, image, action-sequence) input
3. First output token is a V-value; remaining T tokens are per-step Q-values
"""
import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    n_embd: int
    n_head: int
    use_bias: bool
    dropout: float
    residual_std: float  # init std for output projection: 0.02 / sqrt(2 * n_layer)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
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

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("...qd,...kd->...qk", q, k) * scale
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_weights = jnp.where(causal_mask, attn_weights, jnp.finfo(jnp.float32).min)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=not training)

        y = jnp.einsum("...qk,...kd->...qd", attn_weights, v)  # [..., n_head, T, head_dim]

        # [..., n_head, T, head_dim] -> [..., T, C]
        y = y.swapaxes(-3, -2).reshape(y.shape[:-3] + (T, C))

        # Output projection with scaled residual init
        y = nn.Dense(
            C, use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(self.residual_std),
        )(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=not training)
        return y


class GPTMLP(nn.Module):
    """Position-wise feed-forward block (4x expansion, GELU activation)."""

    n_embd: int
    use_bias: bool
    dropout: float
    residual_std: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(
            4 * self.n_embd, use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(0.02),
        )(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(
            self.n_embd, use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(self.residual_std),
        )(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        return x


class GPTBlock(nn.Module):
    """Pre-norm (or no-norm) transformer block."""

    n_embd: int
    n_head: int
    use_bias: bool
    dropout: float
    use_layer_norm: bool
    residual_std: float

    def setup(self):
        self.attn = CausalSelfAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            use_bias=self.use_bias,
            dropout=self.dropout,
            residual_std=self.residual_std,
        )
        self.mlp = GPTMLP(
            n_embd=self.n_embd,
            use_bias=self.use_bias,
            dropout=self.dropout,
            residual_std=self.residual_std,
        )
        if self.use_layer_norm:
            self.ln_1 = nn.LayerNorm(use_bias=self.use_bias)
            self.ln_2 = nn.LayerNorm(use_bias=self.use_bias)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_layer_norm:
            x = x + self.attn(self.ln_1(x), training=training)
            x = x + self.mlp(self.ln_2(x), training=training)
        else:
            x = x + self.attn(x, training=training)
            x = x + self.mlp(x, training=training)
        return x


class CriticGPT(nn.Module):
    """
    Causal-transformer Q/V critic (GPT-style).

    Encodes a (state, image) pair as a single context token prepended to T
    action tokens, then applies a causal transformer to produce per-token
    value heads.

    Output shape: [..., 1 + T, num_q_heads]
      - token 0  -> V-value (no actions consumed)
      - tokens 1..T -> Q-values for each action step

    Mirrors the PyTorch CriticGPT in RLinf/rlinf/models/embodiment/modules/critic_transformer.py.
    """

    state_dim: int
    image_dim: int
    action_dim: int
    action_horizon: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    use_layer_norm: bool
    use_bias: bool
    relative_pos: bool
    num_q_heads: int = 1

    def setup(self):
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)

        self.context_proj = nn.Dense(
            self.n_embd, use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
        )
        self.action_encoder = nn.Dense(
            self.n_embd, use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
        )
        self.pos_enc = nn.Embed(
            num_embeddings=self.action_horizon + 1,
            features=self.n_embd,
            embedding_init=nn.initializers.normal(0.02),
        )
        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [
            GPTBlock(
                n_embd=self.n_embd,
                n_head=self.n_head,
                use_bias=self.use_bias,
                dropout=self.dropout,
                use_layer_norm=self.use_layer_norm,
                residual_std=residual_std,
            )
            for _ in range(self.n_layer)
        ]
        if self.use_layer_norm:
            self.ln_f = nn.LayerNorm(use_bias=self.use_bias)
        self.output_layer = nn.Dense(
            self.num_q_heads, use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
        )

    def __call__(
        self,
        state_features: jnp.ndarray,
        image_features: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        idx_s: Optional[jnp.ndarray] = None,
        idx_a: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Args:
            state_features:  [..., state_dim]
            image_features:  [..., image_dim]
            actions:         [..., T, action_dim]  or None
            idx_s:           [...] state time-step index (only when relative_pos=False)
            idx_a:           [..., T] action time-step indices (only when relative_pos=False)
            training:        enable dropout

        Returns:
            [..., 1 + T, num_q_heads]
        """
        t = 0 if actions is None else actions.shape[-2]
        assert t + 1 <= self.action_horizon + 1

        # Context token: (state, image) -> [..., 1, n_embd]
        context_feat = jnp.concatenate([state_features, image_features], axis=-1)
        context_emb = self.context_proj(context_feat)[..., None, :]  # [..., 1, n_embd]

        if actions is not None:
            action_emb = self.action_encoder(actions)               # [..., T, n_embd]
            seq_emb = jnp.concatenate([context_emb, action_emb], axis=-2)  # [..., 1+T, n_embd]
        else:
            seq_emb = context_emb                                   # [..., 1, n_embd]

        # Positional embeddings
        if self.relative_pos:
            pos = jnp.arange(1 + t)  # [1+T]
        else:
            if idx_s is None:
                raise ValueError("idx_s is required when relative_pos=False")
            if actions is not None and idx_a is None:
                raise ValueError("idx_a is required when relative_pos=False and actions is not None")
            if actions is not None:
                pos = jnp.concatenate([idx_s[..., None], idx_a], axis=-1)  # [..., 1+T]
            else:
                pos = idx_s[..., None]                                       # [..., 1]

        pos_emb = self.pos_enc(pos)  # [1+T, n_embd] or [..., 1+T, n_embd]

        x = self.drop(seq_emb + pos_emb, deterministic=not training)

        for block in self.blocks:
            x = block(x, training=training)

        if self.use_layer_norm:
            x = self.ln_f(x)

        x = self.output_layer(x)  # [..., 1+T, num_q_heads]
        return x


class CriticGPTEnsemble(nn.Module):
    """Ensemble of independent CriticGPT networks via nn.vmap."""

    state_dim: int
    image_dim: int
    action_dim: int
    action_horizon: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    use_layer_norm: bool
    use_bias: bool
    relative_pos: bool
    num_q_heads: int = 1
    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        state_features: jnp.ndarray,
        image_features: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        idx_s: Optional[jnp.ndarray] = None,
        idx_a: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Returns:
            [num_qs, ..., 1 + T, num_q_heads]
        """
        VmapCritic = nn.vmap(
            CriticGPT,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return VmapCritic(
            state_dim=self.state_dim,
            image_dim=self.image_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            use_bias=self.use_bias,
            relative_pos=self.relative_pos,
            num_q_heads=self.num_q_heads,
        )(state_features, image_features, actions, idx_s, idx_a, training)
