"""
AutoregressiveActorTransformer in JAX/Flax
"""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False, mask=None) -> jnp.ndarray:
        residual = x
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
        )(x, x, mask=mask, deterministic=not training)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)

        residual = x
        x = nn.Dense(4 * self.d_model)(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)
        return x


# Distribution wrapper

class AutoregressiveDistribution:
    """Returned by AutoregressiveActorTransformer.__call__.

    Satisfies the interface expected by actor_updater.py and critic_updater.py:
      - sample_and_log_prob(seed) -> (actions [B,T,A], log_probs [B])
      - distribution._loc        -> [B, action_dim]  (non-AR forward, logging only)
      - distribution._scale_diag -> [B, action_dim]  (non-AR forward, logging only)
    """

    def __init__(self, loc, scale_diag, context, variables, module):
        # Expose a fake .distribution so actor_updater can log mean/std.
        class _FakeDist:
            pass
        self.distribution = _FakeDist()
        self.distribution._loc = loc
        self.distribution._scale_diag = scale_diag

        self._context = context
        self._variables = variables
        self._module = module

    def sample_and_log_prob(self, *, seed):
        return self._module.ar_sample(
            self._variables, self._context, seed
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

    def setup(self):
        self.context_proj = nn.Dense(self.d_model, use_bias=False)
        # action_proj bleibt nn.Dense, wird aber als Modul-Methode
        # korrekt über einen dedizierten Wrapper aufgerufen (siehe unten).
        self.action_proj = nn.Dense(self.d_model)
        self.blocks = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        ]
        self.out = nn.Dense(self.action_dim * 2)

    def _forward_tokens(
        self, tokens: jnp.ndarray, training: bool = False, mask=None
    ) -> jnp.ndarray:
        x = tokens
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        return x

    # Dedizierter Wrapper, damit self.apply(..., method=...) korrekt
    # auf action_proj als Sub-Modul zugreift.
    def _embed_action(self, action: jnp.ndarray) -> jnp.ndarray:
        return self.action_proj(action)

    def _head(self, h: jnp.ndarray):
        out = self.out(h)
        mu, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, jnp.exp(log_std)

    def __call__(self, observations, training: bool = False):
        image_features = observations['pixels']
        state = observations['state']
        state_features = state.reshape(state.shape[0], -1)

        features = jnp.concatenate([state_features, image_features], axis=-1)
        context = self.context_proj(features)[:, None, :]
        dummy_action = jnp.zeros(
            (context.shape[0], self.action_dim),
            dtype=context.dtype
        )
        _ = self._embed_action(dummy_action)
        
        h = self._forward_tokens(context, training=training)
        loc, scale = self._head(h[:, -1])

        variables = self.variables
        return AutoregressiveDistribution(loc, scale, context, variables, self)

    def ar_sample(self, variables, context, rng):
        """Autoregressive sampling via jax.lax.scan."""
        B = context.shape[0]
        T = self.chunk_size

        buf = jnp.zeros((B, T + 1, self.d_model))
        buf = buf.at[:, :1, :].set(context)

        def step(carry, t):
            buf, rng = carry

            # Korrekte Attention-Maske für Flax MHA.
            # Shape muss [B, n_heads, seq_len, seq_len] sein.
            # True = Position darf attended werden, False = wird maskiert.
            # Wir maskieren alle Key-Positionen > t (noch nicht geschrieben).
            seq_idx = jnp.arange(T + 1)                         # [T+1]
            key_mask = (seq_idx <= t)                            # [T+1], bool
            # Broadcast zu [B, n_heads, T+1, T+1]:
            # Alle Query-Positionen sehen dieselbe Key-Maske.
            mask = jnp.broadcast_to(
                key_mask[None, None, None, :],
                (B, self.n_heads, T + 1, T + 1),
            )

            h = self.apply(
                variables,
                buf,
                method=self._forward_tokens,
                training=False,
                mask=mask,
            )
            h_last = jax.lax.dynamic_index_in_dim(
                h, t, axis=1, keepdims=False
            )                                                    # [B, d_model]
            mu, std = self.apply(variables, h_last, method=self._head)

            rng, subkey = jax.random.split(rng)
            dist = TanhMultivariateNormalDiag(
                loc=mu,
                scale_diag=std,
                low=jnp.array(self.low) if self.low is not None else None,
                high=jnp.array(self.high) if self.high is not None else None,
            )
            action = dist.sample(seed=subkey)                    # [B, action_dim]
            log_prob = dist.log_prob(action)                     # [B]

            # Korrekte Methode für action embedding
            new_token = self.apply(
                variables, action, method=self._embed_action
            )                                                    # [B, d_model]

            # start_indices als jnp.array für JAX-Tracer-Kompatibilität
            buf_new = jax.lax.dynamic_update_slice(
                buf,
                new_token[:, None, :],
                jnp.array([0, t + 1, 0]),                       # ← FIX
            )

            return (buf_new, rng), (action, log_prob)

        (_, _), (actions, log_probs) = jax.lax.scan(
            step,
            (buf, rng),
            jnp.arange(T),
        )
        actions = jnp.transpose(actions, (1, 0, 2))             # [B, T, action_dim]
        log_probs = log_probs.sum(axis=0)                        # [B]
        return actions, log_probs