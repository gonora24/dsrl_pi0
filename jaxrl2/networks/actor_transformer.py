"""
AutoregressiveActorTransformer in JAX/Flax — ported from RLinf's
transformer_actor.py (PyTorch AutoregressiveActionTransformer).

Architecture: a standard transformer encoder (bidirectional self-attention,
no causal mask) used autoregressively by feeding only the tokens seen so far
at each step. At step t:
  - The sequence so far (context + previously sampled action tokens 0..t-1)
    is fed through the encoder blocks.
  - The last token's hidden state predicts mu and log_std for action t.
  - The action is sampled via the tanh-squashed Normal and appended.

Causality comes from the sequential construction, not from masking — matching
PyTorch's nn.TransformerEncoderLayer (bidirectional) used the same way in
RLinf's AutoregressiveActionTransformer.sample().

The autoregressive loop is implemented with jax.lax.scan over a pre-allocated
token buffer to keep the computation graph static.
"""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag


# ---------------------------------------------------------------------------
# Encoder block (matches nn.TransformerEncoderLayer defaults in PyTorch:
#   post-norm, relu activation, 4x FFN expansion, no causal mask)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Single transformer encoder block — bidirectional self-attention."""

    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Self-attention sub-layer (post-norm)
        residual = x
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout,
        )(x, x, deterministic=not training)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)

        # FFN sub-layer (post-norm, relu, 4x expansion)
        residual = x
        x = nn.Dense(4 * self.d_model)(x)
        x = jax.nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        x = nn.LayerNorm()(residual + x)
        return x


# ---------------------------------------------------------------------------
# Distribution wrapper
# ---------------------------------------------------------------------------

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

        self._context = context      # [B, 1, d_model]
        self._variables = variables  # frozen param dict passed from apply
        self._module = module        # AutoregressiveActorTransformer instance

    def sample_and_log_prob(self, *, seed):
        return self._module.ar_sample(
            self._variables, self._context, seed
        )


# ---------------------------------------------------------------------------
# Flax module
# ---------------------------------------------------------------------------

class AutoregressiveActorTransformer(nn.Module):
    """Autoregressive transformer actor that produces action chunks.

    The module is called via PixelMultiplexer, which passes in an observations
    dict where observations['pixels'] is already the encoded image latent.

    Returns an AutoregressiveDistribution whose sample_and_log_prob uses
    jax.lax.scan to autoregressively sample chunk_size actions.
    """

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

    def _forward_tokens(self, tokens: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Run token sequence through encoder blocks."""
        x = tokens
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def _head(self, h: jnp.ndarray):
        """Project hidden state to (mu, std) for one action."""
        out = self.out(h)                                       # [..., action_dim*2]
        mu, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, jnp.exp(log_std)

    def __call__(self, observations, training: bool = False):
        """Return an AutoregressiveDistribution.

        The non-autoregressive forward pass (context → single prediction) is
        used only to populate _loc/_scale_diag for logging.
        """
        image_features = observations['pixels']                 # [B, image_dim]
        state = observations['state']
        state_features = state.reshape(state.shape[0], -1)      # [B, state_dim]

        features = jnp.concatenate([state_features, image_features], axis=-1)
        context = self.context_proj(features)[:, None, :]       # [B, 1, d_model]

        # Non-AR single-step forward for logging
        h = self._forward_tokens(context, training=training)    # [B, 1, d_model]
        loc, scale = self._head(h[:, -1])                       # [B, action_dim]

        # Collect all module variables so ar_sample can apply sub-modules
        variables = self.variables

        return AutoregressiveDistribution(loc, scale, context, variables, self)

    def ar_sample(self, variables, context, rng):
        """Autoregressive sampling via jax.lax.scan.

        Uses a pre-allocated token buffer of shape [B, chunk_size+1, d_model].
        At step t the buffer holds [context, a_0, ..., a_{t-1}]; the new action
        token is written at position t+1.

        Args:
            variables: frozen param dict (passed from AutoregressiveDistribution)
            context:   [B, 1, d_model]
            rng:       JAX PRNGKey

        Returns:
            actions:   [B, chunk_size, action_dim]
            log_probs: [B]  (sum over chunk steps)
        """
        B = context.shape[0]
        T = self.chunk_size

        # Pre-allocate buffer; slot 0 = context token
        buf = jnp.zeros((B, T + 1, self.d_model))
        buf = buf.at[:, :1, :].set(context)

        def step(carry, t):
            buf, rng = carry

            # Tokens up to and including position t (positions 0..t)
            # We use dynamic slicing so scan sees a static shape.
            tokens = jax.lax.dynamic_slice_in_dim(buf, 0, t + 1, axis=1)  # [B, t+1, d_model]

            # Run through transformer (bound to the captured variables)
            h = self.apply(
                variables,
                tokens,
                method=self._forward_tokens,
                training=False,
            )
            h_last = h[:, -1]                                    # [B, d_model]
            mu, std = self.apply(variables, h_last, method=self._head)

            # Sample tanh-squashed Normal
            rng, subkey = jax.random.split(rng)
            dist = TanhMultivariateNormalDiag(
                loc=mu,
                scale_diag=std,
                low=jnp.array(self.low) if self.low is not None else None,
                high=jnp.array(self.high) if self.high is not None else None,
            )
            action = dist.sample(seed=subkey)                    # [B, action_dim]
            log_prob = dist.log_prob(action)                     # [B]

            # Embed action and write into buffer at position t+1
            new_token = self.apply(
                variables, action, method=self.action_proj
            )                                                    # [B, d_model]
            buf_new = jax.lax.dynamic_update_slice(
                buf,
                new_token[:, None, :],
                (0, t + 1, 0),
            )

            return (buf_new, rng), (action, log_prob)

        (_, _), (actions, log_probs) = jax.lax.scan(
            step,
            (buf, rng),
            jnp.arange(T),
        )
        # scan outputs: [T, B, ...]
        actions = jnp.transpose(actions, (1, 0, 2))             # [B, T, action_dim]
        log_probs = log_probs.sum(axis=0)                       # [B]
        return actions, log_probs
