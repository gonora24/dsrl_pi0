from typing import Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import ActorChunkTransformer
from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.constants import default_init


class ChunkedActionDistribution:
    """Wraps a flat action distribution and exposes (action_horizon, dsrl_action_dim) actions."""

    def __init__(
        self,
        base_dist: distrax.Distribution,
        action_horizon: int,
        dsrl_action_dim: int,
    ):
        self.distribution = base_dist.distribution
        self._base = base_dist
        self.action_horizon = action_horizon
        self.dsrl_action_dim = dsrl_action_dim

    def _reshape_action(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(*x.shape[:-1], self.action_horizon, self.dsrl_action_dim)

    def _flatten_action(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim >= 3:
            return x.reshape(*x.shape[:-2], -1)
        return x

    def sample(self, *, seed) -> jnp.ndarray:
        return self._reshape_action(self._base.sample(seed=seed))

    def sample_and_log_prob(self, *, seed) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, log_prob = self._base.sample_and_log_prob(seed=seed)
        return self._reshape_action(x), log_prob

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._base.log_prob(self._flatten_action(x))

    def mode(self) -> jnp.ndarray:
        return self._reshape_action(self._base.mode())

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None,
                 dsrl_action_dim: int = 32,
                 action_horizon: int = 1):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)
        self.dsrl_action_dim = dsrl_action_dim
        self.low = low
        self.high = high
        self.action_horizon = action_horizon
        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)
            
            def inverse_log_det_jacobian(y):
                high_ = jnp.broadcast_to(high, y.shape)
                low_ = jnp.broadcast_to(low, y.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            def inverse_rescale(y):
                x = (y - low) / (high - low)
                return 2. * x - 1.

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    inverse=inverse_rescale,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    inverse_log_det_jacobian=inverse_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
    
    def compute_marginalized_logprobs(self, means, log_stds, key):
        """Compute marginalized log probabilities for each action chunk independently."""
        batch = means.shape[0]
        single_action = self.dsrl_action_dim
        n_actions = means.shape[1] // single_action
        actions = jnp.empty_like(means)
        logprobs = jnp.empty((batch, n_actions))

        for i in range(n_actions):
            mean_ = means[:, i * single_action : (i + 1) * single_action]
            log_std_ = log_stds[
                :,
                i * single_action : (i + 1) * single_action,
            ]
            dist = TanhMultivariateNormalDiag(
                loc=mean_,
                scale_diag=jnp.exp(log_std_),
                low=self.low,
                high=self.high,
            )

            action, log_prob = dist.sample_and_log_prob(seed=key)
            actions = actions.at[:, i * single_action : (i + 1) * single_action].set(
                action
            )
            logprobs = logprobs.at[:, i].set(log_prob)
        if self.action_horizon > 1:
            actions = actions.reshape(batch, self.action_horizon, self.dsrl_action_dim)
        return actions, logprobs

class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None
    action_horizon: int = 1
    dsrl_action_dim: int = 32
    use_transformer: bool = False
    actor_transformer_n_heads: int = 4
    actor_transformer_n_layers: int = 3
    actor_transformer_weight_norm: bool = False
    marginalize_logprobs: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        if self.use_transformer:
            outputs = ActorChunkTransformer(
                n_embed=self.hidden_dims[-1],
                n_heads=self.actor_transformer_n_heads,
                n_layer=self.actor_transformer_n_layers,
                dropout_rate=self.dropout_rate,
                weight_norm=self.actor_transformer_weight_norm,
            )(observations, training=training)
        else:
            outputs = MLP(self.hidden_dims,
                          activate_final=True,
                          dropout_rate=self.dropout_rate)(observations,
                                                        training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
            low=self.low,
            high=self.high,
            dsrl_action_dim=self.dsrl_action_dim,
            action_horizon=self.action_horizon,
        )
        if self.action_horizon > 1 and not self.marginalize_logprobs:
            return ChunkedActionDistribution(
                distribution,
                self.action_horizon,
                self.dsrl_action_dim,
            ), means, log_stds
        return distribution, means, log_stds