from typing import Callable, Optional, Sequence, Union
from flax.core import frozen_dict

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            # if k == "actions":
            #     v = v[:, 0:1, ...]
            if k == 'state': # flatten action chunk to 1D
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                # v = jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])])
            elif k == 'prev_action' or k == 'actions':
                if v.ndim > 2:
                    # deal with action chunk
                    obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                else:
                    obs.append(v)
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x

def _flatten_dict_special(x):
    if hasattr(x, 'values'):
        obs = []
        action = None
        for k, v in sorted(x.items()):
            if k == 'state' or k == 'prev_action':
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
            elif k == 'actions':
                print ('action shape: ', v.shape)
                action = v
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1), action
    else:
        return x
        

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        # print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            # print('post fc size', x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class MLPActionSep(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        x, action = _flatten_dict_special(x)
        print ('mlp action sep state post flatten', x.shape)
        print ('mlp action sep action post flatten', action.shape)

        for i, size in enumerate(self.hidden_dims):
            x_used = jnp.concatenate([x, action], axis=-1)
            x = nn.Dense(size, kernel_init=default_init())(x_used)
            print ('FF layers: ', x_used.shape, x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    A module to perform multi-head attention using Flax's linen library.
    This combines multiple attention heads into a single operation.
    """

    num_heads: int
    n_embed: int
    dropout_rate: float
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training):
        """
        Apply multi-head attention to the input tensor.

        Parameters:
            x (tensor): Input tensor.
            training (bool): Flag to indicate if the model is training (affects dropout).

        Returns:
            tensor: Output tensor after applying multi-head attention and a dense layer.
        """
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x)
        if self.weight_norm:
            x = nn.WeightNorm(nn.Dense(self.n_embed))(x)
        else:
            x = nn.Dense(self.n_embed)(x)
        return x


class FeedForward(nn.Module):
    """
    A feedforward neural network module using Flax's Linen API with two dense layers
    and a dropout layer for regularization.
    """

    n_embed: int
    dropout_rate: float
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training):
        """
        Applies a sequence of layers to the input tensor.

        Parameters:
            x (tensor): Input tensor to the feedforward network.
            training (bool): Flag to indicate if the model is training.

        Returns:
            tensor: The output tensor after processing through dense and dropout layers.
        """
        if self.weight_norm:
            x = nn.Sequential(
                [
                    nn.WeightNorm(nn.Dense(4 * self.n_embed)),
                    nn.gelu,
                    nn.WeightNorm(nn.Dense(self.n_embed)),
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                ]
            )(x)
        else:
            x = nn.Sequential(
                [
                    nn.Dense(4 * self.n_embed),
                    nn.gelu,
                    nn.Dense(self.n_embed),
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                ]
            )(x)
        return x


class Block(nn.Module):
    """
    A transformer block module using Flax's linen API, which integrates multi-head attention
    and feedforward neural network layers.
    """

    n_embed: int
    n_heads: int
    dropout_rate: float
    weight_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        """
        Process the input tensor through the transformer block.

        Parameters:
            x (tensor): Input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tensor: The output tensor after processing through the transformer block.
        """

        # Initialize the MultiHeadAttention and FeedForward modules
        sa = MultiHeadAttention(
            num_heads=self.n_heads,
            n_embed=self.n_embed,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
        )
        ff = FeedForward(
            n_embed=self.n_embed,
            dropout_rate=self.dropout_rate,
            weight_norm=self.weight_norm,
        )

        # Apply self-attention and residual connection followed by layer normalization
        norm = nn.LayerNorm()
        x = x + sa(norm(x), training=training)

        # Apply feedforward network and residual connection followed by layer normalization
        norm = nn.LayerNorm()
        x = x + ff(norm(x), training=training)

        return x

class ActorChunkTransformer(nn.Module):
    """Transformer backbone for the autoregressive actor."""

    n_embed: int
    n_heads: int
    n_layer: int
    dropout_rate: float
    weight_norm: bool

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        """
        Args:
            obs: [batch, obs_dim] - observation
            training: whether in training mode

        Returns:
            [batch, n_embed] - embeddings for obs token
        """
        obs = _flatten_dict(obs)
        # Observation embedding
        if self.weight_norm:
            obs_embed = nn.WeightNorm(
                nn.Dense(self.n_embed, use_bias=False, name="ObsEmbedding")
            )(obs)
        else:
            obs_embed = nn.Dense(self.n_embed, use_bias=False, name="ObsEmbedding")(obs)
        obs_embed = jnp.expand_dims(obs_embed, 1)  # [batch, 1, n_embed]

        # Transformer blocks
        for _ in range(self.n_layer):
            x = Block(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                weight_norm=self.weight_norm,
            )(obs_embed, training=training)

        # Layer norm
        norm = nn.LayerNorm()
        x = norm(x) # [batch, 1, n_embed]

        # Return all tokens (caller extracts what they need)
        return x.squeeze(1)  # [batch, n_embed]