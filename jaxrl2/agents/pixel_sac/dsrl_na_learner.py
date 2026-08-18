from hashlib import new
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from openpi.policies.policy import Policy

from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.agents.agent import Agent
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints, train_state
from typing import Any
import copy
import functools
import types

from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.critic_updater import (
    _infer_pi0_actions,
    update_critic,
    update_na_critic,
    update_noise_critic,
)
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble, CriticGPTEnsemble
from jaxrl2.types import Params, PRNGKey

class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(
    jax.jit,
    static_argnames=('critic_reduction', 'marginalize_logprobs', 'use_actor_diff', 'frozen', 'only_predict_dims_until', 'backup_entropy'),
)
def _update_jit(
    actor_key: PRNGKey, actor: TrainState, noise_critic: TrainState,
    na_critic: TrainState, target_na_critic_params: Params, temp: TrainState,
    batch_distill: DatasetDict, batch_train: DatasetDict,
    pi0_next_actions: jnp.ndarray, next_log_probs: jnp.ndarray,
    noise_actions: jnp.ndarray, pi0_diffused_actions: jnp.ndarray,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, marginalize_logprobs: bool, use_actor_diff: bool,
    frozen: bool, only_predict_dims_until: int,
    backup_entropy: bool,
) -> Tuple[TrainState, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    target_na_critic = na_critic.replace(params=target_na_critic_params)
    new_na_critic, na_critic_info = update_na_critic(
        na_critic, target_na_critic, temp, batch_train, pi0_next_actions,
        next_log_probs, discount, backup_entropy=backup_entropy, critic_reduction=critic_reduction,
    )
    new_target_na_critic_params = soft_target_update(new_na_critic.params, target_na_critic_params, tau)
    if not frozen:
        new_actor, actor_info = update_actor(
            actor_key, actor, noise_critic, temp, batch_train,
            critic_reduction=critic_reduction,
            marginalize_logprobs=marginalize_logprobs,
            use_actor_diff=use_actor_diff,
        )
        new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
        new_noise_critic, noise_critic_info = update_noise_critic(
            new_na_critic, noise_critic, batch_distill, noise_actions,
            pi0_diffused_actions, only_predict_dims_until=only_predict_dims_until,
        )
    else:
        new_actor = actor
        new_noise_critic = noise_critic
        new_temp = temp
        noise_critic_info = {}
        actor_info = {}
        alpha_info = {}
    return new_actor, new_noise_critic, new_na_critic, new_target_na_critic_params, new_temp, {
        **noise_critic_info,
        **na_critic_info,
        **actor_info,
        **alpha_info
    }

def _count_params(params) -> int:
    """Return the total number of scalar parameters in a Flax param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def prepare_batch(batch: DatasetDict, color_jitter: bool, aug_next: bool, num_cameras: int, rng: PRNGKey) -> DatasetDict:
    aug_pixels = batch['observations']['pixels']
    if aug_next:
        aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
    
    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    return batch


def _sample_target_noise_and_log_probs(
        actor: TrainState, observations: DatasetDict, key: PRNGKey,
        marginalize_logprobs: bool, use_actor_diff: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample the detached noise used for the Pi0 target action and entropy term."""
    dist, means, log_stds = actor.apply_fn({'params': actor.params}, observations)
    if marginalize_logprobs:
        return dist.compute_marginalized_logprobs(means, log_stds, key=key)
    if use_actor_diff:
        return dist.sample_and_log_prob_diff(seed=key)
    return dist.sample_and_log_prob(seed=key)

class DSRLNALearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 use_chunky_actor_critic: bool = False,
                 pi0_action_horizon: int = 50,
                 critic_hidden_dims: Sequence[int] = (128, 128, 128),
                 dsrl_action_dim: int = 32,
                 transformer_n_embd: int = 256,
                 transformer_n_head: int = 4,
                 transformer_n_layer: int = 4,
                 transformer_use_bias: bool = False,
                 transformer_weight_norm: bool = False,
                 use_transformer_actor: bool = False,
                 use_chunk_actor_transformer: bool = False,
                 actor_transformer_d_model: int = 256,
                 actor_transformer_n_layers: int = 3,
                 actor_transformer_n_heads: int = 4,
                 actor_transformer_dropout: float = 0.1,
                 clip_actor_grad_norm: float = 0.0,
                 clip_critic_grad_norm: float = 0.0,
                 marginalize_logprobs: bool = False,
                 use_actor_diff: bool = False,
                 agent_dp: Policy = None,
                 pi0_microbatch_size: int = 0,
                 only_predict_dims_until: int = 0,
                 freeze_latent_models: int = 0,
                 backup_entropy: bool = False,  
                 use_mlp_action_space_critic: bool = False,
    ):
        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras
        self.use_chunky_actor_critic = use_chunky_actor_critic
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.marginalize_logprobs = marginalize_logprobs
        self.use_actor_diff = use_actor_diff
        self.agent_dp = agent_dp
        self.dsrl_action_dim = dsrl_action_dim
        self.pi0_microbatch_size = pi0_microbatch_size
        self._logged_update_boundaries = False
        self._step = 0
        self.only_predict_dims_until = only_predict_dims_until
        self.backup_entropy = backup_entropy
        self.use_mlp_action_space_critic = use_mlp_action_space_critic
        if use_chunky_actor_critic and only_predict_dims_until == -1:
            # Normal chunky mode
            self.action_horizon = pi0_action_horizon
            self.action_chunk_shape = (pi0_action_horizon, dsrl_action_dim)
            self.action_dim = dsrl_action_dim * pi0_action_horizon
            self.noise_repeats_per_vector = 1
            _critic_is_chunky = True
        elif only_predict_dims_until > 0:
            # Repeat mode with only predicting the first N dimensions
            self.action_horizon = 1
            self.action_chunk_shape = (1, only_predict_dims_until)
            self.action_dim = only_predict_dims_until
            self.noise_repeats_per_vector = 1
            _critic_is_chunky = False
            print(f'Repeat mode with only predicting the first {only_predict_dims_until} dimensions', flush=True)
        else:
            self.action_horizon = 1
            self.action_chunk_shape = (1, dsrl_action_dim)
            self.action_dim = dsrl_action_dim
        self.freeze_latent_models = freeze_latent_models
        rng = jax.random.PRNGKey(seed)
        rng, noise_actor_key, noise_critic_key, critic_key, temp_key = jax.random.split(rng, 5)

        encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        policy_def = LearnedStdTanhNormalPolicy(
            hidden_dims,
            self.action_dim,
            dropout_rate=dropout_rate,
            low=-action_magnitude,
            high=action_magnitude,
            action_horizon=self.action_horizon,
            dsrl_action_dim=dsrl_action_dim,
            use_transformer=use_chunk_actor_transformer,
            actor_transformer_n_heads=actor_transformer_n_heads,
            actor_transformer_n_layers=actor_transformer_n_layers,
            actor_transformer_weight_norm=transformer_weight_norm,
            marginalize_logprobs=marginalize_logprobs,
        )

        noise_actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(noise_actor_def)
        noise_actor_def_init = noise_actor_def.init(noise_actor_key, observations)
        noise_actor_params = noise_actor_def_init['params']
        noise_actor_batch_stats = noise_actor_def_init['batch_stats'] if 'batch_stats' in noise_actor_def_init else None

        noise_actor_tx = optax.adam(learning_rate=actor_lr)
        if clip_actor_grad_norm > 0:
            # AR actor backprops through 50 transformer steps; clip to avoid NaNs.
            noise_actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_actor_grad_norm),
                noise_actor_tx,
            )
        noise_actor = TrainState.create(apply_fn=noise_actor_def.apply,
                                  params=noise_actor_params,
                                  tx=noise_actor_tx,
                                  batch_stats=noise_actor_batch_stats)
        if use_mlp_action_space_critic:
            noise_critic_net = StateActionEnsemble(
                hidden_dims,
                num_qs=num_qs,
                use_chunky_actor_critic=_critic_is_chunky,
            )
        else:
            noise_critic_net = StateActionEnsemble(
                    critic_hidden_dims,
                    num_qs=num_qs,
                    use_chunky_actor_critic=_critic_is_chunky,
                )
        noise_critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=noise_critic_net,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        print(noise_critic_def)
        batch_size = actions.shape[0]
        if use_chunky_actor_critic:
            noise_actions = jnp.zeros(
                (batch_size, *self.action_chunk_shape), dtype=actions.dtype)
        else:
            noise_actions = jnp.zeros(
                (batch_size, dsrl_action_dim), dtype=actions.dtype)
        noise_critic_def_init = noise_critic_def.init(noise_critic_key, observations, noise_actions)
        self._noise_critic_init_params = noise_critic_def_init['params']

        noise_critic_params = noise_critic_def_init['params']
        noise_critic_batch_stats = noise_critic_def_init['batch_stats'] if 'batch_stats' in noise_critic_def_init else None
        if clip_critic_grad_norm > 0:
            noise_critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_critic_grad_norm),
                optax.adam(learning_rate=critic_lr),
            )
        else:
            noise_critic_tx = optax.adam(learning_rate=critic_lr)
        noise_critic = TrainState.create(apply_fn=noise_critic_def.apply,
                                   params=noise_critic_params,
                                   tx=noise_critic_tx,
                                   batch_stats=noise_critic_batch_stats
                                   )

        state_dim = int(np.prod(observations['state'].shape[1:]))
        if use_mlp_action_space_critic:
            na_critic_net = StateActionEnsemble(
                critic_hidden_dims,
                num_qs=num_qs,
                use_chunky_actor_critic=_critic_is_chunky,
            )
        else:
            na_critic_net = CriticGPTEnsemble(
                    state_dim=state_dim,
                    image_dim=latent_dim,
                    action_horizon=pi0_action_horizon,
                    n_embd=transformer_n_embd,
                    n_head=transformer_n_head,
                    n_layer=transformer_n_layer,
                    dropout=dropout_rate or 0.0,
                    weight_norm=transformer_weight_norm,
                    use_bias=transformer_use_bias,
                    num_qs=num_qs,
                )
        na_critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=na_critic_net,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        print(na_critic_def)
        na_critic_def_init = na_critic_def.init(critic_key, observations, actions)
        self._na_critic_init_params = na_critic_def_init['params']

        na_critic_params = na_critic_def_init['params']
        na_critic_batch_stats = na_critic_def_init['batch_stats'] if 'batch_stats' in na_critic_def_init else None
        if clip_critic_grad_norm > 0:
            na_critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_critic_grad_norm),
                optax.adam(learning_rate=critic_lr),
            )
        else:
            na_critic_tx = optax.adam(learning_rate=critic_lr)
        na_critic = TrainState.create(apply_fn=na_critic_def.apply,
                                   params=na_critic_params,
                                   tx=na_critic_tx,
                                   batch_stats=na_critic_batch_stats
                                   )

        target_na_critic_params = copy.deepcopy(na_critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)


        self._rng = rng
        self._actor = noise_actor
        self._noise_critic = noise_critic
        self._na_critic = na_critic
        self._target_na_critic_params = target_na_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(f'use_chunky_actor_critic: {self.use_chunky_actor_critic}')
        print(f'noise_action_chunk_shape: {self.action_chunk_shape}')
        print(self.critic_reduction)
        print(f'[params] noise_actor:   {_count_params(noise_actor_params):>12,}')
        print(f'[params] noise_critic:  {_count_params(noise_critic_params):>12,}')
        print(f'[params] na_critic:     {_count_params(na_critic_params):>12,}')
        print(f'[params] temperature:   {_count_params(temp_params):>12,}')
        if freeze_latent_models > 0:
            print(f'[freeze] noise_actor + noise_critic + temp frozen for first {freeze_latent_models} steps; only na_critic will be optimized.')

        # Config saved alongside checkpoints so they can be restored without
        # re-specifying hyperparameters (see restore_from_checkpoint_dir).
        self._ckpt_config = {
            'obs_shapes': {k: list(v.shape) for k, v in observations.items()},
            'obs_dtypes': {k: str(v.dtype) for k, v in observations.items()},
            'action_shape': list(actions.shape),
            'action_dtype': str(actions.dtype),
            'hidden_dims': list(hidden_dims),
            'latent_dim': latent_dim,
            'encoder_type': encoder_type,
            'encoder_norm': encoder_norm,
            'use_spatial_softmax': use_spatial_softmax,
            'softmax_temperature': softmax_temperature,
            'use_bottleneck': use_bottleneck,
            'dropout_rate': dropout_rate,
            'action_magnitude': action_magnitude,
            'num_cameras': num_cameras,
            'use_chunky_actor_critic': use_chunky_actor_critic,
            'pi0_action_horizon': pi0_action_horizon,
            'dsrl_action_dim': dsrl_action_dim,
            'use_transformer_actor': use_transformer_actor,
            'actor_transformer_d_model': actor_transformer_d_model,
            'actor_transformer_n_layers': actor_transformer_n_layers,
            'actor_transformer_n_heads': actor_transformer_n_heads,
            'actor_transformer_dropout': actor_transformer_dropout,
            'use_chunk_actor_transformer': use_chunk_actor_transformer,
            'marginalize_logprobs': marginalize_logprobs,
            'use_actor_diff': use_actor_diff,
            'num_q_heads_noise': num_qs,
            'critic_hidden_dims': list(critic_hidden_dims),
            'transformer_n_embd': transformer_n_embd,
            'transformer_n_head': transformer_n_head,
            'transformer_n_layer': transformer_n_layer,
            'transformer_use_bias': transformer_use_bias,
            'transformer_weight_norm': transformer_weight_norm,
            'freeze_latent_models': freeze_latent_models,
            'only_predict_dims_until': only_predict_dims_until,
            'action_dim': self.action_dim,
            'action_chunk_shape': self.action_chunk_shape,
            'step': self._step,
            'backup_entropy': self.backup_entropy,  
            'use_mlp_action_space_critic': self.use_mlp_action_space_critic,
        }

    def _infer_pi0_actions(
            self, observations: DatasetDict, noise: jnp.ndarray, env: str,
            task_description: str, expected_action_dim: int,
    ) -> jnp.ndarray:
        """Run detached Pi0 inference outside the compiled actor/critic update."""
        pi0_noise = np.asarray(noise)
        if not self.use_chunky_actor_critic:
            if pi0_noise.ndim == 2:
                pi0_noise = pi0_noise[:, None, :]
            pi0_noise = np.concatenate(
                [
                    pi0_noise,
                    np.repeat(
                        pi0_noise[:, -1:, :],
                        self.agent_dp.action_horizon - pi0_noise.shape[1],
                        axis=1,
                    ),
                ],
                axis=1,
            )
        pi0_actions = _infer_pi0_actions(
            self.agent_dp,
            observations,
            pi0_noise,
            types.SimpleNamespace(env=env, task_description=task_description),
            microbatch_size=self.pi0_microbatch_size,
        )
        expected_shape = (
            pi0_noise.shape[0], self.agent_dp.action_horizon, expected_action_dim
        )
        if pi0_actions.shape != expected_shape:
            raise ValueError(
                f"Pi0 action shape mismatch: expected {expected_shape}, "
                f"got {pi0_actions.shape}."
            )
        if not np.isfinite(pi0_actions).all():
            raise ValueError("Pi0 inference returned non-finite actions.")
        return jax.device_put(jnp.asarray(pi0_actions, dtype=jnp.float32))

    def update(self, batch_distill: DatasetDict, batch_train: DatasetDict, env: str, task_description: str, use_noise_mapping_distill: bool) -> Dict[str, float]:
        """Perform one DSRL-NA update without nested JAX execution callbacks."""
        log_boundaries = not self._logged_update_boundaries
        if log_boundaries:
            print("DSRL-NA update: preparing augmented batches", flush=True)
        if use_noise_mapping_distill:
            self.aug_next = False
        batch_distill = prepare_batch(batch_distill, self.color_jitter, self.aug_next, self.num_cameras, self._rng)
        batch_train = prepare_batch(batch_train, self.color_jitter, self.aug_next, self.num_cameras, self._rng)

        target_key, rng = jax.random.split(self._rng)
        actor_key, noise_key = jax.random.split(rng)
        frozen = self._step < self.freeze_latent_models
        if frozen:
            print("DSRL-NA update: frozen mode, using target noise", flush=True)
            self.original_action_dsrl_action_dim = 32
            _batch_size = batch_train['actions'].shape[0]
            target_noise = jax.random.normal(
                target_key,
                (_batch_size, self.agent_dp.action_horizon, self.original_action_dsrl_action_dim),
            )
            next_log_probs = jnp.zeros(target_noise.shape[0])
        elif self.only_predict_dims_until > 0:
            print("DSRL-NA update: only predict dims until", self.only_predict_dims_until, flush=True)
            self.original_action_dsrl_action_dim = 32
            _batch_size = batch_train['actions'].shape[0]
            bg_noise_key, actor_sample_key = jax.random.split(target_key)
            next_actor_noise, next_log_probs = _sample_target_noise_and_log_probs(
                self._actor, batch_train['next_observations'], actor_sample_key,
                self.marginalize_logprobs, self.use_actor_diff)
            target_noise = jax.random.normal(
                bg_noise_key,
                (_batch_size, self.agent_dp.action_horizon, self.original_action_dsrl_action_dim),
            )
            target_noise = target_noise.at[:, :, :self.only_predict_dims_until].set(
                next_actor_noise[:, None, :]
            )
        else:
            target_noise, next_log_probs = _sample_target_noise_and_log_probs(self._actor, batch_train['next_observations'], target_key, self.marginalize_logprobs, self.use_actor_diff)
        if log_boundaries:
            print("DSRL-NA update: running target Pi0 inference", flush=True)
        pi0_next_actions = self._infer_pi0_actions(batch_train['next_observations'], target_noise, env, task_description, int(batch_train['actions'].shape[-1]))

        if use_noise_mapping_distill:
            # noise_actions stays (batch, pi0_action_horizon, 32) here; update_noise_critic
            # slices to [:, :, :only_predict_dims_until] and the (non-chunky) noise_critic's
            # _prepare_critic_actions then takes the LAST timestep as the representative
            # vector. With --repeat_noise-collected mapping data all timesteps are already
            # identical (the same vector tiled across the horizon), so "last timestep"
            # correctly recovers that single vector -- no extra reduction needed here.
            noise_actions = batch_distill['noise']
            pi0_diffused_actions = batch_distill['actions']
        else:
            noise_actions = jax.random.normal(noise_key, (batch_distill['actions'].shape[0], *self.action_chunk_shape))
            if log_boundaries:
                print("DSRL-NA update: running distillation Pi0 inference", flush=True)

            pi0_diffused_actions = self._infer_pi0_actions(
                batch_distill['observations'], noise_actions, env, task_description, int(batch_distill['actions'].shape[-1]))

        if log_boundaries:
            print("DSRL-NA update: starting compiled actor/critic update", flush=True)
        print(f'self._step: {self._step}')
        if self._step == 0:
            optimized = 'na_critic only' if frozen else 'noise_actor + noise_critic + na_critic + temp'
            print(
                f'[step {self._step}] freeze_mode={frozen} => optimizing: {optimized} '
                f'({_count_params(self._na_critic.params):,} params)',
                flush=True,
            )
        elif self._step == self.freeze_latent_models:
            print(
                f'[step {self._step}] unfreezing noise_actor ({_count_params(self._actor.params):,} params), '
                f'noise_critic ({_count_params(self._noise_critic.params):,} params), '
                f'temp ({_count_params(self._temp.params):,} params) — all models now optimized.',
                flush=True,
            )
        new_noise_actor, new_noise_critic, new_na_critic, new_target_na_critic, new_temp, info = _update_jit(
            actor_key,
            self._actor,
            self._noise_critic,
            self._na_critic,
            self._target_na_critic_params,
            self._temp,
            batch_distill,
            batch_train,
            jax.lax.stop_gradient(pi0_next_actions),
            jax.lax.stop_gradient(next_log_probs),
            jax.lax.stop_gradient(noise_actions),
            jax.lax.stop_gradient(pi0_diffused_actions),
            self.discount,
            self.tau,
            self.target_entropy,
            self.critic_reduction,
            self.marginalize_logprobs,
            self.use_actor_diff,
            frozen,
            self.only_predict_dims_until,
            self.backup_entropy if not frozen else False,
        )
        self._step += 1

        self._rng = noise_key
        self._actor = new_noise_actor
        self._noise_critic = new_noise_critic
        self._na_critic = new_na_critic
        self._target_na_critic_params = new_target_na_critic
        self._temp = new_temp
        if log_boundaries:
            print("DSRL-NA update: compiled update returned", flush=True)
            self._logged_update_boundaries = True
        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            if getattr(rewards, 'ndim', 1) > 1:
                rewards = rewards.sum(axis=-1)
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, self._na_critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'noise_critic': self._noise_critic,
            'na_critic': self._na_critic,
            'target_na_critic_params': self._target_na_critic_params,
            'actor': self._actor,
            'temp': self._temp,
            # Needed to correctly resume training: without these, a restored
            # agent restarts freeze_latent_models gating from scratch and
            # replays the exact same RNG stream a fresh run would produce.
            'step': self._step,
            'rng': self._rng,
        }
        return save_dict

    def save_checkpoint(self, dir, step, keep_every_n_steps):
        """Save Flax checkpoint and a companion JSON with hyperparameters."""
        import json
        super().save_checkpoint(dir, step, keep_every_n_steps)
        config_path = pathlib.Path(dir) / f"checkpoint{step}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self._ckpt_config, f, indent=2)
        print(f'saved config to {config_path}')

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        # Peek at the raw (unstructured) checkpoint contents first so we can
        # tell whether this is an older checkpoint saved before `step`/`rng`
        # were tracked -- flax silently keeps the template's default value
        # for any target key missing from the file, so this is purely for
        # an informative warning, not required for correctness.
        raw = checkpoints.restore_checkpoint(dir, target=None)
        has_step = isinstance(raw, dict) and 'step' in raw
        has_rng = isinstance(raw, dict) and 'rng' in raw

        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._noise_critic = output_dict['noise_critic']
        self._na_critic = output_dict['na_critic']
        self._target_na_critic_params = output_dict['target_na_critic_params']
        self._temp = output_dict['temp']
        self._step = int(output_dict['step'])
        self._rng = output_dict['rng']
        if not has_step or not has_rng:
            print(
                f'[restore_checkpoint] {dir} predates `step`/`rng` tracking; '
                f'resuming with step={self._step} and a freshly-seeded rng. '
                'freeze_latent_models gating and the RNG stream will not '
                'exactly continue the original run.',
                flush=True,
            )
        print('restored from ', dir)

    @classmethod
    def restore_from_checkpoint_dir(cls, ckpt_dir: str, seed: int = 0) -> 'DSRLNALearner':
        """Reconstruct a DSRLNALearner from a checkpoint directory.

        Reads the companion ``checkpoint{step}_config.json`` written by
        ``save_checkpoint`` to recover the exact architecture, then restores
        the weights via ``restore_checkpoint``.

        Args:
            ckpt_dir : path to the checkpoint subdirectory,
                       e.g. ``.../run_name/checkpoint941``
            seed     : RNG seed for the dummy initialisation (weights are
                       overwritten by the restore, so value does not matter)

        Returns:
            Fully restored ``DSRLNALearner`` instance.
        """
        import json
        import numpy as np

        ckpt_path = pathlib.Path(ckpt_dir)
        config_path = ckpt_path.parent / f"{ckpt_path.name}_config.json"

        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_dir}"
        assert config_path.exists(), (
            f"Config file not found: {config_path}\n"
            "Checkpoints saved before this feature was added do not have a "
            "companion config. Pass hyperparameters explicitly instead."
        )

        with open(config_path) as f:
            cfg = json.load(f)

        # Reconstruct dummy numpy arrays with the saved shapes / dtypes.
        def _make(shape, dtype_str):
            return np.zeros(shape, dtype=np.dtype(dtype_str))

        sample_obs = {k: _make(cfg['obs_shapes'][k], cfg['obs_dtypes'][k])
                      for k in cfg['obs_shapes']}
        sample_action = _make(cfg['action_shape'], cfg['action_dtype'])

        agent = cls(
            seed=seed,
            observations=sample_obs,
            actions=sample_action,
            hidden_dims=tuple(cfg['hidden_dims']),
            latent_dim=cfg['latent_dim'],
            encoder_type=cfg['encoder_type'],
            encoder_norm=cfg['encoder_norm'],
            use_spatial_softmax=cfg['use_spatial_softmax'],
            softmax_temperature=cfg['softmax_temperature'],
            use_bottleneck=cfg['use_bottleneck'],
            dropout_rate=cfg.get('dropout_rate'),
            action_magnitude=cfg['action_magnitude'],
            num_cameras=cfg['num_cameras'],
            use_chunky_actor_critic=cfg['use_chunky_actor_critic'],
            pi0_action_horizon=cfg['pi0_action_horizon'],
            dsrl_action_dim=cfg['dsrl_action_dim'],
            use_transformer_actor=cfg['use_transformer_actor'],
            actor_transformer_d_model=cfg['actor_transformer_d_model'],
            actor_transformer_n_layers=cfg['actor_transformer_n_layers'],
            actor_transformer_n_heads=cfg['actor_transformer_n_heads'],
            actor_transformer_dropout=cfg['actor_transformer_dropout'],
            use_chunk_actor_transformer=cfg['use_chunk_actor_transformer'],
            marginalize_logprobs=cfg['marginalize_logprobs'],
            use_actor_diff=cfg['use_actor_diff'],
            num_qs=cfg['num_q_heads_noise'],
            critic_hidden_dims=tuple(cfg['critic_hidden_dims']),
            transformer_n_embd=cfg['transformer_n_embd'],
            transformer_n_head=cfg['transformer_n_head'],
            transformer_n_layer=cfg['transformer_n_layer'],
            transformer_use_bias=cfg['transformer_use_bias'],
            transformer_weight_norm=cfg['transformer_weight_norm'],
            only_predict_dims_until=cfg['only_predict_dims_until'],
            freeze_latent_models=cfg['freeze_latent_models'],
            backup_entropy=cfg.get('backup_entropy', False),
        )
        agent.restore_checkpoint(ckpt_dir)
        return agent

@functools.partial(jax.jit)
def get_value(action, observation, critic):
    input_collections = {'params': critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, masks, images):

    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image