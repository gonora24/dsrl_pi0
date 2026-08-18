"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.critic_updater import update_critic
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.actor_transformer import AutoregressiveActorTransformer
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import CriticGPTEnsemble, StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter', 'aug_next', 'num_cameras', 'chunk_reward', 'marginalize_logprobs', 'use_actor_diff', 'use_actor_diff_mean', 'freeze_residual_steps'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int,
    chunk_reward: bool, marginalize_logprobs: bool, use_actor_diff: bool,
    use_actor_diff_mean: bool,
    freeze_residual_steps: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
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
    
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(
        key, actor, critic, target_critic, temp, batch, discount,
        critic_reduction=critic_reduction, chunk_reward=chunk_reward, marginalize_logprobs=marginalize_logprobs, use_actor_diff=use_actor_diff, use_actor_diff_mean=use_actor_diff_mean)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, critic_reduction=critic_reduction, marginalize_logprobs=marginalize_logprobs, use_actor_diff=use_actor_diff, use_actor_diff_mean=use_actor_diff_mean, freeze_residual_steps=freeze_residual_steps)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class PixelSACLearner(Agent):

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
                 chunk_reward: bool = False,
                 use_chunky_actor_critic: bool = False,
                 pi0_action_horizon: int = 50,
                 critic_hidden_dims: Sequence[int] = (128, 128, 128),
                 dsrl_action_dim: int = 32,
                 use_transformer_critic: bool = False,
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
                 residual_bound: float = 1.0,
                 clip_actor_grad_norm: float = 0.0,
                 clip_critic_grad_norm: float = 0.0,
                 marginalize_logprobs: bool = False,
                 use_actor_diff: bool = False,
                 use_actor_diff_mean: bool = False,
                 freeze_residual_steps: int = 0,
                 num_noise_vectors: int = 1,
                 noise_repeats_per_vector: int = 1,
                 interpolate_noise_vectors: bool = False,
                 only_predict_dims_until: int = -1,
                 use_frozen_baseline_residual: bool = False,
                 residual_n_vectors: int = 1,
                 residual_hidden_dims: Sequence[int] = (),
                 use_residual_mlp: bool = False,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras
        self.use_chunky_actor_critic = use_chunky_actor_critic
        self.use_transformer_actor = use_transformer_actor
        self.dsrl_action_dim = dsrl_action_dim
        self.pi0_action_horizon = pi0_action_horizon
        self.num_noise_vectors = num_noise_vectors
        self.only_predict_dims_until = only_predict_dims_until
        self.noise_repeats_per_vector = noise_repeats_per_vector
        self.interpolate_noise_vectors = interpolate_noise_vectors
        self.use_frozen_baseline_residual = use_frozen_baseline_residual
        self.residual_n_vectors = residual_n_vectors
        self.residual_hidden_dims = residual_hidden_dims
        self.use_residual_mlp = use_residual_mlp
        
        if use_frozen_baseline_residual:
            # Frozen-baseline residual mode: frozen MLP predicts 1 x 32-d vector;
            # residual MLP adds corrections for residual_n_vectors extra copies.
            self.action_horizon = 1 + residual_n_vectors
            self.action_chunk_shape = (1 + residual_n_vectors, dsrl_action_dim)
            self.action_dim = dsrl_action_dim * (1 + residual_n_vectors)
            self.noise_repeats_per_vector = noise_repeats_per_vector
            _critic_is_chunky = False
            print(f'Frozen-baseline residual mode', flush=True)
        elif num_noise_vectors > 1 and not only_predict_dims_until > 0:
            # Multi-vector mode: actor predicts N independent 32-d noise vectors;
            # each is tiled noise_repeats_per_vector times at rollout before VLA inference.
            self.action_horizon = num_noise_vectors
            self.action_chunk_shape = (num_noise_vectors, dsrl_action_dim)
            self.action_dim = dsrl_action_dim * num_noise_vectors
            self.noise_repeats_per_vector = noise_repeats_per_vector
            _critic_is_chunky = True
            print(f'Multi-vector mode', flush=True)
        elif use_chunky_actor_critic and only_predict_dims_until == -1:
            # Normal chunky mode
            self.action_horizon = pi0_action_horizon
            self.action_chunk_shape = (pi0_action_horizon, dsrl_action_dim)
            self.action_dim = dsrl_action_dim * pi0_action_horizon
            self.noise_repeats_per_vector = 1
            _critic_is_chunky = True
            print(f'Normal chunky mode', flush=True)
        elif use_chunky_actor_critic and only_predict_dims_until > 0 and num_noise_vectors > 1:
            # Multi-vector mode with only predicting the first N dimensions
            self.action_horizon = num_noise_vectors
            self.action_chunk_shape = (num_noise_vectors, only_predict_dims_until)
            self.action_dim = only_predict_dims_until * num_noise_vectors
            self.noise_repeats_per_vector = noise_repeats_per_vector
            _critic_is_chunky = True
            print(f'Multi-vector mode with only predicting the first {only_predict_dims_until} dimensions', flush=True)
            print(f'action_chunk_shape: {self.action_chunk_shape}', flush=True)
            print(f'action_dim: {self.action_dim}', flush=True)
            print(f'action_horizon: {self.action_horizon}', flush=True)
            print(f'noise_repeats_per_vector: {self.noise_repeats_per_vector}', flush=True)
            print(f'_critic_is_chunky: {_critic_is_chunky}', flush=True)
        elif use_chunky_actor_critic and only_predict_dims_until > 0:
            # Chunky mode with only predicting the first N dimensions
            self.action_horizon = pi0_action_horizon
            self.action_chunk_shape = (pi0_action_horizon, only_predict_dims_until)
            self.action_dim = only_predict_dims_until * pi0_action_horizon
            self.noise_repeats_per_vector = 1
            _critic_is_chunky = True
            print(f'Chunky mode with only predicting the first {only_predict_dims_until} dimensions', flush=True)
            print(f'action_chunk_shape: {self.action_chunk_shape}', flush=True)
            print(f'action_dim: {self.action_dim}', flush=True)
            print(f'action_horizon: {self.action_horizon}', flush=True)
            print(f'noise_repeats_per_vector: {self.noise_repeats_per_vector}', flush=True)
            print(f'_critic_is_chunky: {_critic_is_chunky}', flush=True)
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
            self.noise_repeats_per_vector = 1
            _critic_is_chunky = False
            print(f'Default mode', flush=True)

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.chunk_reward = chunk_reward
        self.marginalize_logprobs = marginalize_logprobs
        self.use_actor_diff = use_actor_diff
        self.use_actor_diff_mean = use_actor_diff_mean
        self.freeze_residual_steps = freeze_residual_steps
        self.interpolate_noise_vectors = interpolate_noise_vectors
        self.only_predict_dims_until = only_predict_dims_until
        self.use_frozen_baseline_residual = use_frozen_baseline_residual
        self.residual_n_vectors = residual_n_vectors
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        if use_transformer_actor:
            assert use_chunky_actor_critic, \
                "use_transformer_actor requires use_chunky_actor_critic=True"
            state_dim = int(np.prod(observations['state'].shape[1:]))
            policy_def = AutoregressiveActorTransformer(
                state_dim=state_dim,
                image_dim=latent_dim,
                action_dim=dsrl_action_dim,
                chunk_size=self.action_horizon,
                d_model=actor_transformer_d_model,
                n_layers=actor_transformer_n_layers,
                n_heads=actor_transformer_n_heads,
                dropout=actor_transformer_dropout,
                log_std_min=-20,
                log_std_max=2,
                low=-action_magnitude,
                high=action_magnitude,
                use_actor_diff_mean=use_actor_diff_mean,
                residual_bound=residual_bound,
            )

        else:
            policy_def = LearnedStdTanhNormalPolicy(
                hidden_dims,
                self.action_dim,
                dropout_rate=dropout_rate,
                low=-action_magnitude,
                high=action_magnitude,
                action_horizon=self.action_horizon,
                dsrl_action_dim=self.dsrl_action_dim,
                use_transformer=use_chunk_actor_transformer,
                actor_transformer_n_heads=actor_transformer_n_heads,
                actor_transformer_n_layers=actor_transformer_n_layers,
                actor_transformer_weight_norm=transformer_weight_norm,
                marginalize_logprobs=marginalize_logprobs,
                use_frozen_baseline_residual=use_frozen_baseline_residual,
                residual_n_vectors=residual_n_vectors,
                residual_hidden_dims=tuple(residual_hidden_dims) if residual_hidden_dims else tuple(hidden_dims),
                use_residual_mlp=use_residual_mlp,
                only_predict_dims_until=only_predict_dims_until,
            )

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor_tx = optax.adam(learning_rate=actor_lr)
        if clip_actor_grad_norm > 0:
            # AR actor backprops through 50 transformer steps; clip to avoid NaNs.
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_actor_grad_norm),
                actor_tx,
            )
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=actor_tx,
                                  batch_stats=actor_batch_stats)

        if use_transformer_critic:
            # use_transformer_critic requires use_chunky_actor_critic=True so that
            # actions arrive as [B, T, action_dim] rather than flattened.
            # assert use_chunky_actor_critic, \
            #     "use_transformer_critic requires use_chunky_actor_critic=True"
            state_dim = int(np.prod(observations['state'].shape[1:]))
            critic_net = CriticGPTEnsemble(
                state_dim=state_dim,
                image_dim=latent_dim,
                action_horizon=self.action_horizon,
                n_embd=transformer_n_embd,
                n_head=transformer_n_head,
                n_layer=transformer_n_layer,
                dropout=dropout_rate or 0.0,
                weight_norm=transformer_weight_norm,
                use_bias=transformer_use_bias,
                num_qs=num_qs,
            )
        else:
            critic_net = StateActionEnsemble(
                critic_hidden_dims,
                num_qs=num_qs,
                use_chunky_actor_critic=_critic_is_chunky,
            )
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_net,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        print(critic_def)
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        if clip_critic_grad_norm > 0:
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_critic_grad_norm),
                optax.adam(learning_rate=critic_lr),
            )
        else:
            critic_tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=critic_tx,
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)
        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)


        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(f'use_chunky_actor_critic: {self.use_chunky_actor_critic}')
        print(f'action_chunk_shape: {self.action_chunk_shape}')
        print(self.critic_reduction)

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
            'residual_bound': residual_bound,
            'use_chunk_actor_transformer': use_chunk_actor_transformer,
            'marginalize_logprobs': marginalize_logprobs,
            'use_actor_diff': use_actor_diff,
            'num_qs': num_qs,
            'critic_hidden_dims': list(critic_hidden_dims),
            'use_transformer_critic': use_transformer_critic,
            'transformer_n_embd': transformer_n_embd,
            'transformer_n_head': transformer_n_head,
            'transformer_n_layer': transformer_n_layer,
            'transformer_use_bias': transformer_use_bias,
            'transformer_weight_norm': transformer_weight_norm,
            'use_actor_diff_mean': use_actor_diff_mean,
        }
        

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params, self._temp, batch, self.discount, self.tau, self.target_entropy, 
            self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras, self.chunk_reward, self.marginalize_logprobs, self.use_actor_diff, self.use_actor_diff_mean, self.freeze_residual_steps
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
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

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
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
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)

    def warm_start_from_baseline(self, baseline_ckpt_path: str, n_vectors: int,
                                   warm_start_critic: bool = False):
        """Copy a 32-d baseline actor into this N×32 multi-vector actor.

        All parameter leaves with matching shapes are copied directly.
        The two output-head Dense layers (means, log_stds) are tiled N times
        along their output dimension so the policy starts as N identical copies
        of the baseline, equivalent to baseline behaviour before training diverges them.

        When warm_start_critic=True (frozen-residual mode), the critic and
        target critic are also copied directly from the baseline checkpoint.
        The critic architecture is identical (32-dim action input) so no
        tiling or reshaping is needed.
        """
        from flax.traverse_util import flatten_dict, unflatten_dict

        assert pathlib.Path(baseline_ckpt_path).exists(), \
            f"Baseline checkpoint {baseline_ckpt_path} does not exist."

        baseline_raw = checkpoints.restore_checkpoint(baseline_ckpt_path, target=None)
        flat_baseline = flatten_dict(baseline_raw['actor']['params'])
        flat_mv = flatten_dict(self._actor.params)

        new_flat = {}
        tiled_keys = []
        for key, mv_val in flat_mv.items():
            if key not in flat_baseline:
                new_flat[key] = mv_val
                continue
            src = jnp.array(flat_baseline[key])
            if src.shape == mv_val.shape:
                new_flat[key] = src
            elif src.ndim == 2:
                new_flat[key] = jnp.tile(src, (1, n_vectors))
                tiled_keys.append(('.'.join(key), src.shape, new_flat[key].shape))
            elif src.ndim == 1:
                new_flat[key] = jnp.tile(src, n_vectors)
                tiled_keys.append(('.'.join(key), src.shape, new_flat[key].shape))
            else:
                new_flat[key] = mv_val

        new_params = unflatten_dict(new_flat)
        self._actor = self._actor.replace(params=new_params)
        print(f'warm-started multi-vector actor (N={n_vectors}) from {baseline_ckpt_path}')
        for name, old_shape, new_shape in tiled_keys:
            print(f'  tiled: {name}  {old_shape} -> {new_shape}')

        if warm_start_critic:
            self._critic = self._critic.replace(
                params=jax.tree_util.tree_map(jnp.array, baseline_raw['critic']['params']))
            self._target_critic_params = jax.tree_util.tree_map(
                jnp.array, baseline_raw['target_critic_params'])
            print(f'warm-started critic from {baseline_ckpt_path}')

    @classmethod
    def restore_from_checkpoint_dir(cls, ckpt_dir: str, seed: int = 0, extra_args: Dict[str, Any] = {}) -> 'PixelSACLearner':
        """Reconstruct a PixelSACLearner from a checkpoint directory.

        Reads the companion ``checkpoint{step}_config.json`` written by
        ``save_checkpoint`` to recover the exact architecture, then restores
        the weights via ``restore_checkpoint``.

        Args:
            ckpt_dir : path to the checkpoint subdirectory,
                       e.g. ``.../run_name/checkpoint941``
            seed     : RNG seed for the dummy initialisation (weights are
                       overwritten by the restore, so value does not matter)

        Returns:
            Fully restored ``PixelSACLearner`` instance.
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
            residual_bound=cfg.get('residual_bound', 1.0),
            use_chunk_actor_transformer=cfg['use_chunk_actor_transformer'],
            marginalize_logprobs=cfg['marginalize_logprobs'],
            use_actor_diff=cfg['use_actor_diff'],
            use_actor_diff_mean=cfg['use_actor_diff_mean'],
            num_qs=cfg['num_qs'],
            critic_hidden_dims=tuple(cfg['critic_hidden_dims']),
            use_transformer_critic=cfg['use_transformer_critic'],
            transformer_n_embd=cfg['transformer_n_embd'],
            transformer_n_head=cfg['transformer_n_head'],
            transformer_n_layer=cfg['transformer_n_layer'],
            transformer_use_bias=cfg['transformer_use_bias'],
            transformer_weight_norm=cfg['transformer_weight_norm'],
            **extra_args
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