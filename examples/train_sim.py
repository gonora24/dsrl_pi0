#! /usr/bin/env python
import os

from jaxrl2.agents.pixel_sac.dsrl_na_learner import DSRLNALearner
from jaxrl2.data.replay_buffer import H5ReplayBuffer
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box
# import metaworld

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop, offline_to_online_training_loop
from examples.xvla_policy import XVLAPolicy
from jaxrl2.utils.launch_util import get_full_config_dict, print_full_config
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.training import checkpoints as openpi_checkpoints
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

# Default OpenPI Libero asset id used inside fine-tuned checkpoints (e.g. pi05_libero).
LIBERO_NORM_STATS_ASSET_ID = "physical-intelligence/libero"

CHECKPOINTS = {
    "openpi": {
        "config": "pi0_libero",
        "source": "gs://openpi-assets/checkpoints/pi0_libero",  # switch from s3://
    },
    "pi05_libero": {
        "config": "pi05_libero",
        "source": "gs://openpi-assets/checkpoints/pi05_libero/",
    },
    "pi05_droid": {
        "config": "pi05_droid",
        "source": "gs://openpi-assets/checkpoints/pi05_droid/",
    },
    "pi05_base": {
        "config": "pi05_libero",
        "source": "gs://openpi-assets/checkpoints/pi05_base",
        # Base checkpoint has no Libero norm_stats; reuse those from the Libero fine-tune.
        "norm_stats_source": "gs://openpi-assets/checkpoints/pi05_libero",
        "norm_stats_asset_id": LIBERO_NORM_STATS_ASSET_ID,
    },
    "rlinf_hf_long": {
        "config": "pi0_libero",
        "hf_repo": "RLinf/RLinf-Pi0-LIBERO-Long-SFT",
        "pytorch": True,
    },
    "rlinf_hf_goalSpatial": {
        "config": "pi0_libero",
        "hf_repo": "RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT",
        "pytorch": True,
    },
    "rlinf_hf_pi05": {
        "config": "pi05_libero",
        "hf_repo": "RLinf/RLinf-Pi05-LIBERO-SFT",
        "pytorch": True,
    },
    "rlinf_hf_pi05_metaworld": {
        "config": "pi05_libero",
        "hf_repo": "RLinf/RLinf-Pi05-MetaWorld-SFT",
        "pytorch": True,
    },
}


def load_pi0_checkpoint(pi0_ckpt: str) -> pathlib.Path:
    if pi0_ckpt not in CHECKPOINTS:
        checkpoint_dir = pathlib.Path(pi0_ckpt).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"--pi0_checkpoint path is not a directory: {checkpoint_dir}")
        return checkpoint_dir

    spec = CHECKPOINTS[pi0_ckpt]
    if "source" in spec:
        return pathlib.Path(download.maybe_download(spec["source"]))

    from huggingface_hub import snapshot_download
    from openpi.shared import transformers_rlinf_patch

    hf_cache = pathlib.Path(download.get_cache_dir()) / "hf" / spec["hf_repo"].replace("/", "_")
    checkpoint_dir = pathlib.Path(
        snapshot_download(spec["hf_repo"], local_dir=str(hf_cache))
    )
    transformers_rlinf_patch.ensure_rlinf_transformers_patched()
    transformers_rlinf_patch.purge_transformers_imports()
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
    return checkpoint_dir


def load_norm_stats_for_checkpoint(pi0_ckpt: str):
    """Load override norm_stats when a CHECKPOINTS entry sets ``norm_stats_source``.

    Returns None so ``create_trained_policy`` falls back to the checkpoint's own assets.
    """
    if pi0_ckpt not in CHECKPOINTS:
        return None
    spec = CHECKPOINTS[pi0_ckpt]
    source = spec.get("norm_stats_source")
    if not source:
        return None
    asset_id = spec.get("norm_stats_asset_id", LIBERO_NORM_STATS_ASSET_ID)
    assets_root = pathlib.Path(download.maybe_download(source)) / "assets"
    return openpi_checkpoints.load_norm_stats(assets_root, asset_id)


# Alias used by some eval scripts.
_load_pi0_checkpoint = load_pi0_checkpoint


def get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


def _get_metaworld_env(task_name: str, seed: int):
    """Initializes and returns the MetaWorld environment, along with the task description."""
    env = gym.make("Meta-World/MT1", env_name=task_name, render_mode="rgb_array")
    task_description = ""
    return env, task_description

class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            elif variant.env == 'metaworld':
                state_dim = 4
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        noise_dim = int(getattr(variant, 'dsrl_action_dim', 32))
        num_noise_vectors = int(getattr(variant, 'num_noise_vectors', 1))
        if num_noise_vectors > 1 and variant.algorithm == 'pixel_sac':
            action_shape = (num_noise_vectors, noise_dim)
        elif variant.use_chunky_actor_critic and variant.algorithm == 'pixel_sac':
            action_shape = (variant.pi0_action_horizon, noise_dim)
        elif variant.algorithm == 'dsrl_na' and variant.env == 'libero' and not variant.chunk_reward:
            action_shape = (1, 7)
        elif variant.algorithm == 'dsrl_na' and variant.env == 'libero' and variant.chunk_reward:
            action_shape = (variant.pi0_action_horizon, 7)
        else:
            action_shape = (1, noise_dim)
        self.action_space = Box(low=-1, high=1, shape=action_shape, dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    assert not (variant.use_actor_diff_mean and variant.use_actor_diff), \
        "use_actor_diff_mean and use_actor_diff cannot be used together"
    assert not (variant.use_actor_diff_mean and variant.marginalize_logprobs), \
        "use_actor_diff_mean and marginalize_logprobs cannot be used together"

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.environ['OUTPUT_DIR'] if 'OUTPUT_DIR' in os.environ else '/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero'
    variant.outputdir = os.path.join(outputdir, expname)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir ', outputdir, flush=True)

    ## Environment
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.libero_suite]() # originally hardcoded: libero_90
        task_id = variant.libero_task_id # originally hardcoded: 57
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        if variant.chunk_reward:
            variant.env_max_reward = 0
        variant.libero_init_states = task_suite.get_task_init_states(task_id)
        # Match OpenPI libero eval horizons (examples/libero/main.py).
        if variant.vla == 'openpi':
            libero_max_timesteps = {
                "libero_spatial": 250, #220, divisible by 50
                "libero_object": 300, #280,
                "libero_goal": 300,
                "libero_10": 550, #520,
                "libero_90": 400,
            }
            variant.max_timesteps = libero_max_timesteps.get(variant.libero_suite, 400)
        elif variant.vla == 'xvla':
            variant.max_timesteps = 400
        # Absolute control is enabled after each episode's delta settle
        # (see prepare_libero_episode_for_xvla). Do NOT set use_delta=False here —
        # zero dummy settle under absolute control teleports the EE to the origin.
    elif variant.env == 'metaworld':
        variant.max_timesteps = 400
        env, task_description = _get_metaworld_env(variant.metaworld_task_name, variant.seed)
        eval_env = copy.deepcopy(env)
        variant.task_description = task_description
        variant.env_max_reward = 1
        if variant.chunk_reward:
            variant.env_max_reward = 0
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        
    ## Wandb Logger    
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)
    pi0_ckpt = getattr(variant, "pi0_checkpoint", "openpi")
    if variant.env == 'libero' or variant.env == 'metaworld':
        if pi0_ckpt in CHECKPOINTS:
            openpi_config_name = CHECKPOINTS[pi0_ckpt]["config"]
        else:
            openpi_config_name = "pi0_libero"
    else:
        openpi_config_name = "pi0_aloha_sim"
    openpi_train_config = openpi_config.get_config(openpi_config_name)
    variant.use_chunky_actor_critic = bool(getattr(variant, 'use_chunky_actor_critic', 0))
    variant.use_actor_diff = bool(getattr(variant, 'use_actor_diff', 0))
    variant.overlap_transitions = bool(getattr(variant, 'overlap_transitions', 0))
    assert not (variant.use_actor_diff and variant.marginalize_logprobs), \
        "use_actor_diff and marginalize_logprobs are mutually exclusive"
    variant.freeze_residual_steps = int(getattr(variant, 'freeze_residual_steps', 0))
    assert not (variant.freeze_residual_steps > 0 and not variant.use_actor_diff), \
        "freeze_residual_steps > 0 requires use_actor_diff=True"
    assert not (variant.overlap_transitions and not variant.chunk_reward), \
        "overlap_transitions requires chunk_reward=1"

    ## Load Policy (before DummyEnv so horizon / noise dim match the VLA)
    if variant.vla == 'openpi':
        if variant.env == 'libero' or variant.env == 'metaworld':
            checkpoint_dir = load_pi0_checkpoint(pi0_ckpt)
            norm_stats = load_norm_stats_for_checkpoint(pi0_ckpt)
        elif variant.env == 'aloha_cube':
            checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
            norm_stats = None
        else:
            raise NotImplementedError()
        agent_dp = policy_config.create_trained_policy(
            openpi_train_config, checkpoint_dir, norm_stats=norm_stats
        )
        variant.pi0_action_horizon = openpi_train_config.model.action_horizon
        variant.dsrl_action_dim = 32
        print(f"Loaded pi0 policy from {checkpoint_dir}", flush=True)
    elif variant.vla == 'xvla' and variant.env == 'libero':
        xvla_device = "cuda" if any(d.platform == "gpu" for d in devices) else "cpu"
        agent_dp = XVLAPolicy.from_pretrained(
            "2toINF/X-VLA-Libero",
            device=xvla_device,
            domain_id=3,
            steps=10,
        )
        variant.pi0_action_horizon = agent_dp.action_horizon
        variant.dsrl_action_dim = agent_dp.action_dim
        # Absolute ee6d chunks must be executed in full before replan (official client).
        if int(getattr(variant, 'query_freq', -1)) != int(agent_dp.action_horizon):
            print(
                f"Overriding query_freq {variant.query_freq} -> {agent_dp.action_horizon} "
                f"for XVLA (must match action horizon)",
                flush=True,
            )
            variant.query_freq = int(agent_dp.action_horizon)
        print(
            f"Loaded XVLA policy (horizon={agent_dp.action_horizon}, "
            f"noise_dim={agent_dp.action_dim}, query_freq={variant.query_freq}, device={xvla_device})",
            flush=True,
        )
    else:
        raise NotImplementedError()

    if variant.only_predict_dims_until > 0:
        variant.dsrl_action_dim = variant.only_predict_dims_until
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    ## Algorithm -> Model
    train_kwargs = dict(variant['train_kwargs'])
    if train_kwargs.pop('cosine_decay', False):
        train_kwargs['decay_steps'] = variant.max_steps
    # if variant.use_transformer_critic:
    #     train_kwargs['num_qs'] = 4
    if variant.use_actor_diff:
        train_kwargs['target_entropy'] = -16.0
    if variant.algorithm == 'dsrl_na':
        agent = DSRLNALearner(
            variant.seed,
            sample_obs,
            sample_action,
            agent_dp=agent_dp,
            use_chunky_actor_critic=variant.use_chunky_actor_critic,
            pi0_action_horizon=variant.pi0_action_horizon,
            pi0_microbatch_size=variant.pi0_microbatch_size,
            dsrl_action_dim=variant.dsrl_action_dim,
            critic_hidden_dims=tuple(variant.critic_hidden_dims),
            hidden_dims=tuple(variant.hidden_dims),
            num_qs=variant.num_qs,
            transformer_n_embd=variant.transformer_n_embd,
            transformer_n_head=variant.transformer_n_head,
            transformer_n_layer=variant.transformer_n_layer,
            transformer_weight_norm=variant.transformer_weight_norm,
            transformer_use_bias=variant.transformer_use_bias,
            freeze_latent_models=variant.freeze_latent_models,
            only_predict_dims_until=variant.only_predict_dims_until,
            backup_entropy=bool(variant.backup_entropy),
            use_mlp_action_space_critic=bool(variant.use_mlp_action_space_critic),
            **train_kwargs,
        )
    else:
        agent = PixelSACLearner(
            variant.seed,
            sample_obs,
            sample_action,
            chunk_reward=bool(variant.chunk_reward),
            use_chunky_actor_critic=variant.use_chunky_actor_critic,
            pi0_action_horizon=variant.pi0_action_horizon,
            dsrl_action_dim=variant.dsrl_action_dim,
            num_qs=variant.num_qs,
            hidden_dims=tuple(variant.hidden_dims),
            critic_hidden_dims=tuple(variant.critic_hidden_dims),
            use_transformer_critic=variant.use_transformer_critic,
            transformer_n_embd=variant.transformer_n_embd,
            transformer_n_head=variant.transformer_n_head,
            transformer_n_layer=variant.transformer_n_layer,
            transformer_weight_norm=variant.transformer_weight_norm,
            transformer_use_bias=variant.transformer_use_bias,
            use_transformer_actor=variant.use_transformer_actor,
            actor_transformer_d_model=variant.actor_transformer_d_model,
            actor_transformer_n_layers=variant.actor_transformer_n_layers,
            actor_transformer_n_heads=variant.actor_transformer_n_heads,
            actor_transformer_dropout=variant.actor_transformer_dropout,
            residual_bound=getattr(variant, 'residual_bound', 1.0),
            residual_mean_bound=getattr(variant, 'residual_mean_bound', 0.3),
            residual_log_std_bound=getattr(variant, 'residual_log_std_bound', 2.0),
            clip_actor_grad_norm=variant.clip_actor_grad_norm,
            clip_critic_grad_norm=variant.clip_critic_grad_norm,
            marginalize_logprobs=variant.marginalize_logprobs,
            use_chunk_actor_transformer=variant.use_chunk_actor_transformer,
            use_actor_diff=variant.use_actor_diff,
            use_actor_diff_mean=variant.use_actor_diff_mean,
            freeze_residual_steps=variant.freeze_residual_steps,
            num_noise_vectors=getattr(variant, 'num_noise_vectors', 1),
            noise_repeats_per_vector=getattr(variant, 'noise_repeats_per_vector', 1),
            interpolate_noise_vectors=bool(getattr(variant, 'interpolate_noise_vectors', 0)),
            only_predict_dims_until=variant.only_predict_dims_until,
            use_frozen_baseline_residual=bool(getattr(variant, 'use_frozen_baseline_residual', 0)),
            residual_n_vectors=getattr(variant, 'residual_n_vectors', 1),
            residual_hidden_dims=tuple(variant.residual_hidden_dims) if getattr(variant, 'residual_hidden_dims', None) else (),
            use_residual_mlp=bool(getattr(variant, 'use_residual_mlp', 0)),
        **train_kwargs,
        )

    if getattr(variant, 'restore_path', None) is not None:
        agent.restore_checkpoint(variant.restore_path)

    if getattr(variant, 'initialize_weights_from', None) is not None:
        # In frozen-residual mode the frozen head is always 32-dim (n_vectors=1);
        # in multi-vector mode pass the full N so the output heads are tiled.
        _warm_n = 1 if getattr(variant, 'use_frozen_baseline_residual', 0) else getattr(variant, 'num_noise_vectors', 1)
        _warm_critic = bool(getattr(variant, 'use_frozen_baseline_residual', 0))
        agent.warm_start_from_baseline(
            variant.initialize_weights_from,
            n_vectors=_warm_n,
            warm_start_critic=_warm_critic,
        )

    if variant.only_predict_dims_until > 0:
        variant.dsrl_action_dim = agent_dp.action_dim
    ## Replay Buffer
    # print_full_config(variant, agent=agent, extra=config_extra)
    if variant.chunk_reward and not variant.algorithm == 'dsrl_na':
        online_buffer_size = variant.online_buffer_size if variant.online_buffer_size > 0 else variant.max_steps
    else:
        online_buffer_size = variant.online_buffer_size if variant.online_buffer_size > 0 else variant.max_steps // variant.multi_grad_step
    chunk_size = variant.query_freq if variant.chunk_reward else 0
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(online_buffer_size),
        chunk_size=chunk_size,
    )
    if variant.algorithm == 'dsrl_na':
        online_replay_buffer.load_from_hdf5(variant.trajectory_hdf5_path, chunk_reward=variant.chunk_reward, query_freq=variant.query_freq, discount=variant.discount)
        print(f"Loaded {online_replay_buffer._traj_counter} trajectories and {online_replay_buffer.size} transitions from {variant.trajectory_hdf5_path}", flush=True)
        if variant.use_noise_mapping_distill:
            noise_mapping_distill_buffer = H5ReplayBuffer(variant.noise_mapping_distill_path, dummy_env.action_space)
            print(f"Loaded {noise_mapping_distill_buffer.size} mappings from {variant.noise_mapping_distill_path}", flush=True)
        else:
            noise_mapping_distill_buffer = None
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    if variant.algorithm == 'dsrl_na':
        # If we restored from a checkpoint, continue the step count (and
        # therefore checkpoint/wandb numbering and the offline/online phase
        # boundary) from where that run left off instead of restarting at 0.
        start_step = agent._step if getattr(variant, 'restore_path', None) is not None else 0
        offline_to_online_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, noise_mapping_distill_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp, start_step=start_step)
    else:
        trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 