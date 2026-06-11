#! /usr/bin/env python
import os
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

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
from jaxrl2.utils.launch_util import get_full_config_dict, print_full_config
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

CHECKPOINTS = {
    "openpi": {
        "config": "pi0_libero",
        "source": "gs://openpi-assets/checkpoints/pi0_libero",  # switch from s3://
    },
    "pi05_libero": {
        "config": "pi05_libero",
        "source": "gs://openpi-assets/checkpoints/pi05_libero/",
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
}


def _load_pi0_checkpoint(pi0_ckpt: str) -> pathlib.Path:
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


def _get_libero_env(task, resolution, seed):
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
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        if variant.use_chunky_actor_critic:
            action_shape = (variant.pi0_action_horizon, 32)
        else:
            action_shape = (1, 32)
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
   
    outputdir = os.path.abspath(os.path.join(os.environ['EXP'], expname))
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.libero_suite]() # originally hardcoded: libero_90
        task_id = variant.libero_task_id # originally hardcoded: 57
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.libero_init_states = task_suite.get_task_init_states(task_id)
        # Match OpenPI libero eval horizons (examples/libero/main.py).
        libero_max_timesteps = {
            "libero_spatial": 250, #220, divisible by 50
            "libero_object": 300, #280,
            "libero_goal": 300,
            "libero_10": 550, #520,
            "libero_90": 400,
        }
        variant.max_timesteps = libero_max_timesteps.get(variant.libero_suite, 400)
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
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)
    pi0_ckpt = getattr(variant, "pi0_checkpoint", "openpi")
    if variant.env == 'libero':
        if pi0_ckpt in CHECKPOINTS:
            openpi_config_name = CHECKPOINTS[pi0_ckpt]["config"]
        else:
            openpi_config_name = "pi0_libero"
    else:
        openpi_config_name = "pi0_aloha_sim"
    openpi_train_config = openpi_config.get_config(openpi_config_name)
    variant.pi0_action_horizon = openpi_train_config.model.action_horizon
    variant.use_chunky_actor_critic = bool(getattr(variant, 'use_chunky_actor_critic', 0))
    if variant.use_chunky_actor_critic:
        if variant.query_freq <= 0:
            raise ValueError("use_chunky_actor_critic requires --query_freq > 0")
        if variant.query_freq != variant.pi0_action_horizon:
            raise ValueError(
                "use_chunky_actor_critic requires --query_freq to match pi0 action horizon "
                f"({variant.pi0_action_horizon}), got query_freq={variant.query_freq}"
            )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    

    if variant.env == 'libero':
        checkpoint_dir = _load_pi0_checkpoint(pi0_ckpt)
    elif variant.env == 'aloha_cube':
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()

    if variant.env == 'libero':
        config = openpi_train_config
    elif variant.env == 'pi05_libero':
        config = openpi_train_config
    else:
        config = openpi_train_config

    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Loaded pi0 policy from {checkpoint_dir}", flush=True)
    train_kwargs = dict(variant['train_kwargs'])
    if train_kwargs.pop('cosine_decay', False):
        train_kwargs['decay_steps'] = variant.max_steps
    if variant.use_transformer_critic:
        train_kwargs['num_qs'] = 1
    agent = PixelSACLearner(
        variant.seed,
        sample_obs,
        sample_action,
        chunk_reward=bool(variant.chunk_reward),
        use_chunky_actor_critic=variant.use_chunky_actor_critic,
        pi0_action_horizon=variant.pi0_action_horizon,
        use_transformer_critic=variant.use_transformer_critic,
        transformer_n_embd=variant.transformer_n_embd,
        transformer_n_head=variant.transformer_n_head,
        transformer_n_layer=variant.transformer_n_layer,
        transformer_use_layer_norm=variant.transformer_use_layer_norm,
        transformer_use_bias=variant.transformer_use_bias,
        use_transformer_actor=variant.use_transformer_actor,
        actor_transformer_d_model=variant.actor_transformer_d_model,
        actor_transformer_n_layers=variant.actor_transformer_n_layers,
        actor_transformer_n_heads=variant.actor_transformer_n_heads,
        actor_transformer_dropout=variant.actor_transformer_dropout,
        **train_kwargs,
    )

    config_extra = {
        "pi0_checkpoint_dir": str(checkpoint_dir),
        "openpi_config": config.name,
        "online_buffer_size": variant.max_steps // variant.multi_grad_step,
    }
    # print_full_config(variant, agent=agent, extra=config_extra)

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    chunk_size = variant.query_freq if variant.chunk_reward else 0
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(online_buffer_size),
        chunk_size=chunk_size,
    )
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 