import argparse
import datetime
import os

# Match train_sim.py: set XLA flags before JAX is imported via downstream modules.
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import h5py
import numpy as np
import tensorflow as tf

from openpi.policies.libero_policy import make_libero_example
from openpi.policies.policy_config import create_trained_policy
from examples.train_sim import CHECKPOINTS, load_pi0_checkpoint, create_exp_name, get_libero_env, openpi_config

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from libero.libero import benchmark

# Prevent TensorFlow from grabbing GPU memory alongside JAX and MuJoCo EGL.
tf.config.set_visible_devices([], "GPU")

HDF5_FILENAME = "trajectories.hdf5"


def collect_trajectories(variant):
    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ['OUTPUT_DIR'] if 'OUTPUT_DIR' in os.environ else '/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero'
    variant.outputdir = os.path.join(outputdir, expname)
    os.makedirs(variant.outputdir, exist_ok=True)
    hdf5_path = os.path.join(variant.outputdir, variant.output_file)
    print('writing to output dir ', variant.outputdir)
    print('writing trajectories to ', hdf5_path)

    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.libero_suite]()
        task_id = variant.libero_task_id
        task = task_suite.get_task(task_id)
        variant.task_description = task.language
        variant.libero_init_states = task_suite.get_task_init_states(task_id)
        libero_max_timesteps = {
            "libero_spatial": 250,
            "libero_object": 300,
            "libero_goal": 300,
            "libero_10": 550,
            "libero_90": 400,
        }
        variant.max_timesteps = libero_max_timesteps.get(variant.libero_suite, 400)

        if variant.pi0_ckpt in CHECKPOINTS:
            openpi_config_name = CHECKPOINTS[variant.pi0_ckpt]["config"]
        else:
            openpi_config_name = "pi0_libero"
    else:
        raise ValueError(f"Environment {variant.env} not supported")

    openpi_train_config = openpi_config.get_config(openpi_config_name)
    variant.pi0_action_horizon = openpi_train_config.model.action_horizon
    checkpoint_dir = load_pi0_checkpoint(variant.pi0_ckpt)
    agent_dp = create_trained_policy(openpi_train_config, checkpoint_dir)
    print(f"Loaded pi0 policy from {checkpoint_dir}", flush=True)

    # Run one JAX inference pass before MuJoCo EGL init. Mixing EGL rendering and the
    # first JAX CUDA compile in the opposite order commonly segfaults on a single GPU.
    warmup_pi0_policy(agent_dp, variant.task_description)

    if variant.env == 'libero':
        env, _ = get_libero_env(task, 256, variant.seed)

    with init_hdf5_dataset(hdf5_path, variant) as hdf5_file:
        for i in range(variant.num_trajectories):
            if i % variant.add_noise_to_actions_percentage == 0:
                print(f'Adding noise to actions for trajectory {i}')
                add_noise = True
            else:
                add_noise = False
            trajectory = collect_traj_pi0(variant, agent_dp, env, i, add_noise, variant.noise_action_freq)
            save_trajectory(trajectory, hdf5_file, i)
        hdf5_file['data'].attrs['num_demos'] = variant.num_trajectories


def init_hdf5_dataset(path, variant):
    hdf5_file = h5py.File(path, 'w')
    data_grp = hdf5_file.create_group('data')
    data_grp.attrs['date'] = datetime.datetime.now().isoformat()
    data_grp.attrs['env'] = variant.env
    data_grp.attrs['libero_suite'] = variant.libero_suite
    data_grp.attrs['libero_task_id'] = variant.libero_task_id
    data_grp.attrs['task_description'] = variant.task_description
    data_grp.attrs['pi0_ckpt'] = variant.pi0_ckpt
    data_grp.attrs['query_freq'] = variant.query_freq
    data_grp.attrs['seed'] = variant.seed
    data_grp.attrs['resize_image'] = variant.resize_image
    data_grp.attrs['add_states'] = bool(variant.add_states)
    data_grp.attrs['add_noise_to_actions_percentage'] = variant.add_noise_to_actions_percentage
    data_grp.attrs['noise_action_freq'] = variant.noise_action_freq
    return hdf5_file


def warmup_pi0_policy(agent_dp, task_description):
    obs = make_libero_example()
    obs["prompt"] = task_description
    print("Warming up pi05 inference...", flush=True)
    agent_dp.infer(obs)
    print("pi05 warmup complete", flush=True)


def _make_obs_dict(raw_obs, variant):
    obs_dict = {'pixels': obs_to_img(raw_obs, variant)}
    if variant.add_states:
        obs_dict['state'] = obs_to_qpos(raw_obs, variant)
    return obs_dict


def collect_traj_pi0(variant, agent_dp, env, traj_id, add_noise=False, noise_freq=0):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps

    obs = env.reset()
    obs_list = [_make_obs_dict(obs, variant)]
    action_list = []
    reward_list = []
    termination_list = []
    actions = None

    for t in range(max_timesteps):
        if t % query_frequency == 0:
            actions = agent_dp.infer(obs_to_pi_zero_input(obs, variant))["actions"]
            if add_noise and np.random.rand() < noise_freq:
                noise = np.random.normal(0, 1, actions.shape)
                actions = actions + noise
                actions = np.clip(actions, -1.0, 1.0)

        action_t = actions[t % query_frequency]
        action_list.append(np.asarray(action_t))
        obs, _, done, _ = env.step(action_t)

        reward_list.append(0.0 if done else -1.0)
        termination_list.append(bool(done))
        obs_list.append(_make_obs_dict(obs, variant))

        if done:
            break

    env_steps = len(action_list)
    is_success = reward_list[-1] == 0.0
    print(f'Rollout {traj_id}: success={is_success}, env_steps={env_steps}')

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': reward_list,
        'terminations': termination_list,
        'is_success': is_success,
        'env_steps': env_steps,
    }


def save_trajectory(traj, hdf5_file, demo_idx):
    traj = flatten_trajectory(traj)
    demo_grp = hdf5_file['data'].create_group(f'demo_{demo_idx}')

    obs_grp = demo_grp.create_group('obs')
    _write_dataset(obs_grp, 'pixels', traj['observations']['pixels'], compress=True)
    if 'state' in traj['observations']:
        _write_dataset(obs_grp, 'state', traj['observations']['state'])

    _write_dataset(demo_grp, 'actions', traj['actions'])
    _write_dataset(demo_grp, 'rewards', traj['rewards'])
    _write_dataset(demo_grp, 'terminations', traj['terminations'])
    _write_dataset(demo_grp, 'masks', traj['masks'])

    demo_grp.attrs['is_success'] = bool(traj['is_success'])
    demo_grp.attrs['env_steps'] = int(traj['env_steps'])


def _write_dataset(group, name, data, compress=False):
    kwargs = {}
    if compress:
        kwargs['compression'] = 'gzip'
    group.create_dataset(name, data=np.asarray(data), **kwargs)


def _stack_dict_list(dict_list):
    keys = dict_list[0].keys()
    return {
        key: np.stack([np.asarray(item[key]) for item in dict_list], axis=0)
        for key in keys
    }


def flatten_trajectory(trajectory):
    terminations = np.asarray(trajectory['terminations'], dtype=np.bool_)
    return {
        'observations': _stack_dict_list(trajectory['observations']),
        'actions': np.stack([np.asarray(action) for action in trajectory['actions']], axis=0),
        'rewards': np.asarray(trajectory['rewards'], dtype=np.float32),
        'terminations': terminations,
        'masks': np.logical_not(terminations).astype(np.float32),
        'is_success': trajectory['is_success'],
        'env_steps': trajectory['env_steps'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='libero')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--libero_suite', type=str, default='libero_90')
    parser.add_argument('--libero_task_id', type=int, default=59)
    parser.add_argument('--pi0_ckpt', type=str, default='pi05_libero')
    parser.add_argument('--num_trajectories', type=int, default=100)
    parser.add_argument('--outputdir', type=str, default='./logs')
    parser.add_argument('--output_file', type=str, default=HDF5_FILENAME)
    parser.add_argument('--query_freq', type=int, default=10)
    parser.add_argument('--add_states', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resize_image', type=int, default=64)
    parser.add_argument('--add_noise_to_actions_percentage', type=int, default=0)
    parser.add_argument('--noise_action_freq', type=float, default=0)
    args = parser.parse_args()
    collect_trajectories(args)
