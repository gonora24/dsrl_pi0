import argparse
import datetime
import os
import jax
# Match train_sim.py: set XLA flags before JAX is imported via downstream modules.
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import h5py
import numpy as np
import tensorflow as tf

from openpi.policies.libero_policy import make_libero_example
from openpi.policies.policy_config import create_trained_policy
from examples.train_sim import (
    CHECKPOINTS,
    create_exp_name,
    get_libero_env,
    load_norm_stats_for_checkpoint,
    load_pi0_checkpoint,
    openpi_config,
)

from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from libero.libero import benchmark

# Prevent TensorFlow from grabbing GPU memory alongside JAX and MuJoCo EGL.
tf.config.set_visible_devices([], "GPU")

H5_FILENAME = "noise_action_mapping.h5"


# ---------------------------------------------------------------------------
# Incremental, resumable HDF5 store.
#
# Design:
#   - Each of noise/actions/obs_pixels/obs_state is its own resizable dataset.
#   - append_trajectory() writes all of them to the SAME row range [n:n+m] in
#     one call, so sample i is guaranteed to correspond across all arrays.
#   - The 'n_saved' attribute (the only source of truth for "how much valid
#     data exists") is only updated *after* every array has been written and
#     flushed to disk. So if the process dies mid-write, you just get some
#     unused trailing rows past n_saved on the next open -- never a
#     misaligned or corrupted dataset.
#   - Because we open in append ('a') mode and resume from n_saved, restarting
#     the job after a crash / wrong time limit just continues collecting
#     instead of starting over.
# ---------------------------------------------------------------------------
def open_or_create_store(path, num_mappings, noise_shape, action_shape, pixel_shape, state_shape=None):
    f = h5py.File(path, 'a')

    if 'n_saved' not in f.attrs:
        f.attrs['n_saved'] = 0
        f.create_dataset('noise', shape=(0, *noise_shape), maxshape=(num_mappings, *noise_shape),
                          dtype=np.float32, chunks=(1, *noise_shape))
        f.create_dataset('actions', shape=(0, *action_shape), maxshape=(num_mappings, *action_shape),
                          dtype=np.float32, chunks=(1, *action_shape))
        f.create_dataset('obs_pixels', shape=(0, *pixel_shape), maxshape=(num_mappings, *pixel_shape),
                          dtype=np.uint8, chunks=(1, *pixel_shape))
        if state_shape is not None:
            f.create_dataset('obs_state', shape=(0, *state_shape), maxshape=(num_mappings, *state_shape),
                              dtype=np.float32, chunks=(1, *state_shape))
        f.flush()
    else:
        # Resuming: sanity-check the shapes match what this run expects.
        assert f['noise'].shape[1:] == noise_shape
        assert f['actions'].shape[1:] == action_shape
        assert f['obs_pixels'].shape[1:] == pixel_shape
        if state_shape is not None:
            assert f['obs_state'].shape[1:] == state_shape

    return f


def append_trajectory(f, noise, actions, obs_pixels, obs_state=None):
    n = f.attrs['n_saved']
    m = noise.shape[0]
    assert actions.shape[0] == m and obs_pixels.shape[0] == m, "trajectory arrays out of sync"
    if obs_state is not None:
        assert obs_state.shape[0] == m

    arrays = [('noise', noise), ('actions', actions), ('obs_pixels', obs_pixels)]
    if obs_state is not None:
        arrays.append(('obs_state', obs_state))

    for name, arr in arrays:
        ds = f[name]
        if ds.shape[0] < n + m:
            ds.resize(n + m, axis=0)
        ds[n:n + m] = arr

    # Update the counter LAST, only once every array above is safely written.
    f.attrs['n_saved'] = n + m
    f.flush()
    return n + m


def load_dataset(path):
    """
    Load the aligned dataset for training. noise[i], actions[i], obs_pixels[i]
    (and obs_state[i]) all correspond to the same collected sample.
    """
    with h5py.File(path, 'r') as f:
        n = f.attrs['n_saved']
        noise = f['noise'][:n]
        actions = f['actions'][:n]
        obs_pixels = f['obs_pixels'][:n]
        obs_state = f['obs_state'][:n] if 'obs_state' in f else None
    return noise, actions, obs_pixels, obs_state


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_trajectories(variant):
    variant.outputdir = os.environ.get('OUTPUT_DIR', '/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp/DSRL_pi0_Libero')
    os.makedirs(variant.outputdir, exist_ok=True)
    h5_path = os.path.join(variant.outputdir, variant.output_file)
    print('writing to output dir ', variant.outputdir)
    print('writing noise action mapping to ', h5_path)

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
    norm_stats = load_norm_stats_for_checkpoint(variant.pi0_ckpt)
    agent_dp = create_trained_policy(openpi_train_config, checkpoint_dir, norm_stats=norm_stats)
    print(f"Loaded pi0 policy from {checkpoint_dir}", flush=True)

    # Run one JAX inference pass before MuJoCo EGL init. Mixing EGL rendering and the
    # first JAX CUDA compile in the opposite order commonly segfaults on a single GPU.
    warmup_pi0_policy(agent_dp, variant.task_description)

    if variant.env == 'libero':
        env, _ = get_libero_env(task, 256, variant.seed)

    # Peek at one real observation so we know the actual pixel/state shapes
    # instead of hard-coding them (avoids the old (10, 32)-shaped obs bug).
    probe_obs = _make_obs_dict(env.reset(), variant)
    pixel_shape = probe_obs['pixels'].shape
    state_shape = probe_obs['state'].shape if variant.add_states else None

    f = open_or_create_store(
        h5_path, variant.num_mappings,
        noise_shape=(10, 32), action_shape=(10, 32),
        pixel_shape=pixel_shape, state_shape=state_shape,
    )

    i = f.attrs['n_saved']
    traj_id = 0
    print(f'Resuming at {i}/{variant.num_mappings} samples already saved', flush=True)

    try:
        while i < variant.num_mappings:
            trajectory = collect_traj_pi0(variant, agent_dp, env, traj_id)
            noise = np.stack(trajectory["noise"], axis=0)
            actions = np.stack(trajectory["actions"], axis=0)
            obs_pixels = np.stack([o['pixels'] for o in trajectory["obs"]], axis=0)
            obs_state = (np.stack([o['state'] for o in trajectory["obs"]], axis=0)
                         if variant.add_states else None)

            # Don't overshoot the target if this trajectory would push past it.
            remaining = variant.num_mappings - i
            if noise.shape[0] > remaining:
                noise = noise[:remaining]
                actions = actions[:remaining]
                obs_pixels = obs_pixels[:remaining]
                if obs_state is not None:
                    obs_state = obs_state[:remaining]

            i = append_trajectory(f, noise, actions, obs_pixels, obs_state)
            traj_id += 1
            print(f'Saved {i}/{variant.num_mappings} samples (traj {traj_id})', flush=True)
    finally:
        # Ensures the file is properly closed (and any buffered attrs flushed)
        # even if collection is interrupted (e.g. Ctrl-C, time-limit SIGTERM).
        f.close()

    return


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


def collect_traj_pi0(variant, agent_dp, env, traj_id):
    rng, rng_noise = jax.random.split(agent_dp._rng)
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps

    obs = env.reset()
    action_list = []
    noise_list = []
    obs_list = []
    actions = None

    for t in range(max_timesteps):
        if t % query_frequency == 0:
            noise = jax.random.normal(rng_noise, (1, 10, 32))
            obs_processed = obs_to_pi_zero_input(obs, variant)
            actions = agent_dp.infer(obs_processed, noise=noise)["actions"]
            actions_to_save = np.pad(actions, ((0, 0), (0, 32 - actions.shape[1])), mode='constant', constant_values=0.0)
            action_list.append(actions_to_save)
            noise_list.append(noise[0, :, :])
            obs_list.append(_make_obs_dict(obs, variant))

        action_t = actions[t % query_frequency]
        obs, _, done, _ = env.step(action_t)

        if done:
            break

    env_steps = len(action_list)
    print(f'Rollout {traj_id}: actions={env_steps}')

    return {
        'actions': action_list,
        'noise': noise_list,
        'obs': obs_list,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='libero')
    parser.add_argument('--libero_suite', type=str, default='libero_90')
    parser.add_argument('--libero_task_id', type=int, default=59)
    parser.add_argument('--pi0_ckpt', type=str, default='pi05_libero')
    parser.add_argument('--num_mappings', type=int, default=1000)
    parser.add_argument('--outputdir', type=str, default='./logs')
    parser.add_argument('--output_file', type=str, default=H5_FILENAME)
    parser.add_argument('--query_freq', type=int, default=10)
    parser.add_argument('--add_states', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resize_image', type=int, default=64)
    args = parser.parse_args()
    collect_trajectories(args)