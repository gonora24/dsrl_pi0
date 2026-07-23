"""XVLA gradient sensitivity: exact Jacobians via torch.func.jacrev."""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import torch

from examples.train_utils_sim import (
    obs_to_img,
    obs_to_qpos,
    obs_to_xvla_input,
    prepare_libero_episode_for_xvla,
)
from examples.xvla_policy import XVLAPolicy
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.tests.gradient_sensitivity import (
    _prepare_pi0_noise,
    create_libero_env,
    plot_sensitivity_matrix,
)
from jaxrl2.utils.general_utils import AttrDict
from libero.libero import benchmark


def _preprocess_obs_xvla(agent_dp, obs_xvla):
    """Encode an XVLA obs dict once (VLM + proprio) outside the jacrev path."""
    return agent_dp.encode_obs(obs_xvla)


def gradient_sensitivity_matrix_xvla(agent_dp, obs_proc, noise_td):
    """Exact (T_action, T_noise) sensitivity via torch.func.jacrev (XVLA).

    Uses reverse-mode AD (``jacrev``) rather than ``jacfwd`` because
    ``scaled_dot_product_attention`` does not implement forward-mode AD.

    Differentiates model actions ``(T, 20)`` from ``generate_actions_from_enc``
    (sigmoid gripper postprocess), not the hard-thresholded Libero 7-D actions.
    """
    enc = obs_proc["enc"]
    proprio = obs_proc["proprio"]
    domain_id = obs_proc["domain_id"]
    steps = agent_dp.steps
    model = agent_dp.model

    # Copy so the tensor is writable (JAX arrays / non-writable numpy buffers).
    noise_t = torch.tensor(
        np.array(noise_td, dtype=np.float32, copy=True),
        device=agent_dp.device,
        dtype=proprio.dtype,
    )

    def fn(z):
        # z: (T, D) — add batch dim, drop it from output
        return model.generate_actions_from_enc(
            enc, domain_id, proprio, z[None], steps=steps
        )[0]

    # jacrev: reverse-mode AD (works with SDPA); same Frobenius reduction as OpenPI.
    J = torch.func.jacrev(fn)(noise_t)  # (T_action, A, T_noise, D)
    mat = torch.sqrt((J ** 2).sum(dim=(1, 3)))
    return jnp.asarray(mat.detach().cpu().numpy())


def gradient_sensitivity_matrix_over_states(agent_dp, obs_procs, noises):
    """Average sensitivity matrices over N (state, noise) pairs."""
    mats = [
        gradient_sensitivity_matrix_xvla(agent_dp, obs_proc, noise)
        for obs_proc, noise in zip(obs_procs, noises)
    ]
    return jnp.mean(jnp.stack(mats, axis=0), axis=0)


def _load_xvla_policy(checkpoint: str):
    if checkpoint == "xvla_base":
        chkpt = "2toINF/X-VLA-Pt"
    elif checkpoint == "xvla_libero":
        chkpt = "2toINF/X-VLA-Libero"
    else:
        raise ValueError(f"Invalid XVLA checkpoint: {checkpoint}")
    print(f"Loading xvla policy from {chkpt}", flush=True)
    return XVLAPolicy.from_pretrained(
        chkpt,
        device="cuda",
        domain_id=3,
        steps=10,
    )


def sensitivity_matrix_test(
    noise_actor_dir,
    libero_suite,
    task_id,
    N=100,
    checkpoint="xvla_libero",
    seed=0,
    filename="sensitivity_matrix_gradient_xvla",
):
    """Compute XVLA action sensitivity to DSRL latent noise via torch.func.jacrev."""

    print("Loading policy...", flush=True)
    agent_dp = _load_xvla_policy(checkpoint)
    action_horizon = agent_dp.action_horizon
    action_dim = agent_dp.action_dim

    if noise_actor_dir is not None:
        print("Restoring DSRL noise actor...", flush=True)
        agent = PixelSACLearner.restore_from_checkpoint_dir(noise_actor_dir)
        C, D = agent.action_chunk_shape
    else:
        C, D = (action_horizon, action_dim)
    print(f"C, D: {C, D}")
    print(f"action_horizon: {action_horizon}", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_suite]()

    if task_id is None:
        state_pool = []
        for tid in range(task_suite.n_tasks):
            suite_task = task_suite.get_task(tid)
            suite_init_states = task_suite.get_task_init_states(tid)
            for init_state in suite_init_states:
                state_pool.append((suite_task, init_state))
        if len(state_pool) == 0:
            raise ValueError(f"No init states found for suite {libero_suite}")
        print(
            f"Sampling from full suite: {task_suite.n_tasks} tasks, "
            f"{len(state_pool)} init states"
        )
        task_for_env = state_pool[0][0]
    else:
        suite_task = task_suite.get_task(task_id)
        suite_init_states = task_suite.get_task_init_states(task_id)
        if len(suite_init_states) == 0:
            raise ValueError(
                f"No init states found for suite {libero_suite}, task_id {task_id}"
            )
        state_pool = [(suite_task, init_state) for init_state in suite_init_states]
        print(f"Sampling from task {task_id}: {len(state_pool)} init states")
        task_for_env = suite_task

    variant = AttrDict({
        "env": "libero",
        "resize_image": 64,
        "task_description": task_for_env.language,
        "vla": "xvla",
    })

    env = create_libero_env(task_for_env, 256, seed)

    def _collect_obs(sample_idx):
        task_i, init_state_i = state_pool[sample_idx % len(state_pool)]
        variant.task_description = task_i.language
        env.reset()
        obs = env.set_init_state(init_state_i)
        agent_dp.reset()
        obs = prepare_libero_episode_for_xvla(env)

        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        obs_dict = {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
        obs_policy = obs_to_xvla_input(obs, variant, env=env)
        return obs_dict, obs_policy

    def _prepare_noise(noise_cd):
        return _prepare_pi0_noise(noise_cd, action_horizon, action_dim=action_dim)[0]

    def _get_noise_cd(obs_dict, i):
        if noise_actor_dir is not None:
            noise_cd = agent.sample_actions(
                obs_dict, marginalize_logprobs=False, use_actor_diff=False
            )
            if agent.action_chunk_shape[0] == 1:
                noise_repeat = np.repeat(
                    noise_cd[-1:, :], action_horizon - noise_cd.shape[0], axis=0
                )
                noise_cd = np.concatenate([noise_cd, noise_repeat], axis=0)
            else:
                noise_cd = noise_cd[0]
        else:
            noise_cd = np.asarray(
                jax.random.normal(jax.random.PRNGKey(seed + i), (C, D)),
                dtype=np.float32,
            )
        return noise_cd

    obs_dict_0, obs_policy_0 = _collect_obs(0)
    noise_cd_0 = _get_noise_cd(obs_dict_0, 0)
    print(f"noise_cd_0 shape: {noise_cd_0.shape}")

    obs_proc_0 = _preprocess_obs_xvla(agent_dp, obs_policy_0)
    noise_0 = _prepare_noise(noise_cd_0)
    print(f"prepared noise shape: {noise_0.shape}", flush=True)

    print("Computing single-state gradient sensitivity matrix...", flush=True)
    mat_0 = gradient_sensitivity_matrix_xvla(agent_dp, obs_proc_0, noise_0)
    print(f"Single-state sensitivity matrix shape: {mat_0.shape}")
    print(mat_0)

    print(f"\nComputing gradient sensitivity matrix over {N} states...", flush=True)
    obs_procs_list = []
    noises_list = []
    for i in range(N):
        obs_dict_i, obs_policy_i = _collect_obs(i)
        noise_cd_i = _get_noise_cd(obs_dict_i, i)
        obs_procs_list.append(_preprocess_obs_xvla(agent_dp, obs_policy_i))
        noises_list.append(_prepare_noise(noise_cd_i))
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{N} states", flush=True)

    mean_mat = gradient_sensitivity_matrix_over_states(
        agent_dp, obs_procs_list, noises_list
    )
    print("Mean sensitivity matrix:")
    print(mean_mat)

    fig, ax = plot_sensitivity_matrix(
        mean_mat,
        title=(
            f"Avg. gradient sensitivity for xvla {checkpoint} "
            f"({N} states, {libero_suite} all_tasks)"
            if task_id is None
            else f"Avg. gradient sensitivity for xvla {checkpoint} "
            f"({N} states, {libero_suite} task {task_id})"
        ),
        action_label=f"action timestep  i  (of {action_horizon})",
        latent_label=f"DSRL latent timestep  j  (of {C})",
    )
    out_path = f"plots/plots/sensitivity_matrices/{filename}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return mean_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_actor_dir",
        type=str,
        default=None,
        help="Path to checkpointXXX directory from PixelSACLearner",
    )
    parser.add_argument(
        "--libero_suite",
        type=str,
        default="libero_90",
        help="LIBERO benchmark suite (default: libero_90)",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Task index within the suite. If omitted, sample across the whole suite.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=100,
        help="Number of states to average over (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filename",
        type=str,
        default="sensitivity_matrix_gradient_xvla",
        help="Filename to save the sensitivity matrix (without extension)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="xvla_libero",
        help="XVLA checkpoint key: xvla_libero or xvla_base",
    )
    args = parser.parse_args()
    sensitivity_matrix_test(
        noise_actor_dir=args.noise_actor_dir,
        libero_suite=args.libero_suite,
        task_id=args.task_id,
        N=args.N,
        seed=args.seed,
        filename=args.filename,
        checkpoint=args.checkpoint,
    )
