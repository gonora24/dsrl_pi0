#!/usr/bin/env python
"""Evaluate Pi0 / Pi0.5 on all tasks in a LIBERO suite."""

import argparse
import json
import os
import pathlib
import sys
import types
from datetime import datetime

import jax
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.training import config as openpi_config

from examples.train_sim import CHECKPOINTS, _load_pi0_checkpoint, load_norm_stats_for_checkpoint
from examples.train_utils_sim import obs_to_pi_zero_input

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

LIBERO_MAX_TIMESTEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _resolve_openpi_config(pi0_checkpoint: str) -> openpi_config.TrainConfig:
    if pi0_checkpoint in CHECKPOINTS:
        return openpi_config.get_config(CHECKPOINTS[pi0_checkpoint]["config"])
    return openpi_config.get_config("pi0_libero")


def _run_rollout(env, init_state, variant, policy, rng, max_timesteps, query_freq, pi0_action_horizon, action_dim):
    env.reset()
    obs = env.set_init_state(init_state)

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    actions = None
    for t in range(max_timesteps):
        if t % query_freq == 0:
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, pi0_action_horizon, action_dim))
            actions = policy.infer(obs_pi_zero, noise=noise)["actions"]

        action = actions[t % query_freq]
        obs, reward, done, _ = env.step(action)
        if done:
            return True, t + 1, rng

    return False, max_timesteps, rng


def evaluate_task(
    task_id,
    task,
    task_suite,
    policy,
    args,
    openpi_train_config,
    rng,
):
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = task_suite.get_task_init_states(task_id)

    variant = types.SimpleNamespace(
        env="libero",
        task_description=task_description,
        resize_image=-1,
    )

    successes = []
    episode_lens = []
    for rollout_id in range(args.num_rollouts):
        init_state = init_states[rollout_id % len(init_states)]
        is_success, episode_len, rng = _run_rollout(
            env,
            init_state,
            variant,
            policy,
            rng,
            args.max_timesteps,
            args.query_freq,
            openpi_train_config.model.action_horizon,
            policy.action_dim,
        )
        successes.append(is_success)
        episode_lens.append(episode_len)
        print(
            f"  rollout {rollout_id + 1}/{args.num_rollouts}: "
            f"success={is_success}, len={episode_len}",
            flush=True,
        )

    env.close()
    success_rate = float(np.mean(successes))
    return rng, {
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task_description,
        "success_rate": success_rate,
        "successes": int(np.sum(successes)),
        "num_rollouts": args.num_rollouts,
        "avg_episode_len": float(np.mean(episode_lens)),
    }


def evaluate_suite(args):
    if args.libero_suite not in LIBERO_MAX_TIMESTEPS:
        valid = ", ".join(sorted(LIBERO_MAX_TIMESTEPS))
        raise ValueError(f"Unknown suite {args.libero_suite!r}. Choose one of: {valid}")

    args.max_timesteps = LIBERO_MAX_TIMESTEPS[args.libero_suite]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    if len(args.task_ids) > 0:
        num_tasks = len(args.task_ids)
    else:
        num_tasks = task_suite.n_tasks

    openpi_train_config = _resolve_openpi_config(args.pi0_checkpoint)
    checkpoint_dir = _load_pi0_checkpoint(args.pi0_checkpoint)
    norm_stats = load_norm_stats_for_checkpoint(args.pi0_checkpoint)
    policy = policy_config.create_trained_policy(
        openpi_train_config, checkpoint_dir, norm_stats=norm_stats
    )
    print(f"Loaded policy from {checkpoint_dir}", flush=True)
    print(
        f"Evaluating {args.pi0_checkpoint} on {args.libero_suite} "
        f"({num_tasks} tasks, {args.num_rollouts} rollouts/task)",
        flush=True,
    )

    rng = jax.random.PRNGKey(args.seed)
    task_results = []
    if len(args.task_ids) > 0:
        for task_id in args.task_ids:
            task = task_suite.get_task(task_id)
            print(f"\nTask {task_id}/{num_tasks - 1}: {task.language}", flush=True)
            rng, task_result = evaluate_task(
                task_id,
                task,
                task_suite,
                policy,
                args,
                openpi_train_config,
                rng,
            )
            task_results.append(task_result)
            print(
                f"Task success rate: {task_result['success_rate']:.1%} "
                f"({task_result['successes']}/{task_result['num_rollouts']})",
                flush=True,
            )
        overall_success_rate = float(np.mean([r["success_rate"] for r in task_results]))
    else:
        for task_id in tqdm(range(num_tasks), desc="tasks"):
            task = task_suite.get_task(task_id)
            print(f"\nTask {task_id}/{num_tasks - 1}: {task.language}", flush=True)
            rng, task_result = evaluate_task(
                task_id,
                task,
                task_suite,
                policy,
                args,
                openpi_train_config,
                rng,
            )
            task_results.append(task_result)
            print(
                f"Task success rate: {task_result['success_rate']:.1%} "
                f"({task_result['successes']}/{task_result['num_rollouts']})",
                flush=True,
            )

    overall_success_rate = float(np.mean([r["success_rate"] for r in task_results]))
    summary = {
        "suite": args.libero_suite,
        "checkpoint": args.pi0_checkpoint,
        "openpi_config": openpi_train_config.name,
        "checkpoint_dir": str(checkpoint_dir),
        "num_rollouts": args.num_rollouts,
        "query_freq": args.query_freq,
        "seed": args.seed,
        "max_timesteps": args.max_timesteps,
        "overall_success_rate": overall_success_rate,
        "num_tasks": num_tasks,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tasks": task_results,
    }

    print("\n" + "=" * 80)
    print(f"Suite: {args.libero_suite}")
    print(f"Overall success rate: {overall_success_rate:.1%}")
    print("Per-task success rates:")
    for result in task_results:
        print(
            f"  [{result['task_id']:3d}] {result['success_rate']:6.1%}  {result['task_description']}"
        )
    print("=" * 80)

    if args.output_json:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote results to {output_path}", flush=True)

    return summary


def parse_args(argv=None):
    suites = ", ".join(sorted(LIBERO_MAX_TIMESTEPS))
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0 / Pi0.5 on all tasks in a LIBERO suite."
    )
    parser.add_argument(
        "--libero_suite",
        default="libero_90",
        choices=sorted(LIBERO_MAX_TIMESTEPS.keys()),
        help=f"LIBERO task suite ({suites})",
    )
    parser.add_argument(
        "--num_rollouts",
        default=10,
        type=int,
        help="Number of evaluation rollouts per task",
    )
    parser.add_argument(
        "--query_freq",
        default=5,
        type=int,
        help="Replan every N environment steps (should match pi0 action chunk usage)",
    )
    parser.add_argument(
        "--pi0_checkpoint",
        default="pi05_libero",
        type=str,
        help=(
            "Checkpoint preset or local path. Presets: "
            + ", ".join(sorted(CHECKPOINTS.keys()))
        ),
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--output_json",
        default="",
        type=str,
        help="Optional path to write JSON results",
    )
    parser.add_argument(
        "--task_ids",
        default=[],
        type=int,
        nargs="+",
        help="Optional list of task IDs to evaluate",
    )
    return parser.parse_args(argv)


def main(argv=None):
    # Match train_sim XLA settings for GPU performance.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    args = parse_args(argv)
    if not args.output_json:
        args.output_json = (
            f"logs/eval_{args.pi0_checkpoint}_{args.libero_suite}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    evaluate_suite(args)


if __name__ == "__main__":
    main(sys.argv[1:])
