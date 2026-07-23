#!/usr/bin/env python
#SBATCH --job-name=eval_xvla_libero_10
"""Evaluate pretrained XVLA on all tasks in a LIBERO suite."""

import argparse
import json
import pathlib
import sys
import types
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from examples.train_utils_sim import obs_to_xvla_input, prepare_libero_episode_for_xvla
from examples.xvla_policy import XVLAPolicy

LIBERO_ENV_RESOLUTION = 256
XVLA_MAX_TIMESTEPS = 800

LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


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


def _run_rollout(env, init_state, variant, policy, max_timesteps, query_freq):
    env.reset()
    env.set_init_state(init_state)
    policy.reset()
    obs = prepare_libero_episode_for_xvla(env)

    actions = None
    for t in range(max_timesteps):
        if t % query_freq == 0:
            obs_xvla = obs_to_xvla_input(obs, variant, env=env)
            actions = policy.infer(
                obs_xvla,
                noise=None,
                proprio_from_step=query_freq - 1,
            )["actions"]

        action = actions[t % query_freq]
        obs, reward, done, _ = env.step(action)
        if done:
            return True, t + 1

    return False, max_timesteps


def evaluate_task(task_id, task, task_suite, policy, args):
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = task_suite.get_task_init_states(task_id)

    variant = types.SimpleNamespace(
        env="libero",
        vla="xvla",
        task_description=task_description,
        resize_image=-1,
    )

    successes = []
    episode_lens = []
    for rollout_id in range(args.num_rollouts):
        init_state = init_states[rollout_id % len(init_states)]
        is_success, episode_len = _run_rollout(
            env,
            init_state,
            variant,
            policy,
            args.max_timesteps,
            args.query_freq,
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
    return {
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task_description,
        "success_rate": success_rate,
        "successes": int(np.sum(successes)),
        "num_rollouts": args.num_rollouts,
        "avg_episode_len": float(np.mean(episode_lens)),
    }


def evaluate_suite(args):
    if args.libero_suite not in LIBERO_SUITES:
        valid = ", ".join(LIBERO_SUITES)
        raise ValueError(f"Unknown suite {args.libero_suite!r}. Choose one of: {valid}")

    args.max_timesteps = XVLA_MAX_TIMESTEPS

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    num_tasks = task_suite.n_tasks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = XVLAPolicy.from_pretrained(
        args.checkpoint,
        device=device,
        domain_id=args.domain_id,
        steps=args.steps,
    )
    # Absolute ee6d chunks must be executed in full before replan.
    if args.query_freq != policy.action_horizon:
        print(
            f"Warning: query_freq={args.query_freq} != action_horizon={policy.action_horizon}; "
            f"overriding query_freq to {policy.action_horizon}",
            flush=True,
        )
        args.query_freq = int(policy.action_horizon)

    print(f"Loaded XVLA from {args.checkpoint} (device={device})", flush=True)
    print(
        f"Evaluating XVLA on {args.libero_suite} "
        f"({num_tasks} tasks, {args.num_rollouts} rollouts/task, "
        f"query_freq={args.query_freq}, max_timesteps={args.max_timesteps})",
        flush=True,
    )

    task_results = []
    for task_id in tqdm(range(num_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        print(f"\nTask {task_id}/{num_tasks - 1}: {task.language}", flush=True)
        task_result = evaluate_task(task_id, task, task_suite, policy, args)
        task_results.append(task_result)
        print(
            f"Task success rate: {task_result['success_rate']:.1%} "
            f"({task_result['successes']}/{task_result['num_rollouts']})",
            flush=True,
        )

    overall_success_rate = float(np.mean([r["success_rate"] for r in task_results]))
    summary = {
        "suite": args.libero_suite,
        "checkpoint": args.checkpoint,
        "domain_id": args.domain_id,
        "steps": args.steps,
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
    suites = ", ".join(LIBERO_SUITES)
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained XVLA on all tasks in a LIBERO suite."
    )
    parser.add_argument(
        "--libero_suite",
        default="libero_90",
        choices=list(LIBERO_SUITES),
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
        default=30,
        type=int,
        help="Replan every N environment steps (should match XVLA action_horizon)",
    )
    parser.add_argument(
        "--checkpoint",
        default="2toINF/X-VLA-Libero",
        type=str,
        help="HuggingFace model id or local path for XVLA",
    )
    parser.add_argument(
        "--domain_id",
        default=3,
        type=int,
        help="XVLA domain id (3 = Libero)",
    )
    parser.add_argument(
        "--steps",
        default=10,
        type=int,
        help="Number of diffusion steps for XVLA generate_actions",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--output_json",
        default="",
        type=str,
        help="Optional path to write JSON results",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.output_json:
        ckpt_tag = args.checkpoint.replace("/", "_")
        args.output_json = (
            f"logs/eval_xvla_{ckpt_tag}_{args.libero_suite}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    evaluate_suite(args)


if __name__ == "__main__":
    main(sys.argv[1:])
