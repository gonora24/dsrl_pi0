#!/usr/bin/env python
"""Evaluate Pi0 on gym_aloha tasks (TransferCube, Insertion) with video saving."""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

import gymnasium as gym
import gym_aloha  # noqa: F401 — registers gym_aloha envs as side effect
import imageio
import jax
import numpy as np
from gymnasium.envs.registration import register
from openpi_client import image_tools
from tqdm import tqdm

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

ALOHA_TASKS = {
    "transfer_cube": {
        "gym_id": "gym_aloha/AlohaTransferCube-v0",
        "entry_point": "gym_aloha.env:AlohaEnv",
        "task_kwarg": "transfer_cube",
        "prompt": "Transfer cube",
        "max_steps": 400,
    },
    "insertion": {
        "gym_id": "gym_aloha/AlohaInsertion-v0",
        "entry_point": "gym_aloha.env:AlohaEnv",
        "task_kwarg": "insertion",
        "prompt": "Insertion",
        "max_steps": 400,
    },
}

# Maximum staged reward that marks task success.
ALOHA_SUCCESS_REWARD = 4

# Render FPS used for saved videos (matches gym_aloha metadata).
VIDEO_FPS = 50


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_env(task_name: str, seed: int) -> gym.Env:
    """Create and return a gym_aloha environment for *task_name*."""
    spec = ALOHA_TASKS[task_name]
    # Re-register with extended episode length so we control truncation here.
    gym_id = spec["gym_id"]
    try:
        register(
            id=gym_id,
            entry_point=spec["entry_point"],
            max_episode_steps=spec["max_steps"],
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": spec["task_kwarg"]},
        )
    except gym.error.Error:
        # Already registered (e.g. by gym_aloha's own __init__); fine to reuse.
        pass

    env = gym.make(gym_id, obs_type="pixels_agent_pos", render_mode="rgb_array")
    return env


def _obs_to_policy_input(obs: dict) -> dict:
    """Convert a raw gym_aloha observation to the format expected by Pi0."""
    img = np.ascontiguousarray(obs["pixels"]["top"])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return {
        "state": obs["agent_pos"],
        "images": {"cam_high": img},
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(pi0_checkpoint: str) -> pathlib.Path:
    """Resolve *pi0_checkpoint* to a local directory, downloading if needed."""
    if pi0_checkpoint == "pi0_aloha_sim":
        return pathlib.Path(
            download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
        )
    path = pathlib.Path(pi0_checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"--pi0_checkpoint is not a directory and not a known preset: {path}"
        )
    return path


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _run_rollout(
    env: gym.Env,
    seed: int,
    policy,
    rng,
    max_timesteps: int,
    query_freq: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[bool, int, list[np.ndarray], object]:
    """Run a single rollout.

    Returns:
        (is_success, episode_len, frames, rng)
        where *frames* is a list of HxWx3 uint8 arrays.
    """
    obs, _ = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions = None
    is_success = False

    for t in range(max_timesteps):
        # Collect raw pixel frame before stepping (includes initial state).
        frames.append(obs["pixels"]["top"].copy())

        if t % query_freq == 0:
            obs_pi_zero = _obs_to_policy_input(obs)
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, action_horizon, action_dim))
            actions = policy.infer(obs_pi_zero, noise=noise)["actions"]

        action = actions[t % query_freq]
        obs, reward, terminated, truncated, _ = env.step(action)

        if reward >= ALOHA_SUCCESS_REWARD:
            is_success = True

        if terminated or truncated:
            # Append final frame.
            frames.append(obs["pixels"]["top"].copy())
            return is_success, t + 1, frames, rng

    return is_success, max_timesteps, frames, rng


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def _save_video(frames: list[np.ndarray], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=VIDEO_FPS)


def evaluate_task(
    task_name: str,
    policy,
    args,
    openpi_train_config,
    rng,
    output_dir: pathlib.Path,
) -> tuple[object, dict]:
    spec = ALOHA_TASKS[task_name]
    max_timesteps = spec["max_steps"]
    action_horizon = openpi_train_config.model.action_horizon

    video_dir = output_dir / task_name
    video_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(task_name, seed=args.seed)

    successes = []
    episode_lens = []

    for rollout_id in tqdm(range(args.num_rollouts), desc=task_name, leave=False):
        rollout_seed = args.seed + rollout_id
        is_success, episode_len, frames, rng = _run_rollout(
            env=env,
            seed=rollout_seed,
            policy=policy,
            rng=rng,
            max_timesteps=max_timesteps,
            query_freq=args.query_freq,
            action_horizon=action_horizon,
            action_dim=policy.action_dim,
        )
        successes.append(is_success)
        episode_lens.append(episode_len)

        outcome = "success" if is_success else "fail"
        video_path = video_dir / f"rollout_{rollout_id:03d}_{outcome}.mp4"
        _save_video(frames, video_path)

        print(
            f"  [{task_name}] rollout {rollout_id + 1}/{args.num_rollouts}: "
            f"success={is_success}, len={episode_len}, video={video_path}",
            flush=True,
        )

    env.close()
    success_rate = float(np.mean(successes))
    return rng, {
        "task_name": task_name,
        "task_prompt": spec["prompt"],
        "success_rate": success_rate,
        "successes": int(np.sum(successes)),
        "num_rollouts": args.num_rollouts,
        "avg_episode_len": float(np.mean(episode_lens)),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args) -> dict:
    openpi_train_config = openpi_config.get_config("pi0_aloha_sim")
    checkpoint_dir = _load_checkpoint(args.pi0_checkpoint)
    policy = policy_config.create_trained_policy(
        openpi_train_config, checkpoint_dir, norm_stats=None
    )
    print(f"Loaded policy from {checkpoint_dir}", flush=True)
    print(
        f"Evaluating on tasks: {args.tasks} "
        f"({args.num_rollouts} rollouts/task, query_freq={args.query_freq})",
        flush=True,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(args.seed)
    task_results = []

    for task_name in tqdm(args.tasks, desc="tasks"):
        print(f"\nTask: {task_name}", flush=True)
        rng, task_result = evaluate_task(
            task_name=task_name,
            policy=policy,
            args=args,
            openpi_train_config=openpi_train_config,
            rng=rng,
            output_dir=output_dir,
        )
        task_results.append(task_result)
        print(
            f"[{task_name}] success rate: {task_result['success_rate']:.1%} "
            f"({task_result['successes']}/{task_result['num_rollouts']})",
            flush=True,
        )

    overall_success_rate = float(np.mean([r["success_rate"] for r in task_results]))
    summary = {
        "checkpoint": args.pi0_checkpoint,
        "checkpoint_dir": str(checkpoint_dir),
        "tasks": args.tasks,
        "num_rollouts": args.num_rollouts,
        "query_freq": args.query_freq,
        "seed": args.seed,
        "overall_success_rate": overall_success_rate,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task_results": task_results,
    }

    print("\n" + "=" * 70)
    print(f"Overall success rate: {overall_success_rate:.1%}")
    for r in task_results:
        print(f"  {r['task_name']:20s}  {r['success_rate']:6.1%}  ({r['successes']}/{r['num_rollouts']})")
    print("=" * 70)

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote results to {results_path}", flush=True)
    print(f"Videos saved to  {output_dir}/", flush=True)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    valid_tasks = sorted(ALOHA_TASKS.keys())
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0 on gym_aloha tasks with video saving."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=valid_tasks,
        choices=valid_tasks,
        help=f"Tasks to evaluate. Choices: {valid_tasks}. Default: all.",
    )
    parser.add_argument(
        "--num_rollouts",
        default=10,
        type=int,
        help="Number of evaluation rollouts per task.",
    )
    parser.add_argument(
        "--query_freq",
        default=10,
        type=int,
        help="Replan every N environment steps (should match action chunk usage).",
    )
    parser.add_argument(
        "--pi0_checkpoint",
        default="pi0_aloha_sim",
        type=str,
        help='Checkpoint preset ("pi0_aloha_sim") or local directory path.',
    )
    parser.add_argument("--seed", default=0, type=int, help="Base random seed.")
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Directory for videos and results.json. Auto-generated if empty.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    args = parse_args(argv)
    if not args.output_dir:
        args.output_dir = (
            f"logs/eval_aloha_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
