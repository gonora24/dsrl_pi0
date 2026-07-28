import argparse
import json
import os
import pathlib
import sys
import types
from datetime import datetime

import imageio
import jax
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.training import config as openpi_config

from examples.train_sim import CHECKPOINTS, load_norm_stats_for_checkpoint, load_pi0_checkpoint
from examples.train_utils_sim import obs_to_pi_zero_input, obs_to_img, _prepare_pi0_noise, obs_to_qpos
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10
MAX_TIMESTEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _run_rollout(env, obs, variant, noise_actor_variant, policy, noise_actor,
                 max_steps, query_freq, action_horizon, query_repeat):
    """Run one rollout from the given initial *obs*.

    The noise actor is queried at the first chunk and then every *query_repeat*
    chunks.  Between re-queries the same noise vector is fed to the pi0
    denoising process.

    *variant* is used for pi0 observation formatting (obs_to_pi_zero_input).
    *noise_actor_variant* is used for the noise actor's image observation
    (obs_to_img) and must have resize_image set to the value the noise actor
    was trained with.

    Returns:
        success (bool), frames (list of H×W×3 uint8), actions (list of 1-D arrays)
    """
    frames = []
    all_actions = []
    pi_actions = None
    noise = None  # will be set on the first chunk boundary

    for t in range(max_steps):
        frame = obs["agentview_image"][::-1, ::-1].copy()
        frames.append(frame)

        chunk_idx = t // query_freq  # monotonically increasing chunk counter

        if t % query_freq == 0:
            obs_pi = obs_to_pi_zero_input(obs, variant)

            # Refresh noise every query_repeat chunks (and always on chunk 0).
            if noise is None or chunk_idx % query_repeat == 0:
                print(f"New Noise! query_repeat: {query_repeat}, chunk_idx: {chunk_idx}", flush=True)
                curr_image = obs_to_img(obs, noise_actor_variant)
                qpos = obs_to_qpos(obs, noise_actor_variant)
                obs_dict = {"pixels": curr_image[np.newaxis, ..., np.newaxis], "qpos": qpos[np.newaxis, ...]}
                actions_noise = noise_actor.sample_actions(obs_dict)
                noise = _prepare_pi0_noise(actions_noise, noise_actor, action_horizon)

            pi_actions = policy.infer(obs_pi, noise=noise)["actions"]

        action = pi_actions[t % query_freq]
        all_actions.append(np.array(action))
        obs, _reward, done, _info = env.step(action)

        if done:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())
            return True, frames, all_actions

    return False, frames, all_actions


def evaluate(args):
    if args.libero_suite not in MAX_TIMESTEPS:
        raise ValueError(
            f"Unknown suite {args.libero_suite!r}. "
            f"Choose from: {', '.join(sorted(MAX_TIMESTEPS))}"
        )
    max_steps = MAX_TIMESTEPS[args.libero_suite]

    # --- resolve output directory ---
    base_output = pathlib.Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output / (
        f"{timestamp}_{args.libero_suite}_task{args.task_id}"
        f"_{args.pi0_checkpoint}_repeat{args.query_repeat}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}", flush=True)

    # --- load task ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    init_states = task_suite.get_task_init_states(args.task_id)
    print(f"Task {args.task_id}: {task_description}", flush=True)
    print(f"Init states available: {len(init_states)}", flush=True)

    # --- build env ---
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)

    # --- load pi0 policy ---
    cfg = openpi_config.get_config(CHECKPOINTS[args.pi0_checkpoint]["config"])
    checkpoint_dir = load_pi0_checkpoint(args.pi0_checkpoint)
    norm_stats = load_norm_stats_for_checkpoint(args.pi0_checkpoint)
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir, norm_stats=norm_stats)
    print(f"Loaded pi0 policy from {checkpoint_dir}", flush=True)

    action_horizon = cfg.model.action_horizon

    # --- load noise actor ---
    noise_actor = PixelSACLearner.restore_from_checkpoint_dir(args.noise_actor_dir)
    print(f"Loaded noise actor from {args.noise_actor_dir}", flush=True)

    # variant for pi0 policy (obs_to_pi_zero_input handles its own image sizing)
    variant = types.SimpleNamespace(
        env="libero",
        task_description=task_description,
        resize_image=-1,
        add_states=False,
    )

    # noise actor is always trained on 64×64 images; obs_to_img handles the resize
    noise_actor_variant = types.SimpleNamespace(
        env="libero",
        resize_image=64,
        add_states=False,
    )

    # --- rollouts ---
    results = []

    for i in tqdm(range(args.num_rollouts), desc="rollouts", unit="rollout"):
        init_state = init_states[i % len(init_states)]
        env.reset()
        obs = env.set_init_state(init_state)

        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        success, frames, actions = _run_rollout(
            env, obs, variant, noise_actor_variant, policy, noise_actor,
            max_steps, args.query_freq, action_horizon, args.query_repeat,
        )

        video_name = f"rollout_{i:03d}_success_{int(success)}.mp4"
        video_path = run_dir / video_name
        imageio.mimwrite(str(video_path), frames, fps=30, macro_block_size=1)

        results.append({
            "rollout_id": i,
            "init_state_idx": i % len(init_states),
            "success": success,
            "episode_len": len(frames),
            "video_path": str(video_path),
        })

        print(
            f"  rollout {i + 1}/{args.num_rollouts}: "
            f"success={success}, len={len(frames)}",
            flush=True,
        )

    env.close()

    success_rate = float(np.mean([r["success"] for r in results]))

    # --- write summary JSON ---
    summary = {
        "timestamp": timestamp,
        "libero_suite": args.libero_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_description": task_description,
        "pi0_checkpoint": args.pi0_checkpoint,
        "noise_actor_dir": args.noise_actor_dir,
        "num_rollouts": args.num_rollouts,
        "query_freq": args.query_freq,
        "query_repeat": args.query_repeat,
        "seed": args.seed,
        "max_steps": max_steps,
        "success_rate": success_rate,
        "rollouts": results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}", flush=True)

    # --- compact result table ---
    n = args.num_rollouts
    print("\n" + "=" * 60)
    print("NOISE REPEAT EVALUATION RESULTS")
    print(f"Task: {task_description}")
    print(f"Noise actor: {args.noise_actor_dir}")
    print(f"query_freq={args.query_freq}, query_repeat={args.query_repeat}")
    print("-" * 60)
    print(f"  Success: {success_rate:.1%}  ({int(round(success_rate * n))}/{n})")
    print("=" * 60)

    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pi0 with a DSRL noise actor whose predicted noise vector "
            "is repeated for query_repeat consecutive action chunks."
        )
    )
    parser.add_argument(
        "--noise_actor_dir", required=True,
        help="Path to PixelSACLearner checkpoint directory",
    )
    parser.add_argument(
        "--query_repeat", default=1, type=int,
        help=(
            "Number of consecutive action chunks that share the same noise vector. "
            "1 = re-query every chunk (baseline)."
        ),
    )
    parser.add_argument("--libero_suite", default="libero_90",
                        choices=sorted(MAX_TIMESTEPS.keys()))
    parser.add_argument("--task_id", default=0, type=int,
                        help="LIBERO task index within the chosen suite")
    parser.add_argument("--num_rollouts", default=20, type=int,
                        help="Number of evaluation rollouts")
    parser.add_argument("--pi0_checkpoint", default="pi05_libero",
                        choices=sorted(CHECKPOINTS.keys()))
    parser.add_argument("--query_freq", default=5, type=int,
                        help="Re-plan every N env steps (action chunk length)")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.environ.get("OUTPUT_DIR", "./logs"), "repeat_noise_eval"
        ),
        help="Root directory for videos and summary JSON",
    )
    return parser.parse_args(argv)


def main(argv=None):
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
