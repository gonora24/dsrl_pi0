#!/usr/bin/env python
"""Multimodality evaluation for pi0.5 on a LIBERO frying-pan task.

Three experiment parts, each with --num_rollouts rollouts:
  Part 1 – Standard rollout (default LIBERO init state).
  Part 2 – Pan handle rotated by --pan_rotation_deg degrees around world Z.
  Part 3 – Target stove shifted by (--target_offset_x, --target_offset_y) metres.

Videos and a JSON summary are written to:
  <output_dir>/multimodality_eval/<timestamp>/
"""

import argparse
import json
import math
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
from examples.train_utils_sim import obs_to_pi_zero_input

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10
MAX_TIMESTEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Quaternion helpers (MuJoCo convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def _quat_multiply(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _yaw_quat(angle_deg):
    """Unit quaternion [w,x,y,z] for a pure yaw (Z-axis) rotation."""
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


# ---------------------------------------------------------------------------
# Sim-level state manipulators
# ---------------------------------------------------------------------------

def _apply_pan_rotation(env, rotation_deg):
    """Rotate the frying pan's yaw by *rotation_deg* in the current sim state.

    Must be called after env.set_init_state().  Returns the updated observation.
    """
    sim = env.env.sim
    pan_joint_id = None
    for j in range(sim.model.njnt):
        try:
            name = sim.model.joint_id2name(j)
        except Exception:
            name = ""
        if "chefmate_8_frypan" in name:
            pan_joint_id = j
            break
        if "black_book_1" in name:
            pan_joint_id = j
            break
        if "red_coffee_mug_1" in name:
            pan_joint_id = j
            break
        if "akita_black_bowl_1" in name:
            pan_joint_id = j
            break

    if pan_joint_id is None:
        # Fallback: scan by joint type (free == 0 in MuJoCo) near known body
        raise RuntimeError(
            "Could not locate a joint containing 'chefmate_8_frypan' or 'black_book_1' in the sim model. "
            "Check that task_id points to a frying-pan or black book task."
        )

    qadr = int(sim.model.jnt_qposadr[pan_joint_id])
    # Free-joint qpos layout: [x, y, z, qw, qx, qy, qz]
    current_quat = np.array(sim.data.qpos[qadr + 3: qadr + 7])
    rot = _yaw_quat(rotation_deg)
    new_quat = _quat_multiply(rot, current_quat)
    new_quat /= np.linalg.norm(new_quat)
    sim.data.qpos[qadr + 3: qadr + 7] = new_quat
    sim.forward()
    env.env._update_observables(force=True)
    return env.env._get_observations()


def _apply_target_shift(env, offset_x, offset_y):
    """Translate the flat_stove_1 body in the model by (offset_x, offset_y).

    Because the BDDL cook_region is anchored to flat_stove_1, the success
    predicate moves with the stove, so evaluation remains valid.
    Must be called after env.set_init_state().  Returns the updated observation.
    """
    sim = env.env.sim
    stove_body_id = None
    for b in range(sim.model.nbody):
        try:
            name = sim.model.body_id2name(b)
        except Exception:
            name = ""
        if "flat_stove_1" in name:
            stove_body_id = b
            break
        if "desk_caddy_1" in name:
            stove_body_id = b
            break

    if stove_body_id is None:
        raise RuntimeError(
            "Could not locate body 'flat_stove_1' or 'desk_caddy_1' in the sim model. "
            "Check that task_id points to a task with a flat stove or desk caddy."
        )

    sim.model.body_pos[stove_body_id][0] += offset_x
    sim.model.body_pos[stove_body_id][1] += offset_y
    sim.forward()
    env.env._update_observables(force=True)
    return env.env._get_observations()


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------

def _run_rollout(env, obs, variant, policy, rng, max_steps, query_freq,
                 action_horizon, action_dim):
    """Run one rollout from the given initial *obs*.

    Returns:
        success (bool), frames (list of H×W×3 uint8), actions (list of 1-D arrays), rng
    """
    # warm-up dummy steps already happened before this call (via set_init_state)
    frames = []
    all_actions = []
    pi_actions = None

    for t in range(max_steps):
        frame = obs["agentview_image"][::-1, ::-1].copy()  # flip to upright
        frames.append(frame)

        if t % query_freq == 0:
            obs_pi = obs_to_pi_zero_input(obs, variant)
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (1, action_horizon, action_dim))
            pi_actions = policy.infer(obs_pi, noise=noise)["actions"]

        action = pi_actions[t % query_freq]
        all_actions.append(np.array(action))
        obs, _reward, done, _info = env.step(action)

        if done:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())
            return True, frames, all_actions, rng

    return False, frames, all_actions, rng


# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

def _reset_env(env, init_state):
    """Reset env and apply init_state; run dummy wait steps; return obs."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def run_part(
    part_label,
    part_dir,
    env,
    init_states,
    variant,
    policy,
    rng,
    num_rollouts,
    max_steps,
    query_freq,
    action_horizon,
    action_dim,
    *,
    pan_rotation_deg=None,
    target_offset_x=None,
    target_offset_y=None,
):
    """Execute *num_rollouts* for one experiment condition.

    Returns a list of per-rollout result dicts.
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i in tqdm(range(num_rollouts), desc=part_label, unit="rollout"):
        init_state = init_states[i % len(init_states)]
        env.reset()
        obs = env.set_init_state(init_state)

        # Apply condition-specific perturbation *before* wait steps so the
        # robot settles in the perturbed configuration.
        if pan_rotation_deg is not None:
            obs = _apply_pan_rotation(env, pan_rotation_deg)
        if target_offset_x is not None:
            obs = _apply_target_shift(env, target_offset_x, target_offset_y)

        # Settle with dummy actions
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        success, frames, actions, rng = _run_rollout(
            env, obs, variant, policy, rng,
            max_steps, query_freq, action_horizon, action_dim,
        )

        video_name = f"rollout_{i:03d}_success_{int(success)}.mp4"
        video_path = part_dir / video_name
        imageio.mimwrite(str(video_path), frames, fps=30, macro_block_size=1)

        results.append({
            "rollout_id": i,
            "init_state_idx": i % len(init_states),
            "success": success,
            "episode_len": len(frames),
            "video_path": str(video_path),
            "actions": [a.tolist() for a in actions],
        })

        print(
            f"  [{part_label}] rollout {i + 1}/{num_rollouts}: "
            f"success={success}, len={len(frames)}",
            flush=True,
        )

    return results, rng


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

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
    run_dir = base_output / f"{timestamp}_{args.libero_suite}_task_{args.task_id}_{args.pi0_checkpoint}"
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

    # --- load policy ---
    cfg = openpi_config.get_config(CHECKPOINTS[args.pi0_checkpoint]["config"])
    checkpoint_dir = load_pi0_checkpoint(args.pi0_checkpoint)
    norm_stats = load_norm_stats_for_checkpoint(args.pi0_checkpoint)
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir, norm_stats=norm_stats)
    print(f"Loaded policy from {checkpoint_dir}", flush=True)

    action_horizon = cfg.model.action_horizon
    action_dim = policy.action_dim

    variant = types.SimpleNamespace(
        env="libero",
        task_description=task_description,
        resize_image=-1,
    )

    rng = jax.random.PRNGKey(args.seed)
    summary_parts = {}

    # -----------------------------------------------------------------------
    # Part 1 – standard
    # -----------------------------------------------------------------------
    if 1 in args.parts:
        label = "part1_standard"
        results, rng = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            rng,
            args.num_rollouts,
            max_steps,
            args.query_freq,
            action_horizon,
            action_dim,
        )
        success_rate = float(np.mean([r["success"] for r in results]))
        summary_parts[label] = {
            "description": "Standard pi0.5 rollouts (no perturbation)",
            "success_rate": success_rate,
            "rollouts": results,
        }
        print(f"[Part 1] success rate: {success_rate:.1%}", flush=True)

    # -----------------------------------------------------------------------
    # Part 2 – pan handle rotated
    # -----------------------------------------------------------------------
    if 2 in args.parts:
        label = f"part2_pan_rotated_{int(args.pan_rotation_deg)}deg"
        results, rng = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            rng,
            args.num_rollouts,
            max_steps,
            args.query_freq,
            action_horizon,
            action_dim,
            pan_rotation_deg=args.pan_rotation_deg,
        )
        success_rate = float(np.mean([r["success"] for r in results]))
        summary_parts[label] = {
            "description": (
                f"Pan handle rotated by {args.pan_rotation_deg} deg around Z-axis"
            ),
            "pan_rotation_deg": args.pan_rotation_deg,
            "success_rate": success_rate,
            "rollouts": results,
        }
        print(f"[Part 2] success rate: {success_rate:.1%}", flush=True)

    # -----------------------------------------------------------------------
    # Part 3 – target location shifted
    # -----------------------------------------------------------------------
    if 3 in args.parts:
        label = (
            f"part3_target_shifted_{args.target_offset_x:.3f}_{args.target_offset_y:.3f}"
        )
        results, rng = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            rng,
            args.num_rollouts,
            max_steps,
            args.query_freq,
            action_horizon,
            action_dim,
            target_offset_x=args.target_offset_x,
            target_offset_y=args.target_offset_y,
        )
        success_rate = float(np.mean([r["success"] for r in results]))
        summary_parts[label] = {
            "description": (
                f"Stove (target) shifted by dx={args.target_offset_x} m, "
                f"dy={args.target_offset_y} m"
            ),
            "target_offset_x": args.target_offset_x,
            "target_offset_y": args.target_offset_y,
            "success_rate": success_rate,
            "rollouts": results,
        }
        print(f"[Part 3] success rate: {success_rate:.1%}", flush=True)

    env.close()

    # -----------------------------------------------------------------------
    # Write summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "timestamp": timestamp,
        "libero_suite": args.libero_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_description": task_description,
        "pi0_checkpoint": args.pi0_checkpoint,
        "num_rollouts": args.num_rollouts,
        "query_freq": args.query_freq,
        "seed": args.seed,
        "max_steps": max_steps,
        "parts": summary_parts,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}", flush=True)

    # -----------------------------------------------------------------------
    # Print compact table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MULTIMODALITY EVALUATION RESULTS")
    print(f"Task: {task_description}")
    print("-" * 60)
    for part_key, part_data in summary_parts.items():
        sr = part_data["success_rate"]
        n = args.num_rollouts
        print(f"  {part_key}")
        print(f"    {part_data['description']}")
        print(f"    Success: {sr:.1%}  ({int(round(sr * n))}/{n})")
    print("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Multimodality evaluation for pi0.5 on LIBERO frying-pan task."
    )
    parser.add_argument("--libero_suite", default="libero_90",
                        choices=sorted(MAX_TIMESTEPS.keys()))
    parser.add_argument("--task_id", default=18, type=int,
                        help="LIBERO task index (default 18 = put_the_frying_pan_on_the_stove)")
    parser.add_argument("--num_rollouts", default=50, type=int,
                        help="Number of rollouts per experiment part")
    parser.add_argument("--pi0_checkpoint", default="pi05_libero",
                        choices=sorted(CHECKPOINTS.keys()))
    parser.add_argument("--query_freq", default=5, type=int,
                        help="Re-plan every N env steps")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pan_rotation_deg", default=90.0, type=float,
                        help="Yaw rotation applied to frying pan handle (Part 2)")
    parser.add_argument("--target_offset_x", default=0.15, type=float,
                        help="Stove X-shift in metres (Part 3)")
    parser.add_argument("--target_offset_y", default=0.0, type=float,
                        help="Stove Y-shift in metres (Part 3)")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.environ.get("OUTPUT_DIR", "./logs"), "multimodality_eval"
        ),
        help="Root directory for videos and summary JSON",
    )
    parser.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="Which experiment parts to run (default: 1 2 3)")
    return parser.parse_args(argv)


def main(argv=None):
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
