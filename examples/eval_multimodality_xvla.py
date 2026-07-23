#!/usr/bin/env python
"""Multimodality evaluation for XVLA on a LIBERO frying-pan task.

Three experiment parts, each with --num_rollouts rollouts:
  Part 1 – Standard rollout (default LIBERO init state).
  Part 2 – Pan handle rotated by --pan_rotation_deg degrees around world Z.
  Part 3 – Target stove shifted by (--target_offset_x, --target_offset_y) metres.

Videos and a JSON summary are written to:
  <output_dir>/<timestamp>_<suite>_task_<id>_<checkpoint_tag>/
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
import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from examples.train_utils_sim import obs_to_xvla_input, prepare_libero_episode_for_xvla
from examples.xvla_policy import XVLAPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_ENV_RESOLUTION = 256
XVLA_MAX_TIMESTEPS = 500

LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


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

    if pan_joint_id is None:
        raise RuntimeError(
            "Could not locate a joint containing 'chefmate_8_frypan' in the sim model. "
            "Check that task_id points to a frying-pan task."
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

    if stove_body_id is None:
        raise RuntimeError(
            "Could not locate body 'flat_stove_1' in the sim model. "
            "Check that task_id points to a task with a flat stove."
        )

    sim.model.body_pos[stove_body_id][0] += offset_x
    sim.model.body_pos[stove_body_id][1] += offset_y
    sim.forward()
    env.env._update_observables(force=True)
    return env.env._get_observations()


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------

def _run_rollout(env, obs, variant, policy, max_steps, query_freq):
    """Run one rollout from the given initial *obs*.

    Returns:
        success (bool), frames (list of H×W×3 uint8), actions (list of 1-D arrays)
    """
    frames = []
    all_actions = []
    actions = None

    for t in range(max_steps):
        frame = obs["agentview_image"][::-1, ::-1].copy()  # flip to upright
        frames.append(frame)

        if t % query_freq == 0:
            obs_xvla = obs_to_xvla_input(obs, variant, env=env)
            actions = policy.infer(
                obs_xvla,
                noise=None,
                proprio_from_step=query_freq - 1,
            )["actions"]

        action = actions[t % query_freq]
        all_actions.append(np.array(action))
        obs, _reward, done, _info = env.step(action)

        if done:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())
            return True, frames, all_actions

    return False, frames, all_actions


# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

def run_part(
    part_label,
    part_dir,
    env,
    init_states,
    variant,
    policy,
    num_rollouts,
    max_steps,
    query_freq,
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
        env.set_init_state(init_state)

        # Apply condition-specific perturbation *before* settle so the
        # robot settles in the perturbed configuration.
        if pan_rotation_deg is not None:
            _apply_pan_rotation(env, pan_rotation_deg)
        if target_offset_x is not None:
            _apply_target_shift(env, target_offset_x, target_offset_y)

        policy.reset()
        obs = prepare_libero_episode_for_xvla(env)

        success, frames, actions = _run_rollout(
            env, obs, variant, policy, max_steps, query_freq,
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

    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    if args.libero_suite not in LIBERO_SUITES:
        raise ValueError(
            f"Unknown suite {args.libero_suite!r}. "
            f"Choose from: {', '.join(LIBERO_SUITES)}"
        )
    max_steps = args.max_timesteps

    # --- resolve output directory ---
    ckpt_tag = args.checkpoint.replace("/", "_")
    base_output = pathlib.Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        base_output
        / f"{timestamp}_{args.libero_suite}_task_{args.task_id}_{ckpt_tag}"
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
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)

    # --- load policy ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = XVLAPolicy.from_pretrained(
        args.checkpoint,
        device=device,
        domain_id=args.domain_id,
        steps=args.steps,
    )
    if args.query_freq != policy.action_horizon:
        print(
            f"Warning: query_freq={args.query_freq} != action_horizon={policy.action_horizon}; "
            f"overriding query_freq to {policy.action_horizon}",
            flush=True,
        )
        args.query_freq = int(policy.action_horizon)
    print(f"Loaded XVLA from {args.checkpoint} (device={device})", flush=True)

    variant = types.SimpleNamespace(
        env="libero",
        vla="xvla",
        task_description=task_description,
        resize_image=-1,
    )

    summary_parts = {}

    # -----------------------------------------------------------------------
    # Part 1 – standard
    # -----------------------------------------------------------------------
    if 1 in args.parts:
        label = "part1_standard"
        results = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            args.num_rollouts,
            max_steps,
            args.query_freq,
        )
        success_rate = float(np.mean([r["success"] for r in results]))
        summary_parts[label] = {
            "description": "Standard XVLA rollouts (no perturbation)",
            "success_rate": success_rate,
            "rollouts": results,
        }
        print(f"[Part 1] success rate: {success_rate:.1%}", flush=True)

    # -----------------------------------------------------------------------
    # Part 2 – pan handle rotated
    # -----------------------------------------------------------------------
    if 2 in args.parts:
        label = f"part2_pan_rotated_{int(args.pan_rotation_deg)}deg"
        results = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            args.num_rollouts,
            max_steps,
            args.query_freq,
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
        results = run_part(
            label,
            run_dir / label,
            env,
            init_states,
            variant,
            policy,
            args.num_rollouts,
            max_steps,
            args.query_freq,
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
        "checkpoint": args.checkpoint,
        "domain_id": args.domain_id,
        "steps": args.steps,
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
    print("MULTIMODALITY EVALUATION RESULTS (XVLA)")
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
        description="Multimodality evaluation for XVLA on LIBERO frying-pan task."
    )
    parser.add_argument(
        "--libero_suite",
        default="libero_90",
        choices=list(LIBERO_SUITES),
    )
    parser.add_argument(
        "--task_id",
        default=18,
        type=int,
        help="LIBERO task index (default 18 = put_the_frying_pan_on_the_stove)",
    )
    parser.add_argument(
        "--num_rollouts",
        default=50,
        type=int,
        help="Number of rollouts per experiment part",
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
    parser.add_argument(
        "--query_freq",
        default=30,
        type=int,
        help="Re-plan every N env steps (overridden to action_horizon if mismatch)",
    )
    parser.add_argument(
        "--max_timesteps",
        default=XVLA_MAX_TIMESTEPS,
        type=int,
        help="Max environment steps per rollout",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--pan_rotation_deg",
        default=90.0,
        type=float,
        help="Yaw rotation applied to frying pan handle (Part 2)",
    )
    parser.add_argument(
        "--target_offset_x",
        default=0.15,
        type=float,
        help="Stove X-shift in metres (Part 3)",
    )
    parser.add_argument(
        "--target_offset_y",
        default=0.0,
        type=float,
        help="Stove Y-shift in metres (Part 3)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            os.environ.get("OUTPUT_DIR", "./logs"), "multimodality_eval"
        ),
        help="Root directory for videos and summary JSON",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which experiment parts to run (default: 1 2 3)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main(sys.argv[1:])
