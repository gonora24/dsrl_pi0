from tqdm import tqdm
import numpy as np
import wandb
import jax
import flax.traverse_util
from openpi_client import image_tools
import math
import PIL

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_NUM_STEPS_WAIT = 10


def prepare_libero_episode_for_xvla(env):
    """Settle with delta actions, then switch to absolute EEF control.

    Matches xvla/evaluation/libero/libero_client.py: dummy settle happens while
    ``use_delta=True``; absolute mode is enabled only afterward. Calling this
    after ``reset`` / ``set_init_state`` returns the post-settle observation.
    """
    for robot in env.env.robots:
        robot.controller.use_delta = True
    obs = None
    for _ in range(LIBERO_NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    for robot in env.env.robots:
        robot.controller.use_delta = False
    return obs


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def _pad_chunk_rewards(rewards, size, pad_value=0.0):
    if len(rewards) >= size:
        return np.asarray(rewards[:size], dtype=np.float32)
    return np.concatenate(
        [np.asarray(rewards, dtype=np.float32), np.full(size - len(rewards), pad_value, dtype=np.float32)]
    )


def _pad_chunk_terminations(terminations, size):
    if len(terminations) >= size:
        return np.asarray(terminations[:size], dtype=np.bool_)
    padded = np.zeros(size, dtype=np.bool_)
    padded[:len(terminations)] = np.asarray(terminations, dtype=np.bool_)
    if len(terminations) > 0:
        padded[len(terminations):] = True
    return padded


def _finalize_chunk_rewards(chunk_rewards, chunk_terminations, query_frequency):
    rewards = np.stack(
        [_pad_chunk_rewards(r, query_frequency) for r in chunk_rewards],
        axis=0,
    )
    terminations = np.stack(
        [_pad_chunk_terminations(t, query_frequency) for t in chunk_terminations],
        axis=0,
    )
    masks = np.logical_not(terminations.any(axis=-1)).astype(np.float32)
    return rewards, terminations, masks


def _prepare_pi0_noise(actions_noise, agent, pi0_action_horizon):
    """Reshape SAC noise and expand/pad/truncate to pi0's inference horizon.

    In multi-vector mode (noise_repeats_per_vector > 1) each of the N noise
    vectors is tiled K times: (1, N, 32) -> (1, N*K, 32).  If N*K is still
    shorter than pi0_action_horizon the last vector is padded; if longer the
    sequence is truncated.
    """
    if not agent.only_predict_dims_until > 0:
        actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
    noise = actions_noise[None]  # (1, N, 32)

    repeats = getattr(agent, 'noise_repeats_per_vector', 1)
    if repeats > 1:
        # Tile each vector K times along the horizon axis
        noise = np.repeat(noise, repeats, axis=1)  # (1, N*K, 32)

    if noise.shape[1] < pi0_action_horizon:
        pad = np.repeat(
            noise[:, -1:, :], pi0_action_horizon - noise.shape[1], axis=1
        )
        noise = np.concatenate([noise, pad], axis=1)
    elif noise.shape[1] > pi0_action_horizon:
        noise = noise[:, :pi0_action_horizon, :]
    return noise


def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0:
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image


def attach_libero_ee_pose(obs, env):
    """Attach live EE pose fields used by XVLA Libero proprio packing."""
    import pathlib
    import sys

    xvla_root = pathlib.Path(__file__).resolve().parents[1] / "xvla"
    if str(xvla_root) not in sys.path:
        sys.path.insert(0, str(xvla_root))
    from evaluation.libero.action_processor import LiberoAbsActionProcessor

    processor = LiberoAbsActionProcessor()
    robot = env.env.robots[0]
    obs = dict(obs)
    obs["robo_pos"] = np.asarray(robot.controller.ee_pos, dtype=np.float32)
    obs["robo_ori"] = processor.Mat_to_Rotate6D(
        np.asarray(robot.controller.ee_ori_mat)
    ).astype(np.float32)
    return obs


def obs_to_xvla_input(obs, variant, env=None):
    """Build XVLAPolicy.infer obs dict from raw Libero or replay-buffer obs."""
    if 'agentview_image' in obs:
        if env is not None and ('robo_pos' not in obs or 'robo_ori' not in obs):
            obs = attach_libero_ee_pose(obs, env)
        out = {
            "agentview_image": np.ascontiguousarray(obs["agentview_image"]),
            "robot0_eye_in_hand_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
            "prompt": str(variant.task_description),
        }
        if "robo_pos" in obs and "robo_ori" in obs:
            out["robo_pos"] = np.asarray(obs["robo_pos"], dtype=np.float32)
            out["robo_ori"] = np.asarray(obs["robo_ori"], dtype=np.float32)
        else:
            out["robot0_eef_pos"] = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            out["robot0_eef_quat"] = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        return out

    if 'pixels' in obs:
        # Replay-buffer / DummyEnv path → OpenPI-like keys that XVLAPolicy also accepts.
        pixels = np.asarray(obs['pixels'])
        state = np.asarray(obs['state'])
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        if state.shape[-1] == 1:
            state = state[..., 0]
        if pixels.ndim == 3:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(np.ascontiguousarray(pixels), 224, 224))
            wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = np.stack([
                image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(np.ascontiguousarray(pixels[i]), 224, 224))
                for i in range(pixels.shape[0])
            ])
            wrist_img = np.zeros((pixels.shape[0], 224, 224, 3), dtype=np.uint8)
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state.astype(np.float32),
            "prompt": str(variant.task_description),
        }

    raise KeyError(
        f"obs has neither 'agentview_image' nor 'pixels'; keys={list(obs.keys())}"
    )


def obs_to_policy_input(obs, variant, env=None):
    """Dispatch to Pi0 or XVLA obs packing based on ``variant.vla``."""
    if getattr(variant, 'vla', 'openpi') == 'xvla':
        return obs_to_xvla_input(obs, variant, env=env)
    return obs_to_pi_zero_input(obs, variant)


def _maybe_reset_vla_policy(agent_dp):
    if agent_dp is not None and hasattr(agent_dp, 'reset'):
        agent_dp.reset()


def obs_to_pi_zero_input(obs, variant):
    if 'agentview_image' in obs:
        # Raw LIBERO environment observation (from env.step / collect_traj / eval).
        if variant.env == 'libero':
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
            )
            obs_pi_zero = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": str(variant.task_description),
            }
        elif variant.env == 'aloha_cube':
            img = np.ascontiguousarray(obs["pixels"]["top"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )
            obs_pi_zero = {
                "state": obs["agent_pos"],
                "images": {"cam_high": np.transpose(img, (2, 0, 1))}
            }
        else:
            raise NotImplementedError()
    elif 'pixels' in obs:
        # DSRL replay-buffer observation: pixels (B,H,W,C,1) or (H,W,C,1), state (B,8,1) or (8,1).
        if variant.env == 'libero':
            pixels = np.asarray(obs['pixels'])
            state = np.asarray(obs['state'])
            # squeeze trailing num_cameras dim added by DummyEnv
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]   # (..., H, W, C)
            if state.shape[-1] == 1:
                state = state[..., 0]     # (..., 8)
            if pixels.ndim == 3:
                # single observation
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(np.ascontiguousarray(pixels), 224, 224))
                wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # batched observations — resize each frame individually then stack
                img = np.stack([
                    image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(np.ascontiguousarray(pixels[i]), 224, 224))
                    for i in range(pixels.shape[0])
                ])
                wrist_img = np.zeros((pixels.shape[0], 224, 224, 3), dtype=np.uint8)
            obs_pi_zero = {
                "observation/image":       img,
                "observation/wrist_image": wrist_img,
                "observation/state":       state.astype(np.float32),
                "prompt":                  str(variant.task_description),
            }
        else:
            raise NotImplementedError(
                f"DSRL pixel branch not implemented for env={variant.env}")
    else:
        raise KeyError(
            f"obs has neither 'agentview_image' nor 'pixels'; keys={list(obs.keys())}")
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos


def _param_count(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def _print_param_tree(name, params):
    flat = flax.traverse_util.flatten_dict(jax.device_get(params), sep="/")
    print(f"\n{name} parameter tree ({_param_count(params):,} scalars):")
    for path, arr in sorted(flat.items()):
        print(f"  {path}: shape={arr.shape}, dtype={arr.dtype}")


def print_pre_update_summary(variant, agent):
    from jaxrl2.utils.launch_util import print_full_config

    print("=" * 80)
    print("PRE-UPDATE SUMMARY (first online grad step)")
    print("=" * 80)
    print_full_config(variant, agent=agent)

    alpha = float(agent._temp.apply_fn({"params": jax.device_get(agent._temp.params)}))
    print(f"\n[initial temperature alpha] {alpha}")

    for name, state in [
        ("actor", agent._actor),
        ("critic", agent._critic),
        ("temperature", agent._temp),
    ]:
        print(f"\n[{name}] train_state.step = {int(state.step)}")
        _print_param_tree(name, state.params)
        if getattr(state, "batch_stats", None) is not None:
            _print_param_tree(f"{name}/batch_stats", state.batch_stats)

    _print_param_tree("target_critic", agent._target_critic_params)
    print("=" * 80)

def offline_to_online_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    num_offline_steps = getattr(variant, 'num_offline_steps', 0)
    total_env_steps = 0
    i = 0

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        print('performing evaluation for initial checkpoint')
        perform_control_eval(agent, eval_env, 0, variant, wandb_logger, agent_dp)
        # Phase 1: offline
        while i < num_offline_steps:
            print(f'offline update step: {i} started')
            batch_distill = next(replay_buffer_iterator)
            batch_train = next(replay_buffer_iterator)

            update_info = agent.update(batch_distill, batch_train, variant.env, variant.task_description)
            pbar.update()
            print(f'offline update step: {i} done')
            i += 1

            if i % variant.log_interval == 0:
                update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                for k, v in update_info.items():
                    if v.ndim == 0:
                        wandb_logger.log({f'training/{k}': v}, step=i)
                    elif v.ndim <= 2:
                        wandb_logger.log_histogram(f'training/{k}', v, i)

            if i % variant.eval_interval == 0:
                if perform_control_evals:
                    perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                if hasattr(agent, 'perform_eval'):
                    agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

            if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
        if variant.max_steps == num_offline_steps:
            print('offline steps completed')
            return

        # Phase 2: online (optional)
        wandb_logger.log({'num_online_samples': 0}, step=i)
        wandb_logger.log({'num_online_trajs': 0}, step=i)
        wandb_logger.log({'env_steps': 0}, step=i)

        while i <= variant.max_steps:
            if getattr(variant, 'chunk_reward', 0):
                raw_traj = collect_traj_chunked(variant, agent, env, i, agent_dp)
                traj = _build_chunked_insert_traj(raw_traj, variant.query_freq)
            else:
                traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)

            if getattr(variant, 'num_online_gradsteps_batch', -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = traj.get('num_transitions', len(traj['rewards'])) * variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    batch_distill = next(replay_buffer_iterator)
                    batch_train = next(replay_buffer_iterator)
                    update_info = agent.update(batch_distill, batch_train, variant.env, variant.task_description)
                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    printed_pre_update_summary = False

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        print('performing evaluation for initial checkpoint')
        if perform_control_evals:
            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
        # Skip agent.perform_eval here: replay buffer is empty, so value/reward
        # visualizations would fail (get_random_trajs with _traj_counter == 0).

        while i <= variant.max_steps:
            if getattr(variant, 'overlap_transitions', 0):
                raw_traj = collect_traj_chunked(
                    variant, agent, env, i, agent_dp, store_noise=True
                )
                traj = _build_chunked_insert_traj(
                    raw_traj, variant.query_freq, stack_actions=False
                )
            else:
                traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            # if variant.get("num_online_gradsteps_batch", -1) > 0:
            #     num_gradsteps = variant.num_online_gradsteps_batch
            # else:
            #     num_gradsteps = traj.get('num_transitions', len(traj['rewards'])) * variant.multi_grad_step

            if getattr(variant, 'chunk_reward', 1):
                num_gradsteps = traj.get('num_transitions', len(traj['rewards'])) * variant.multi_grad_step
            else:
                num_gradsteps = len(traj['rewards'])*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                if not printed_pre_update_summary:
                    # print_pre_update_summary(variant, agent)
                    printed_pre_update_summary = True
                for _ in range(num_gradsteps):
                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    terminations = traj.get('terminations')
    if terminations is not None:
        terminations = np.array(terminations)

    # Support explicit next_observations: used by _build_chunked_insert_traj where
    # next_obs is Q env steps ahead rather than one query step ahead.
    if 'next_observations' in traj:
        next_obs_seq = traj['next_observations']
    else:
        next_obs_seq = traj['observations'][1:]

    # Support explicit next_actions: used by _build_chunked_insert_traj where
    # next_actions is the Q-step-ahead chunk rather than the immediately following query.
    if 'next_actions' in traj:
        next_actions_arr = np.array(traj['next_actions'])
    else:
        next_actions_arr = None

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = next_obs_seq[t]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        if next_actions_arr is not None:
            na = next_actions_arr[t]
        else:
            na = actions[t + 1] if t < episode_len - 1 else actions[t]

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=na,
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        if terminations is not None:
            insert_dict['terminations'] = terminations[t]
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    chunk_reward = bool(variant.get('chunk_reward', 0))

    agent._rng, rng = jax.random.split(agent._rng)
    _maybe_reset_vla_policy(agent_dp)
    
    if 'libero' in variant.env:
        obs = env.reset()
        if getattr(variant, 'vla', 'openpi') == 'xvla':
            obs = prepare_libero_episode_for_xvla(env)
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    elif 'metaworld' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    env_rewards = []
    action_list = []
    obs_list = []
    chunk_rewards = []
    chunk_terminations = []
    current_chunk_rewards = []
    current_chunk_terminations = []

    for t in range(max_timesteps):
        # jax.debug.print('obs: {obs}', obs=obs.keys())
        curr_image = obs_to_img(obs, variant)
        
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_policy = obs_to_policy_input(obs, variant, env=env)
            if i == 0:
                noise = jax.random.normal(rng, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                if noise.shape[1] < variant.pi0_action_horizon:
                    noise_repeat = jax.numpy.repeat(
                        noise[:, -1:, :], variant.pi0_action_horizon - noise.shape[1], axis=1
                    )
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                if agent.only_predict_dims_until > 0:
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :variant.only_predict_dims_until]
                else:
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                actions_noise = agent.sample_actions(obs_dict, marginalize_logprobs=variant.marginalize_logprobs,
                                                     use_actor_diff=getattr(variant, 'use_actor_diff', False))
                if agent.only_predict_dims_until > 0:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    actions_noise_complete = noise[0, :len(actions_noise), :]
                    print(f'actions_noise: {actions_noise_complete.shape}')
                else:
                    actions_noise_complete = actions_noise
                noise = _prepare_pi0_noise(actions_noise_complete, agent, variant.pi0_action_horizon)
                print(f'noise: {noise.shape}')
            
            infer_kwargs = {}
            if getattr(variant, 'vla', 'openpi') == 'xvla':
                infer_kwargs['proprio_from_step'] = query_frequency - 1
            actions = agent_dp.infer(obs_policy, noise=noise, **infer_kwargs)["actions"]
            if not agent.only_predict_dims_until > 0:
                action_list.append(np.reshape(actions_noise, agent.action_chunk_shape))
            else:
                action_list.append(actions_noise)
            obs_list.append(obs_dict)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        if chunk_reward:
            if done:
                reward = 0.0
            else:
                reward = -1.0
            current_chunk_rewards.append(reward)
            current_chunk_terminations.append(bool(done))
            
        env_rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

        if chunk_reward and (t + 1) % query_frequency == 0:
            chunk_rewards.append(np.array(current_chunk_rewards, dtype=np.float32))
            chunk_terminations.append(np.array(current_chunk_terminations, dtype=np.bool_))
            current_chunk_rewards = []
            current_chunk_terminations = []

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    env_rewards = np.array(env_rewards)
    episode_return = np.sum(env_rewards[env_rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')

    if chunk_reward:
        if len(current_chunk_rewards) > 0:
            chunk_rewards.append(np.array(current_chunk_rewards, dtype=np.float32))
            chunk_terminations.append(np.array(current_chunk_terminations, dtype=np.bool_))
        rewards, terminations, masks = _finalize_chunk_rewards(
            chunk_rewards, chunk_terminations, query_frequency
        )
        traj = {
            'observations': obs_list,
            'actions': action_list,
            'rewards': rewards,
            'terminations': terminations,
            'masks': masks,
            'is_success': is_success,
            'episode_return': episode_return,
            'images': image_list,
            'env_steps': t + 1,
        }
        return traj
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1 
    }

def collect_traj_chunked(variant, agent, env, i, agent_dp=None, store_noise=False):
    """Collect a rollout storing per-env-step actions for chunked / overlapping replay.

    Unlike collect_traj, which stores one SAC noise action per Pi0 query, this
    function records an action at every environment step along with every obs.
    The Pi0/noise sampling logic at query boundaries is preserved exactly.
    The returned raw trajectory is consumed by _build_chunked_insert_traj.

    Args:
        store_noise: If False (default), store the physical Pi0 action sent to
            env.step at each step (shape (action_dim,)). If True, store the
            current query's SAC noise reshaped to agent.action_chunk_shape at
            every step (same noise broadcast across the replan interval).
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)
    _maybe_reset_vla_policy(agent_dp)

    if 'libero' in variant.env:
        obs = env.reset()
        if getattr(variant, 'vla', 'openpi') == 'xvla':
            obs = prepare_libero_episode_for_xvla(env)
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    elif 'metaworld' in variant.env:
        obs, _ = env.reset()

    image_list = []
    env_rewards = []
    all_obs = []          # one obs_dict per step + terminal obs, length T+1
    all_actions = []      # per-step action (physical or noise), length T
    step_rewards = []     # sparse scalar per step (-1 / 0), length T
    step_terminations = []  # bool per step, length T
    current_noise = None

    for t in range(max_timesteps):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        # Store obs before stepping (same batch-dim convention as collect_traj)
        all_obs.append(obs_dict)

        if t % query_frequency == 0:
            assert agent_dp is not None
            rng, key = jax.random.split(rng)
            obs_policy = obs_to_policy_input(obs, variant, env=env)
            if i == 0:
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                if noise.shape[1] < variant.pi0_action_horizon:
                    noise_repeat = jax.numpy.repeat(
                        noise[:, -1:, :], variant.pi0_action_horizon - noise.shape[1], axis=1
                    )
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                actions_noise = agent.sample_actions(
                    obs_dict,
                    marginalize_logprobs=variant.marginalize_logprobs,
                    use_actor_diff=getattr(variant, 'use_actor_diff', False),
                )
                noise = _prepare_pi0_noise(actions_noise, agent, variant.pi0_action_horizon)

            infer_kwargs = {}
            if getattr(variant, 'vla', 'openpi') == 'xvla':
                infer_kwargs['proprio_from_step'] = query_frequency - 1
            actions = agent_dp.infer(obs_policy, noise=noise, **infer_kwargs)["actions"]
            if store_noise:
                current_noise = np.reshape(
                    np.asarray(actions_noise, dtype=np.float32), agent.action_chunk_shape
                )

        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
        elif 'metaworld' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        if store_noise:
            all_actions.append(current_noise)
        else:
            all_actions.append(np.asarray(action_t, dtype=np.float32))
        # Sparse reward: 0 at the terminal/success step, -1 for every live step
        step_rewards.append(0.0 if done else -1.0)
        step_terminations.append(bool(done))
        image_list.append(curr_image)
        if done:
            reward = 0.0
            break
        else:  
            reward = -1.0

    # Append terminal observation so all_obs has length T+1
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    if variant.add_states:
        terminal_obs = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
    else:
        terminal_obs = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
        }
    all_obs.append(terminal_obs)
    image_list.append(curr_image)

    episode_return = float(np.sum(step_rewards))
    is_success = (reward == env_max_reward) # env_max_reward is 0 for chunked reward
    print(f'Rollout Done (chunked): {episode_return=}, Success: {is_success}')

    return {
        'observations': all_obs,            # T+1 obs dicts with batch dim
        'all_actions': all_actions,         # T actions (physical or noise)
        'step_rewards': step_rewards,       # T floats
        'step_terminations': step_terminations,  # T bools
        'episode_return': episode_return,
        'is_success': is_success,
        'images': image_list,
        'env_steps': t + 1,
    }


def _build_chunked_insert_traj(raw_traj, Q, stack_actions=True):
    """Convert per-step collect_traj_chunked output to overlapping chunk transitions.

    Produces overlapping transitions for a rollout of T executed env steps.
    Incomplete episode tails (< Q remaining steps) are omitted — no padding.

    Windows that lack a full next-action span (need 2Q steps ahead when
    stack_actions, or step t+Q when not) are skipped, except the single last
    complete window t = T - Q, which is kept once with next_actions falling
    back to the current action(s).

    Each kept transition t covers env steps [t, t+Q):
      observations      obs at step t
      next_observations obs at step t+Q
      actions           if stack_actions: physical actions at steps t..t+Q-1,
                        shape (Q, action_dim); else: the per-step action at t
                        (e.g. full SAC noise chunk), unchanged shape
      next_actions      if stack_actions: physical actions at steps t+Q..t+2Q-1
                        (or current chunk for the single last window);
                        else: action at t+Q if available else action at t
      rewards           sparse reward per step in [t, t+Q), shape (Q,)
      terminations      done flag per step in [t, t+Q), shape (Q,)
      masks             0.0 if any step in the window terminated, else 1.0
    """
    all_obs = raw_traj['observations']         # T+1 entries
    all_actions = raw_traj['all_actions']      # T entries
    step_rewards = raw_traj['step_rewards']
    step_terminations = raw_traj['step_terminations']
    T = len(all_actions)
    N = max(T - Q + 1, 0)

    observations = []
    next_observations = []
    actions = []
    next_actions_list = []
    rewards = []
    terminations = []
    masks = []

    for t in range(N):
        if stack_actions:
            chunk_actions = np.stack(all_actions[t: t + Q])  # (Q, action_dim)
            if t + 2 * Q <= T:
                chunk_next_actions = np.stack(all_actions[t + Q: t + 2 * Q])
        else:
            chunk_actions = np.asarray(all_actions[t])
            if t + Q < T:
                chunk_next_actions = np.asarray(all_actions[t + Q])

        chunk_rewards = np.asarray(step_rewards[t: t + Q], dtype=np.float32)
        chunk_terminations = np.asarray(step_terminations[t: t + Q], dtype=np.bool_)

        observations.append(all_obs[t])
        next_observations.append(all_obs[t + Q])
        actions.append(chunk_actions)
        next_actions_list.append(chunk_next_actions)
        rewards.append(chunk_rewards)
        terminations.append(chunk_terminations)
        masks.append(0.0 if chunk_terminations.any() else 1.0)

    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'next_actions': next_actions_list,
        'rewards': rewards,
        'terminations': terminations,
        'masks': masks,
        'num_transitions': N,
        'episode_return': raw_traj['episode_return'],
        'is_success': raw_traj['is_success'],
        'env_steps': raw_traj['env_steps'],
    }


def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)
    video_log_interval = 10
    log_video = True

    for rollout_id in range(variant.eval_episodes):
        _maybe_reset_vla_policy(agent_dp)
        if 'libero' in variant.env:
            obs = env.reset()
            # init_states = variant.libero_init_states
            # obs = env.set_init_state(init_states[rollout_id % len(init_states)])
            if getattr(variant, 'vla', 'openpi') == 'xvla':
                obs = prepare_libero_episode_for_xvla(env)
            else:
                for _ in range(LIBERO_NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in range(max_timesteps):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_policy = obs_to_policy_input(obs, variant, env=env)
                
                
                if i == 0:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, variant.pi0_action_horizon, variant.dsrl_action_dim))
                else:
                    actions_noise = agent.sample_actions(obs_dict, marginalize_logprobs=variant.marginalize_logprobs,
                                                         use_actor_diff=getattr(variant, 'use_actor_diff', False))
                    if agent.only_predict_dims_until > 0:
                        noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                        actions_noise = noise[0, :len(actions_noise), :]

                    noise = _prepare_pi0_noise(actions_noise, agent, variant.pi0_action_horizon)

                infer_kwargs = {}
                if getattr(variant, 'vla', 'openpi') == 'xvla':
                    infer_kwargs['proprio_from_step'] = query_frequency - 1
                actions = agent_dp.infer(obs_policy, noise=noise, **infer_kwargs)["actions"]
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            if variant.chunk_reward:
                if done:
                    reward = 0.0
                else:
                    reward = -1.0
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        if log_video and t % video_log_interval == 0:
            wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50, format="gif")}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
