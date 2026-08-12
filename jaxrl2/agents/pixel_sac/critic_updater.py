from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from examples.train_utils_sim import obs_to_policy_input
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

# Cached once at first use so every microbatch call avoids re-querying JAX.
_PI0_DEVICE = None


def _pi0_device():
    global _PI0_DEVICE
    if _PI0_DEVICE is None:
        gpus = jax.devices('gpu')
        _PI0_DEVICE = gpus[0] if gpus else jax.local_devices()[0]
    return _PI0_DEVICE


def _put_on_device(x, device):
    if isinstance(x, (str, bytes)):
        return x
    arr = np.asarray(x)
    if arr.dtype.kind in ('U', 'S', 'O'):
        return x
    return jax.device_put(jnp.asarray(arr), device)


def _slice_policy_input(obs_policy: dict, indx: np.ndarray) -> dict:
    """Slice a preprocessed policy input dict along the batch axis.

    String values (e.g. the language prompt) are kept as-is; all other values
    are indexed with *indx*.
    """
    return {
        k: v if isinstance(v, (str, bytes)) else v[indx]
        for k, v in obs_policy.items()
    }


def _pad_policy_input(obs_policy: dict, pad: int) -> dict:
    """Append *pad* zero rows to every array in a preprocessed policy input dict."""
    result = {}
    for k, v in obs_policy.items():
        if isinstance(v, (str, bytes)):
            result[k] = v
        else:
            result[k] = np.concatenate(
                [v, np.zeros((pad, *v.shape[1:]), dtype=v.dtype)], axis=0
            )
    return result


def _infer_pi0_actions_single(agent_dp: Any, obs_policy: dict, noise, variant) -> np.ndarray:
    """Run a single policy inference call with inputs already preprocessed.

    Expects *obs_policy* to be the output of ``obs_to_policy_input``; this
    function only handles device placement and the actual ``agent_dp.infer``
    call so that preprocessing is not repeated for every microbatch chunk.
    """
    device = _pi0_device()
    # XVLAPolicy expects numpy host tensors; OpenPI JAX Policy wants device arrays.
    if getattr(variant, 'vla', 'openpi') == 'xvla':
        return np.asarray(agent_dp.infer(obs_policy, noise=noise)['actions'], dtype=np.float32)

    obs_policy = jax.tree.map(
        lambda x: _put_on_device(x, device),
        obs_policy,
    )
    if noise is not None:
        noise = jax.device_put(jnp.asarray(noise), device)
    return np.asarray(agent_dp.infer(obs_policy, noise=noise)['actions'], dtype=np.float32)


def _infer_pi0_actions(
        agent_dp: Any, obs, noise, variant, microbatch_size: int = 0,
) -> np.ndarray:
    """Run pi0 inference, optionally splitting the batch into smaller chunks.

    Observations are preprocessed (image resize etc.) once for the full batch
    before the microbatch loop so that expensive per-image CPU work is not
    repeated per chunk.  The last chunk is zero-padded to *microbatch_size* so
    that JAX always sees a fixed compiled shape and avoids recompilation.
    """
    obs_policy = obs_to_policy_input(obs, variant)
    batch_size = int(noise.shape[0])
    if microbatch_size <= 0 or microbatch_size >= batch_size:
        return _infer_pi0_actions_single(agent_dp, obs_policy, noise, variant)

    chunks = []
    for start in range(0, batch_size, microbatch_size):
        end = min(start + microbatch_size, batch_size)
        chunk_len = end - start
        indx = np.arange(start, end)
        obs_chunk = _slice_policy_input(obs_policy, indx)
        noise_chunk = noise[start:end]

        # Pad the last (potentially shorter) chunk to microbatch_size so JAX
        # always compiles for a single fixed shape.
        if chunk_len < microbatch_size:
            pad = microbatch_size - chunk_len
            obs_chunk = _pad_policy_input(obs_chunk, pad)
            noise_chunk = np.concatenate(
                [noise_chunk, np.zeros((pad, *noise_chunk.shape[1:]), dtype=noise_chunk.dtype)],
                axis=0,
            )

        result = _infer_pi0_actions_single(agent_dp, obs_chunk, noise_chunk, variant)

        # Discard padded rows from the output.
        if chunk_len < microbatch_size:
            result = result[:chunk_len]

        chunks.append(result)
    return np.concatenate(chunks, axis=0)


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min', chunk_reward: bool = False,
        marginalize_logprobs: bool = False, use_actor_diff: bool = False, use_actor_diff_mean: bool = False) -> Tuple[TrainState, Dict[str, float]]:
    dist, means, log_stds = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    if marginalize_logprobs:
        next_actions, next_log_probs = dist.compute_marginalized_logprobs(means, log_stds, key=key)
    elif use_actor_diff:
        next_actions, next_log_probs = dist.sample_and_log_prob_diff(seed=key)
    elif use_actor_diff_mean:
        next_actions, next_log_probs, _, _ = dist.sample_and_log_prob_diff_mean(seed=key)
    else:
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)

    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    rewards = batch['rewards']
    if chunk_reward:
        chunk_size = rewards.shape[-1]
        exponents = jnp.arange(chunk_size, dtype=jnp.float32)
        gamma_powers = discount ** exponents
        rewards_for_bootstrap = jnp.sum(rewards * gamma_powers, axis=-1)
        bootstrap_discount = discount ** chunk_size
        # bootstrap_mask = jnp.logical_not(jnp.any(batch['terminations'], axis=-1))
        bootstrap_mask = batch['masks']
        # jax.debug.print('bootstrap_mask shape: {bootstrap_mask}', bootstrap_mask=bootstrap_mask.shape)
        # jax.debug.print('bootstrap_mask: {bootstrap_mask}', bootstrap_mask=bootstrap_mask[0])
        # jax.debug.print('terminations: {terminations}', terminations=batch['terminations'][0])
        # jax.debug.print('terminations shape: {terminations}', terminations=batch['terminations'].shape)
    else:
        rewards_for_bootstrap = rewards
        bootstrap_discount = batch['discount']
        bootstrap_mask = batch['masks']

    target_q = (
        rewards_for_bootstrap
        + bootstrap_discount * bootstrap_mask * next_q
    )
    # jax.debug.print('target_q: {target_q}', target_q=target_q)
    # jax.debug.print('rewards_for_bootstrap: {rewards_for_bootstrap}', rewards_for_bootstrap=rewards_for_bootstrap)

    if backup_entropy:
        target_q -= bootstrap_discount * bootstrap_mask * temp.apply_fn(
            {'params': temp.params}) * next_log_probs

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_std': jnp.std(qs),
            'q_min': jnp.min(qs),
            'q_max': jnp.max(qs),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    info = {**info, 'critic_grad_norm': optax.global_norm(grads)}
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info

def update_na_critic(
        critic: TrainState, target_critic: TrainState, temp: TrainState,
        batch: DatasetDict, pi0_next_actions: jnp.ndarray,
        next_log_probs: jnp.ndarray, discount: float,
        backup_entropy: bool = False,
        critic_reduction: str = 'min',
) -> Tuple[TrainState, Dict[str, float]]:
    """Update the NA critic from detached Pi0 actions computed outside JIT."""
    next_actions = jax.lax.stop_gradient(pi0_next_actions)
    next_log_probs = jax.lax.stop_gradient(next_log_probs)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)

    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    rewards = batch['rewards']

    chunk_size = rewards.shape[-1]
    exponents = jnp.arange(chunk_size, dtype=jnp.float32)
    gamma_powers = discount ** exponents
    rewards_for_bootstrap = jnp.sum(rewards * gamma_powers, axis=-1)
    bootstrap_discount = discount ** chunk_size
    bootstrap_mask = batch['masks']

    if backup_entropy:
        next_q -= temp.apply_fn({'params': temp.params}) * next_log_probs

    target_q = (
        rewards_for_bootstrap
        + bootstrap_discount * bootstrap_mask * next_q
    )

    # if backup_entropy:
    #     target_q -= bootstrap_discount * bootstrap_mask * temp.apply_fn(
    #         {'params': temp.params}) * next_log_probs

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = 0.5*((qs - target_q)**2).mean()
        return critic_loss, {
            'na_critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_std': jnp.std(qs),
            'q_min': jnp.min(qs),
            'q_max': jnp.max(qs),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    info = {**info, 'na_critic_grad_norm': optax.global_norm(grads)}
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info

def update_noise_critic(
    critic: TrainState,
    noise_critic: TrainState,
    batch: DatasetDict,
    noise_actions: jnp.ndarray,
    pi0_diffused_actions: jnp.ndarray,
    only_predict_dims_until: int = 0,
) -> Tuple[TrainState, Dict[str, float]]:
    """Distil the NA critic into the noise critic using detached Pi0 actions."""
    diffused_actions = jax.lax.stop_gradient(pi0_diffused_actions)
    noise_actions = jax.lax.stop_gradient(noise_actions)
    teacher_params = jax.lax.stop_gradient(critic.params)

    def distill_loss_fn(
        noise_critic_params: Params,
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn(
            {'params': teacher_params}, batch['observations'], diffused_actions
        )
        # if not use_chunky_actor_critic:
        #     if critic_reduction == 'min':
        #         qs = qs.min(axis=0)
        #     elif critic_reduction == 'mean':
        #         qs = qs.mean(axis=0)
        if only_predict_dims_until > 0:
            _noise_actions = noise_actions[:, :, :only_predict_dims_until]
        else:
            _noise_actions = noise_actions
        noise_qs = noise_critic.apply_fn(
            {'params': noise_critic_params}, batch['observations'], _noise_actions
        )
        distill_loss = 0.5*((qs - noise_qs) ** 2).mean()
        return distill_loss, {'distill_loss': distill_loss}

    grads, info = jax.grad(distill_loss_fn, has_aux=True)(noise_critic.params)
    info = {**info, 'noise_critic_grad_norm': optax.global_norm(grads)}
    new_noise_critic = noise_critic.apply_gradients(grads=grads)

    return new_noise_critic, info