from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min', chunk_reward: bool = False) -> Tuple[TrainState, Dict[str, float]]:
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
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
