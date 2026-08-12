from audioop import cross
from typing import Dict, Tuple

import jax
import optax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def _maybe_freeze_residual(grads, step, freeze_steps):
    """Zero out residual_out gradients while step < freeze_steps."""
    def mask_leaf(path, g):
        path_str = '/'.join(
            k.key if hasattr(k, 'key') else str(k) for k in path
        )
        if 'residual_out' in path_str:
            return jnp.where(step < freeze_steps, jnp.zeros_like(g), g)
        return g
    return jax.tree_util.tree_map_with_path(mask_leaf, grads)


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min', marginalize_logprobs:bool=False, use_actor_diff:bool=False, use_actor_diff_mean:bool=False, freeze_residual_steps:int=0) -> Tuple[TrainState, Dict[str, float]]:
    
    key, key_act, key_dropout = jax.random.split(key, num=3)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, means, log_stds, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
            if cross_norm:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist, means, log_stds = actor.apply_fn({'params': actor_params}, batch['observations'], training=True, rngs={'dropout': key_dropout})
            # next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)
        
        if marginalize_logprobs:
            actions, log_probs = dist.compute_marginalized_logprobs(means, log_stds, key=key_act)
        elif use_actor_diff:
            actions, log_probs = dist.sample_and_log_prob_diff(seed=key_act)
        elif use_actor_diff_mean:
            actions, log_probs, residual_mean, residual_std = dist.sample_and_log_prob_diff_mean(seed=key_act)
        else:
            actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, batch['observations'],
                            actions, mutable=['batch_stats'])
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        if marginalize_logprobs:
            discount = batch['discount'][0] ** jnp.arange(dist.action_horizon)
            assert log_probs.shape == (actions.shape[0], dist.action_horizon)
            cum_log_prob = log_probs * discount
            nonfinite_action_logprobs = 1 - jnp.mean(jnp.isfinite(log_probs))
            cum_log_prob = jnp.nan_to_num(
                        cum_log_prob, nan=0, posinf=0, neginf=0
                    )
            log_probs = cum_log_prob.sum(axis=1).reshape(actions.shape[0], 1)
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
        }
        if use_actor_diff:
            residual = [actions[:, i] - actions[:, i-1] for i in range(1, dist.action_horizon)]
            residual = jnp.stack(residual, axis=1)
            things_to_log['residual_mean'] = residual.mean()
            things_to_log['residual_std'] = residual.std(axis=1).mean()
            things_to_log['residual_min'] = residual.min(axis=1).mean()
            things_to_log['residual_max'] = residual.max(axis=1).mean()
        if use_actor_diff_mean:
            things_to_log['residual_mean_mean'] = residual_mean.mean()
            things_to_log['residual_std_mean'] = residual_std.mean()
            things_to_log['residual_mean_min'] = residual_mean.min(axis=1).mean()
            things_to_log['residual_mean_max'] = residual_mean.max(axis=1).mean()
            things_to_log['residual_std_min'] = residual_std.min(axis=1).mean()
            things_to_log['residual_std_max'] = residual_std.max(axis=1).mean()
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    if freeze_residual_steps > 0:
        grads = _maybe_freeze_residual(grads, actor.step, freeze_residual_steps)
    info = {**info, 'actor_grad_norm': optax.global_norm(grads)}
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info