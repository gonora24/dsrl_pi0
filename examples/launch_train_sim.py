import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--restore_path', default=None, help='Path to checkpoint to restore from.', type=str)
    parser.add_argument('--initialize_weights_from', default=None, help='Baseline checkpoint path for warm-starting multi-vector actor.', type=str)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--online_buffer_size', default=-1, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--chunk_reward', default=0, help='sum discounted per-step env rewards within each action chunk for critic bootstrap (RLinf-style)', type=int)
    parser.add_argument(
        '--overlap_transitions', default=0, type=int,
        help='When set with chunk_reward: collect per-env-step obs and store SAC noise '
             'at every step, then insert overlapping Q-step reward windows (stride 1). '
             'Requires --chunk_reward 1.',
    )
    parser.add_argument('--use_chunky_actor_critic', default=0, help='use full (pi0_action_horizon x 32) noise for actor+critic; requires query_freq == pi0 horizon. If off, actor outputs 32-d noise repeated to pi0 length.', type=int)
    parser.add_argument('--num_noise_vectors', default=1, type=int,
        help='Number of independent noise vectors the actor predicts (N). '
             'Total SAC action dim = N * dsrl_action_dim. '
             'When N > 1, overrides use_chunky_actor_critic.')
    parser.add_argument('--noise_repeats_per_vector', default=1, type=int,
        help='Times each of the N vectors is repeated before being fed to the VLA (K). '
             'Should satisfy N*K == pi0_action_horizon for an exact fit; '
             'otherwise the last vector is padded or the sequence is truncated.')
    parser.add_argument('--interpolate_noise_vectors', default=0, type=int,
        help='If 1, place the N noise vectors as evenly-spaced anchors across the '
             'horizon and fill intermediate steps with piecewise linear interpolation '
             'instead of hard-tiling each vector K times.')
    parser.add_argument('--use_frozen_baseline_residual', default=0, type=int,
        help='If 1, use a frozen warm-started 32-d MLP as anchor and a trainable residual '
             'MLP that adds corrections for residual_n_vectors extra copies.')
    parser.add_argument('--residual_n_vectors', default=1, type=int,
        help='Number of residual copies K. Total action dim = (1+K)*32. '
             'The frozen vector is tiled K times and the residual is added on top.')
    parser.add_argument('--residual_hidden_dims', default=None, nargs='+', type=int,
        help='Hidden dims for the residual MLP. Defaults to hidden_dims if not set.')
    parser.add_argument('--use_transformer_critic', default=0, help='use transformer critic', type=int)
    parser.add_argument('--transformer_n_embd', default=256, help='transformer embedding dimension', type=int)
    parser.add_argument('--transformer_n_head', default=4, help='transformer number of heads', type=int)
    parser.add_argument('--transformer_n_layer', default=4, help='transformer number of layers', type=int)
    parser.add_argument('--transformer_weight_norm', default=1, help='transformer use weight norm', type=int)
    parser.add_argument('--transformer_use_bias', default=0, help='transformer use bias', type=int)
    parser.add_argument('--use_transformer_actor', default=0, help='use transformer actor', type=int)
    parser.add_argument('--actor_transformer_d_model', default=128, help='actor transformer embedding dimension', type=int)
    parser.add_argument('--actor_transformer_n_layers', default=3, help='actor transformer number of layers', type=int)
    parser.add_argument('--actor_transformer_n_heads', default=4, help='actor transformer number of heads', type=int)
    parser.add_argument('--actor_transformer_dropout', default=0.1, help='actor transformer dropout', type=float)
    parser.add_argument('--clip_actor_grad_norm', default=0.0, help='clip actor gradient norm', type=float)
    parser.add_argument('--clip_critic_grad_norm', default=0.0, help='clip critic gradient norm', type=float)
    parser.add_argument('--libero_suite', default="libero_90", help='libero task suite', type=str)
    parser.add_argument('--libero_task_id', default=57, help='which libero task in suite', type=int)
    parser.add_argument('--metaworld_task_name', default='basketball-v3', help='which metaworld task', type=str)
    parser.add_argument('--use_chunk_actor_transformer', default=0, help='use chunk actor transformer', type=int)
    parser.add_argument('--marginalize_logprobs', default=0, help='marginalize logprobs', type=int)
    parser.add_argument('--use_actor_diff', default=0, help='use autoregressive difference predictor', type=int)
    parser.add_argument('--freeze_residual_steps', default=0, type=int,
                        help='freeze residual head gradients for first N steps (requires use_actor_diff)')
    parser.add_argument('--num_qs', default=2, help='number of Q-heads', type=int)
    parser.add_argument('--critic_hidden_dims', default=[128, 128, 128], help='critic hidden dimensions', nargs="+", type=int)
    parser.add_argument('--hidden_dims', default=[128, 128, 128], help='actor hidden dimensions', nargs="+", type=int)
    parser.add_argument('--trajectory_hdf5_path', default=None, help='path to trajectory hdf5 file', type=str)
    parser.add_argument('--num_offline_steps', default=0, help='Gradient steps on offline data before online collection starts', type=int)
    parser.add_argument('--backup_entropy', default=0, help='backup entropy', type=int)
    parser.add_argument(
        '--pi0_checkpoint',
        default='openpi',
        type=str,
        help=(
            "Pi0 weights for LIBERO: 'openpi' (Orbax pi0_libero), 'pi05_libero' (Orbax gs://.../pi05_libero), "
            "'rlinf_hf_long', 'rlinf_hf_goalSpatial', 'rlinf_hf_pi05' (HF PyTorch safetensors), "
            "or a local directory containing either Orbax 'params/' or 'model.safetensors'."
        ),
    )
    parser.add_argument(
        '--pi0_microbatch_size', default=0, type=int,
        help='Max batch size per Pi0 inference call during DSRL-NA updates. '
             '0 = use full SAC batch_size (current behavior).',
    )
    parser.add_argument('--vla', default='openpi', help='vla type', type=str)
    parser.add_argument('--only_predict_dims_until', default=-1, help='only predict dimensions until this dimension', type=int)
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    