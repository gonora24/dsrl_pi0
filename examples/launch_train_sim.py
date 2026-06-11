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
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--chunk_reward', default=0, help='sum discounted per-step env rewards within each action chunk for critic bootstrap (RLinf-style)', type=int)
    parser.add_argument('--use_chunky_actor_critic', default=0, help='use full (pi0_action_horizon x 32) noise for actor+critic; requires query_freq == pi0 horizon. If off, actor outputs 32-d noise repeated to pi0 length.', type=int)
    parser.add_argument('--use_transformer_critic', default=0, help='use transformer critic', type=int)
    parser.add_argument('--transformer_n_embd', default=256, help='transformer embedding dimension', type=int)
    parser.add_argument('--transformer_n_head', default=4, help='transformer number of heads', type=int)
    parser.add_argument('--transformer_n_layer', default=4, help='transformer number of layers', type=int)
    parser.add_argument('--transformer_use_layer_norm', default=1, help='transformer use layer norm', type=int)
    parser.add_argument('--transformer_use_bias', default=0, help='transformer use bias', type=int)
    parser.add_argument('--use_transformer_actor', default=0, help='use transformer actor', type=int)
    parser.add_argument('--actor_transformer_d_model', default=128, help='actor transformer embedding dimension', type=int)
    parser.add_argument('--actor_transformer_n_layers', default=3, help='actor transformer number of layers', type=int)
    parser.add_argument('--actor_transformer_n_heads', default=4, help='actor transformer number of heads', type=int)
    parser.add_argument('--actor_transformer_dropout', default=0.1, help='actor transformer dropout', type=float)
    parser.add_argument('--libero_suite', default="libero_90", help='libero task suite', type=str)
    parser.add_argument('--libero_task_id', default=57, help='which libero task in suite', type=int)
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
    
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
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
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    