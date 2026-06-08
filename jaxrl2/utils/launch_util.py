import inspect
from pprint import pprint

from jaxrl2.utils.general_utils import AttrDict


def _pixel_sac_init_defaults():
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

    sig = inspect.signature(PixelSACLearner.__init__)
    skip = {"self", "seed", "observations", "actions"}
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if k not in skip and v.default is not inspect.Parameter.empty
    }


def get_full_config_dict(variant, agent=None, extra=None):
    """Full hyperparameter dict: run config, explicit train_kwargs, and effective SAC values."""
    extra = extra or {}
    run_cfg = {k: v for k, v in dict(variant).items() if k != "train_kwargs"}
    train_kwargs = dict(variant.train_kwargs)
    effective_sac = {**_pixel_sac_init_defaults(), **train_kwargs}
    config = {
        "run": run_cfg,
        "train_kwargs": train_kwargs,
        "effective_sac": effective_sac,
    }
    if agent is not None:
        config["agent_runtime"] = {
            "tau": agent.tau,
            "discount": agent.discount,
            "critic_reduction": agent.critic_reduction,
            "target_entropy": agent.target_entropy,
            "action_dim": agent.action_dim,
            "action_chunk_shape": tuple(agent.action_chunk_shape),
            "aug_next": agent.aug_next,
            "color_jitter": agent.color_jitter,
            "num_cameras": agent.num_cameras,
            "chunk_reward": agent.chunk_reward,
            "use_chunky_actor_critic": agent.use_chunky_actor_critic,
            "pi0_action_horizon": agent.pi0_action_horizon,
            "action_horizon": agent.action_horizon,
        }
    if extra:
        config["extra"] = extra
    return config


def print_full_config(variant, agent=None, extra=None):
    """Print all hyperparameters, including argparse and PixelSACLearner defaults."""
    print("=" * 80)
    print("FULL CONFIG (CLI defaults + runtime overrides + effective SAC kwargs)")
    print("=" * 80)
    config = get_full_config_dict(variant, agent=agent, extra=extra)
    for section, values in config.items():
        print(f"\n[{section}]")
        pprint(values, sort_dicts=True)
    print("=" * 80)


def parse_training_args(train_args_dict, parser):
    for k, v in train_args_dict.items():
        if type(v) == tuple:
            parser.add_argument('--' + k, nargs="+", default=v, type=type(v[0]))
        elif type(v) != bool:
            parser.add_argument('--' + k, default=v, type=type(v))
        else:
            parser.add_argument('--' + k, default=int(v), type=int)
    args = parser.parse_args()
    config = {}
    for key in train_args_dict.keys():
        config[key] = getattr(args, key)
    variant = AttrDict(vars(args))
    variant['train_kwargs'] = config
    return variant, args
