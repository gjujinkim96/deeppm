import argparse
import yaml
from types import SimpleNamespace

def get_configs(args, show=False):
    cfg = read_from_file(args.cfg, 'cfg', show=show)
    return cfg

def get_args(show=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, help='Name of the experiment to run')
    parser.add_argument('--cfg', required=True, help='Configuration')

    parser.add_argument('--small_size', required=False, action='store_true', help='For quick test')
    parser.add_argument('--small_training', required=False, action='store_true', help='For quick validation test')
    parser.add_argument('--wandb_disabled', required=False, action='store_true', help='For turning wandb logging off')
    parser.add_argument('--exp_override', required=False, action='store_true', help='For overriding run')
    args = parser.parse_args()

    if show:
        print('###################')
        print(f'Args')
        print('###################')
        print(yaml.dump(vars(args), default_flow_style=False))
        print()

    return args


def read_from_file(file, config_name, show=False):
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if show:
        print('###################')
        print(f'{config_name}')
        print('###################')
        print(yaml.dump(config, default_flow_style=False))
        print()

    return dict_to_simple_namespace(config)

def dict_to_simple_namespace(obj):
    new_dict = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            v = dict_to_simple_namespace(v)
        new_dict[k] = v
    return SimpleNamespace(**new_dict)
