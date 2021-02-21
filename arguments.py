import argparse
from typing import Any

data = None
defaults = {'data_folder': '../cardiovascular-risk-data/', 'fast': False, 'single_core': False}


def parse():
    global data
    parser = argparse.ArgumentParser(description='Cardiovascular risk backend.')
    parser.add_argument('--data-folder', )
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--single-core', action='store_true')
    data = parser.parse_args()


def get_params(key: str) -> Any:
    try:
        param_value = getattr(data, key)
    except AttributeError:
        param_value = None

    return param_value if param_value is not None else defaults[key]
