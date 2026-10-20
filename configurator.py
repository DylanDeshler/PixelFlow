"""
Poor Man's Configurator (from nanoGPT). Override any config global from the command line:
$ python train_dino.py --batch_size=128 --out_dir=my_run
or with a python config file:
$ python train_dino.py config/my_idea.py --batch_size=128
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            if isinstance(globals()[key], str):
                # string configs stay raw strings (e.g. comma separated lists like --cfgs=1.0,2.0)
                attempted_val = val
            else:
                try:
                    # attempt to eval it (e.g. if bool, number, or etc)
                    attempted_val = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempted_val = val
            # ints are fine where floats are expected
            if isinstance(globals()[key], float) and isinstance(attempted_val, int):
                attempted_val = float(attempted_val)
            assert type(attempted_val) == type(globals()[key]), f'{key}: expected {type(globals()[key])}, got {type(attempted_val)}'
            print(f"Overriding: {key} = {attempted_val}")
            globals()[key] = attempted_val
        else:
            raise ValueError(f"Unknown config key: {key}")
