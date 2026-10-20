"""
Poor Man's Configurator (from nanoGPT). Override any config global from the command line:
$ python train_dino.py --batch_size=128 --out_dir=my_run
or with python config files (applied in order, then --key=value overrides on top):
$ python train_dino.py config/my_idea.py --batch_size=128

Config files are plain python assignments. Unlike nanoGPT, their keys and types are checked the same way as
command line overrides, so a typo like `batchsize = 64` is an error instead of being silently ignored.
"""

import sys
import types
from ast import literal_eval

def _set_config(key, val, source):
    if key not in globals():
        raise ValueError(f"Unknown config key in {source}: {key}")
    # ints are fine where floats are expected
    if isinstance(globals()[key], float) and isinstance(val, int) and not isinstance(val, bool):
        val = float(val)
    assert type(val) == type(globals()[key]), f'{key} ({source}): expected {type(globals()[key]).__name__}, got {type(val).__name__}'
    print(f"Overriding: {key} = {val}")
    globals()[key] = val

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--'), f'expected --key=value, got {arg}'
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            source = f.read()
        print(source)
        namespace = {}
        exec(source, {}, namespace)
        for key, val in namespace.items():
            # skip helpers the file may define (imports, functions, _private names)
            if key.startswith('_') or isinstance(val, (types.ModuleType, types.FunctionType, type)):
                continue
            _set_config(key, val, config_file)
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--'), f'expected --key=value, got {arg}'
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals() and isinstance(globals()[key], str):
            # string configs stay raw strings (e.g. comma separated lists like --cfgs=1.0,2.0)
            attempted_val = val
        else:
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempted_val = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempted_val = val
        _set_config(key, attempted_val, 'command line')
