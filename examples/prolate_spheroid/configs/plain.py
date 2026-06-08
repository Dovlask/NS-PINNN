"""Plain baseline = scientific CONTROL (CLAUDE.md Sec. 6): MLP [3,64,64,64,64,1],
tanh, Glorot, NO Fourier, NO RWF, single weight normalization at step 0 then
fixed. All advanced techniques OFF."""
from config_base import build_config, make_plain


def get_config():
    config = make_plain(build_config())
    config.run_name = "plain"
    return config
