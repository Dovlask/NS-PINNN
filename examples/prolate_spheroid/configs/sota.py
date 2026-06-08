"""Best-tuned setup: ModifiedMlp (width 256) + Fourier + RWF + adaptive NTK,
longer training. Optional L-BFGS polish can be enabled once Adam is confirmed
working on the GPU (config.training.use_lbfgs = True)."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "sota"
    config.arch.hidden_dim = 256
    config.weighting.scheme = "ntk"
    config.training.max_steps = 80000
    config.training.use_lbfgs = False
    return config
