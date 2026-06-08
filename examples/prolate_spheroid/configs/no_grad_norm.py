"""Ablate adaptive grad-norm weighting (single normalization at step 0, then
fixed). Keeps ModifiedMlp + Fourier + RWF."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "no_grad_norm"
    config.weighting.adaptive = False
    return config
