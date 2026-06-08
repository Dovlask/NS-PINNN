"""Ablate Random Weight Factorization from the default (keep ModifiedMlp +
Fourier + grad-norm)."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "no_rwf"
    config.arch.reparam = None
    return config
