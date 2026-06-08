"""Ablate Fourier features from the default (keep ModifiedMlp + RWF + grad-norm)."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "no_fourier_feature"
    config.arch.fourier_emb = None
    return config
