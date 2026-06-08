"""Default = COMPLETE case (config C) with advanced techniques ON:
ModifiedMlp (width 128) + Fourier features + RWF + adaptive grad-norm weighting.
This is exactly build_config()."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "default"
    return config
