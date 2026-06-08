"""NTK adaptive weighting variant of the default architecture."""
from config_base import build_config


def get_config():
    config = build_config()
    config.run_name = "ntk"
    config.weighting.scheme = "ntk"
    return config
