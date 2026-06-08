"""Ablation C: COMPLETE (pde + bc1 + bc2 + data), the article configuration
(CLAUDE.md Sec. 8). Plain control architecture; A-vs-C quantifies the data term."""
from config_base import build_config, make_plain


def get_config():
    config = make_plain(build_config())   # all four terms already active
    config.run_name = "ablation_C"
    return config
