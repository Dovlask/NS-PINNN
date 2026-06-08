"""Ablation A (physics only): pde + bc1 + bc2, NO data (CLAUDE.md Sec. 8).
Plain control architecture so A-vs-C isolates the data term."""
from config_base import build_config, make_plain, prune_weights


def get_config():
    config = make_plain(build_config())
    config.run_name = "ablation_A"
    config.physics.use_data = False
    return prune_weights(config, ["pde", "bc1", "bc2"])
