"""Ablation B: pde + bc1 + data, NO far-field (CLAUDE.md Sec. 8) -- data anchors
in place of BC-2. Plain control architecture."""
from config_base import build_config, make_plain, prune_weights


def get_config():
    config = make_plain(build_config())
    config.run_name = "ablation_B"
    config.physics.use_bc2 = False
    return prune_weights(config, ["pde", "bc1", "data"])
