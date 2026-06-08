"""Ablation D: DATA ONLY (no pde/bc1/bc2), CLAUDE.md Sec. 8 -- must fail in the
field (shows the physics is necessary). Plain control architecture."""
from config_base import build_config, make_plain, prune_weights


def get_config():
    config = make_plain(build_config())
    config.run_name = "ablation_D"
    config.physics.use_pde = False
    config.physics.use_bc1 = False
    config.physics.use_bc2 = False
    return prune_weights(config, ["data"])
