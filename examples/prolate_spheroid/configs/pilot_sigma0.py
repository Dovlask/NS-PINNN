"""Noise-free pilot (CLAUDE.md Sec. 9.7): complete physics + data that is 100%
coherent with the analytic Lamb solution (sigma = 0). Plain control architecture.
Purpose: fix hyperparameters and sanity-check that physics + perfect data
reproduces Lamb to high accuracy.  Requires the sigma=0 dataset:
    python data_gen.py --sigma0
"""
from config_base import build_config, make_plain


def get_config():
    config = make_plain(build_config())   # complete physics, plain control arch
    config.run_name = "pilot_sigma0"
    config.tap_file = "cp_synthetic_sigma0.csv"
    config.meta_file = "metadata_sigma0.json"
    return config
