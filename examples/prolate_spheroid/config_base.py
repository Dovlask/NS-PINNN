"""Shared base configuration builder for every config in configs/.

Importable from the config files because main.py's directory (this folder) is on
sys.path at run time.

build_config() returns the COMPLETE case (config C) with the paper's advanced
techniques ON: ModifiedMlp + Fourier features + RWF + adaptive grad-norm
weighting (the user's directive -- DEVIATIONS.md D1).  Variants turn things OFF.

ml_collections type-safety note: every override below is either to None
(ConfigDict -> None is always permitted) or same-type (str/int/bool), so we never
rely on a None -> ConfigDict transition.  make_plain() recovers the CLAUDE.md
Sec. 6 control: MLP [3,64,64,64,64,1], tanh, no Fourier, no RWF, fixed weights.
"""
import ml_collections


def build_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"
    config.run_name = "default"

    # --- architecture (advanced techniques ON by default) ---
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.out_dim = 1                       # scalar potential phi*
    arch.activation = "tanh"               # smooth: loss uses 2nd derivatives
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 128})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 0.5, "stddev": 0.1})

    # --- optimizer (Adam + exponential decay) ---
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0

    # --- training ---
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 40000
    training.batch_size_per_device = 2048   # collocation minibatch
    training.use_lbfgs = False              # optional refinement (DEVIATIONS.md D4)
    training.lbfgs_max_iter = 2000

    # --- weighting ---
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"          # {grad_norm, ntk}
    weighting.adaptive = True               # False -> single normalization at step 0
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000
    weighting.init_weights = ml_collections.ConfigDict(
        {"pde": 1.0, "bc1": 1.0, "bc2": 1.0, "data": 1.0}
    )

    # --- physics terms active (A/B/C/D ablations); base = complete (C) ---
    config.physics = physics = ml_collections.ConfigDict()
    physics.use_pde = True
    physics.use_bc1 = True
    physics.use_bc2 = True
    physics.use_data = True

    # --- collocation / surface / far-field pool sizes ---
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_collocation = 20000
    sampling.n_surface = 4000
    sampling.n_farfield = 2000

    # --- logging ---
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 200
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False
    logging.log_errors = False

    # --- saving ---
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 1

    # --- shapes and seeds (CLAUDE.md Sec. 2.2: separate, registered) ---
    config.input_dim = 3
    config.seed = 42                # (c) network init
    config.seed_collocation = 101   # (d) collocation pool + sampler
    config.seed_surface = 202       #     surface pool
    config.seed_far = 303           #     far-field pool
    config.seed_split = 404         # (e) 80/20 tap split

    return config


def make_plain(config):
    """CLAUDE.md Sec. 6 control: MLP [3,64,64,64,64,1], tanh, no Fourier, no RWF,
    single weight normalization at step 0 then fixed. Only ->None / same-type
    overrides (always permitted by ml_collections)."""
    config.arch.arch_name = "Mlp"
    config.arch.hidden_dim = 64
    config.arch.fourier_emb = None
    config.arch.reparam = None
    config.weighting.adaptive = False
    return config


def prune_weights(config, active_keys):
    """Rebuild init_weights to match the active loss terms (ablations)."""
    config.weighting.init_weights = ml_collections.ConfigDict(
        {k: 1.0 for k in active_keys}
    )
    return config
