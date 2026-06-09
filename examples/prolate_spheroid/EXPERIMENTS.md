# EXPERIMENTS.md — pre-registration

Written **before** the final training runs (CLAUDE.md Sec. 2.1). Criteria here are
fixed and are not changed retroactively after seeing results.

## Hypotheses

* **H-pipeline.** The full pipeline (Lamb → noise → filter → train → evaluate) runs
  end-to-end and is reproducible from the recorded seeds.
* **H-denoising.** With config C (complete), the noisy Cp term does **not** degrade
  the field relative to config A (physics only): `rel_L2_field(C) ≈ rel_L2_field(A)`.
* **H-sigma.** The residual scale recovered at the taps matches the injected noise:
  `sigma_hat ≈ sigma`.
* **H-physics-needed.** Config D (data only) fails in the field (large `rel_L2_field`).
* **H-architecture.** Advanced techniques (Fourier, RWF, ModifiedMlp, grad-norm/NTK)
  change convergence/accuracy; reported as a before/after table.

## Metrics (evaluate.py → metrics.json)

`rel_L2_field` = ‖grad phi_w − grad phi_lamb‖₂ / ‖grad phi_lamb‖₂ on the fixed 10k
3D set (seed `SEED_EVAL_FIELD`, never used in training) · `RMSE_cp_surface`,
`max_abs_dCp_surface` on the dense η grid (600 pts, not the taps) · `sigma_hat_train`,
`sigma_hat_holdout` vs `sigma` · `rms_pde/bc1/bc2_residual`.

## Acceptance criteria (CLAUDE.md Sec. 7)

| criterion | threshold |
|---|---|
| `rel_L2_field` | ≤ 0.02 |
| `RMSE_cp_surface` | ≤ sigma (denoising by physics) |
| `sigma_hat_holdout` | in [0.5 sigma, 1.5 sigma] |
| `rms_pde/bc1/bc2_residual` | ≤ 1e-2 (≥ 2 orders below O(1)) |

`sigma_hat ≪ sigma` ⇒ noise overfitting (report as failure); `sigma_hat ≫ sigma`
⇒ underfitting. Failures are documented in `REPORT.md`, not silently fixed.

## Frozen dataset

`data/cp_synthetic.csv` + `data/metadata.json`. sigma = 0.029113 (drawn once,
`seed_sigma = 20240601`). 40/40 taps survive the θ < 128.2° filter. Re_b ≈ 2.97e5
(H1 ✓), Ma ≈ 0.087 (H3 ✓). Training never reads sigma.

Noise-free pilot dataset: `data/cp_synthetic_sigma0.csv` + `metadata_sigma0.json`
(cp_med = cp_true, sigma = 0; `python data_gen.py --sigma0`), used by
`configs/pilot_sigma0.py` to fix hyperparameters (CLAUDE.md Sec. 9.7).

## Runs

* **Noise-free pilot** (sigma=0, complete physics, plain arch): `pilot_sigma0`.
* **Physics ablations** (plain control arch): A, B, C, D.
* **Architecture / technique sweep** (config C physics): `plain`, `default`,
  `no_fourier_feature`, `no_rwf`, `no_grad_norm`, `ntk`, `sota`.
* **Seeds:** each reported config runs with ≥ 5 network seeds
  (`--config.seed ∈ {42, 7, 13, 21, 100}`, `--config.run_name=<cfg>_s<seed>`).
  Report mean ± std; reporting only the best seed is forbidden.

## Selection protocol (integrity — no tuning on the test)

Architecture and hyper-parameters are chosen using ONLY (a) a sigma = 0 pilot or
(b) the 80/20 tap split (validation). The field metric vs Lamb is computed ONCE
per frozen config and is **never** used for selection (CLAUDE.md Sec. 2.5).

## Honest claim

A vs C quantifies the data-term effect: expected `C ≈ A` (no degradation). The
report must not state that data improved the solution in the synthetic case.
