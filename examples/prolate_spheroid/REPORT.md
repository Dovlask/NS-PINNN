# REPORT.md — results (to be filled after the runs)

> Skeleton committed with the code. Numbers are filled by `evaluate.py`
> (metrics.json) after the GPU runs. No figure is edited by hand.

## Setup

* Geometry: prolate spheroid a/b = 4, alpha = 0 (axisymmetric).
* Dataset: `data/cp_synthetic.csv`, sigma = 0.029113 (seed_sigma = 20240601),
  40 taps, Re_b ≈ 2.97e5, Ma ≈ 0.087.
* Commit hash of the runs: _TBD_.

## Physics ablations (plain control arch, mean ± std over 5 seeds)

| config | rel_L2_field | RMSE_cp_surface | sigma_hat_holdout | pde res | bc1 res | bc2 res |
|---|---|---|---|---|---|---|
| A (pde+bc1+bc2) | _TBD_ | – | – | _TBD_ | _TBD_ | _TBD_ |
| B (pde+bc1+data) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | – |
| C (complete) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| D (data only) | _TBD_ (expected large) | _TBD_ | _TBD_ | – | – | – |

**A vs C (honest claim):** _TBD_ — expected C ≈ A (data does not degrade).

## Architecture / technique sweep (config C physics, mean ± std over 5 seeds)

| config | rel_L2_field | RMSE_cp_surface | steps-to-converge |
|---|---|---|---|
| plain | _TBD_ | _TBD_ | _TBD_ |
| default (ModifiedMlp+Fourier+RWF+grad-norm) | _TBD_ | _TBD_ | _TBD_ |
| no_fourier_feature | _TBD_ | _TBD_ | _TBD_ |
| no_rwf | _TBD_ | _TBD_ | _TBD_ |
| no_grad_norm | _TBD_ | _TBD_ | _TBD_ |
| ntk | _TBD_ | _TBD_ | _TBD_ |
| sota | _TBD_ | _TBD_ | _TBD_ |

## Acceptance criteria (pass/fail)

_TBD_ — from each run's `metrics.json["pass"]`.

## sigma_hat vs sigma

_TBD_ — table of sigma_hat_train / sigma_hat_holdout vs sigma = 0.029113.

## Limitations

* Synthetic case: Lamb holds everywhere, so data cannot add information the
  physics lacks; only non-degradation + reproducibility are claimed.
* Future experimental phase: at Re_a ≈ 1.2e6, turbulent transition before
  separation is plausible; treat Thwaites as an upper bound (Calude.md Sec. 4,
  .tex H1/H4a).
* L-BFGS refinement was not validated on the dev machine (DEVIATIONS.md D4).

## Failures / deviations

_TBD_ — any config that did not converge is reported here, not silently dropped.
