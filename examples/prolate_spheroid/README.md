# Prolate spheroid (a/b = 4) — potential flow PINN

Physics-informed neural network for the **axisymmetric (alpha = 0) potential
flow past a prolate spheroid** with aspect ratio `a/b = 4`, validated against the
analytic Lamb (1932) solution using **synthetic noisy Cp data**.

The network learns a harmonic potential `phi*_w(x*, y*, z*)` (Laplace equation),
with surface pressure recovered through Bernoulli: `Cp_w = 1 - ||grad* phi*_w||^2`.
Non-dimensionalization `L = 2b` gives `a* = 2`, `b* = 0.5`, `r*_inf = 10`.

Canonical sources: `../../Calude.md` (execution & integrity protocol, binding) and
`../../Hipoteses_loss_Cp.tex` (physics: H1–H4b, BC-1/BC-2, Lamb, Cp loss).

> **Scientific claim (synthetic phase).** The well-posed BVP (Laplace + BC-1 +
> BC-2) already determines the field without data. This case shows only that the
> pipeline is reproducible and that the noisy Cp term **does not degrade** the
> solution (compare ablation A vs C). It does **not** show that data "improves"
> the solution. No overclaiming.

## How to run (GPU machine, JAX installed)

```bash
# 1. Freeze the synthetic dataset (NumPy/SciPy only; already committed).
python data_gen.py

# 2. Tests must be green before any training.
python -m pytest tests/ -q

# 3. Train a config (writes results/<run_name>/).
python main.py --config=configs/plain.py        # scientific control (baseline)
python main.py --config=configs/default.py       # advanced techniques ON
python main.py --config=configs/ablation_A.py     # physics only (no data)

# 4. Evaluate (metrics.json + figures), after training the same config.
python main.py --config=configs/plain.py --config.mode=eval
```

Run several seeds by overriding from the CLI, e.g.
`--config.seed=7 --config.run_name=plain_s7`.

## Architecture / technique zoo (one of the project goals)

| config | arch | Fourier | RWF | weighting |
|---|---|---|---|---|
| `plain` | Mlp tanh | – | – | fixed (norm @ step 0) |
| `default` | ModifiedMlp | ✓ | ✓ | grad-norm (adaptive) |
| `no_fourier_feature` / `no_rwf` / `no_grad_norm` | ablate one technique from `default` |
| `ntk` | ModifiedMlp | ✓ | ✓ | NTK (adaptive) |
| `sota` | ModifiedMlp (256) | ✓ | ✓ | NTK, 80k steps |

Physics ablations (CLAUDE.md Sec. 8), plain control architecture:
`ablation_A` (pde+bc1+bc2), `ablation_B` (pde+bc1+data), `ablation_C` (complete),
`ablation_D` (data only — expected to fail in the field).

## Equation → code map

| Source (`.tex` / Lamb) | Code |
|---|---|
| eq.(e),(alpha0),(k1); Art.103 | `constants.py`: `E`, `ALPHA0`, `K1` |
| eq.(Q1),(Q1') | `lamb.py`: `Q1`, `Q1_prime` |
| eq.(phi), Art.103 p.139 | `lamb.py`: `phi` |
| eq.(q2),(Cp), Art.114 eq.(13) | `constants.py`: `cp_surface` |
| inverse map (xi,eta)<->(x,rho) | `lamb.py`: `cart_to_spheroidal`, `spheroidal_to_cart` |
| eq.(Cp-PINN) Cp = 1 - ‖grad*phi*‖² | `losses.py`: `cp_from_grad`; `lamb.py`: `cp_field` |
| eq.(Laplace), loss-pde | `losses.py`: `laplacian`; `models.py` pde term |
| BC-1, loss-bc1; n_hat = grad f/‖grad f‖ | `losses.py`: `bc1_residual`; `geometry.py`: `surface_normals` |
| BC-2, loss-bc2 | `losses.py`: `bc2_residual` |
| loss-data | `models.py`: `losses` data term |
| S = 2πb*²(1+(a*/(b*e))arcsin e) | `constants.py`: `surface_area_star` |

## Files

`constants.py` physics/geometry (single source of truth) · `lamb.py` analytic
solution (evaluate/tests only) · `geometry.py` seeded sampling pools · `losses.py`
pure residual operators · `models.py` PINN (jaxpi `ForwardBVP`) · `data_gen.py`
freezes `data/cp_synthetic.csv`+`metadata.json` · `train.py` / `evaluate.py` /
`main.py` · `configs/` zoo · `tests/` pytest · `EXPERIMENTS.md` pre-registration ·
`REPORT.md` results · `DEVIATIONS.md` authorized deviations.

## Integrity (CLAUDE.md Sec. 2)

Lamb never enters training (only `data_gen.py` + `evaluate.py`). Dataset frozen;
seeds separate and recorded in `data/metadata.json`. Finite differences appear
only in tests, as an independent oracle for autodiff. Criteria are pre-registered
in `EXPERIMENTS.md` and never changed after seeing results.
