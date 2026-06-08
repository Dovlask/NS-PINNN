# DEVIATIONS.md — authorized departures from the protocol

Every entry is an explicit deviation from `Calude.md`, authorized by the user
(CLAUDE.md Sec. 2.10). Nothing here changes the physics, the loss, the noise
protocol, the ablations, or the acceptance criteria.

## D1 — Advanced techniques ON by default
*Authorized by the user on 2026-06-07.*
CLAUDE.md Sec. 3.5 / Sec. 6 set the advanced techniques (Fourier features, RWF,
ModifiedMlp, grad-norm/NTK weighting) OFF by default. They are now **ON by default**
because **comparing architectures and their parameters is an explicit goal of the
paper**. The CLAUDE.md Sec. 6 baseline is preserved exactly as the `plain` config
(scientific control), so the before/after comparison is intact.

## D2 — Reuse of the jaxpi training stack
The PINN subclasses `jaxpi.models.ForwardBVP` and reuses `archs` (the architecture
zoo), `samplers`, `evaluator`, `utils`, and the grad-norm/NTK weighting. This
overrides the "standalone, at most one class" style of CLAUDE.md Sec. 3.3.
Reason: reusing the repo's tested training/weighting code is more rigorous than
re-implementing it untested, and it directly provides the architectures to compare.
The physics core (`lamb`, `geometry`, `losses`, `constants`, `data_gen`) stays
standalone and is validated independently (NumPy oracle + pytest).

## D3 — Configuration via `configs/` + ml_collections
CLAUDE.md Sec. 3.3 suggested a single `config.py`. We use the repo convention
(`configs/*.py` with `ml_collections`, loaded by `config_flags`) so each
architecture/technique variant is a first-class config (required by the sweep).
Physics constants remain centralized in `constants.py`.

## D4 — Final L-BFGS refinement OFF by default
CLAUDE.md Sec. 6 lists an L-BFGS polish after Adam. It is **implemented**
(`train.lbfgs_refine`, `config.training.use_lbfgs`) but **default OFF** because it
could not be executed/validated on the development machine (no JAX there). The
Adam checkpoint is always saved first and the L-BFGS phase is wrapped, so enabling
it cannot corrupt a run. Enable it on the GPU once Adam is confirmed working.

## Logging
W&B is not used (CLAUDE.md Sec. 3 allows this); logging is local CSV/JSON in
`results/<run_name>/`. This is a permitted choice, listed here for completeness.
