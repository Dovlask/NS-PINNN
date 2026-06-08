# RUN_TOMORROW.md — passo a passo na máquina GPU

Guia para amanhã: validar o ambiente, rodar os testes, fazer um *smoke run*, e
então treinar o PINN de verdade e **conferir a convergência**. Tudo é rodado de
dentro de `examples/prolate_spheroid/`.

> Lembrete de integridade: o dataset já está **congelado** em `data/` (σ=0.029113,
> sementes registradas). Não regere a menos que queira; é determinístico pelas
> sementes. Lamb só entra em `data_gen.py` e `evaluate.py` — nunca no treino.

---

## 0. Pré-requisitos (uma vez)

```bash
# na RAIZ do repo (onde está setup.py): instala o pacote jaxpi em modo editável
pip install -e .
# confirme que JAX enxerga a GPU e que as libs estão lá
python -c "import jax; print('devices:', jax.devices())"
python -c "import flax, optax, ml_collections, scipy, torch; print('deps OK')"
```
`torch` é necessário porque `jaxpi/samplers.py` usa `torch.utils.data.Dataset`.
Se `jax.devices()` mostrar só CPU, o treino roda mas lento — confira a instalação
CUDA do JAX.

```bash
cd examples/prolate_spheroid
```

---

## 1. Testes (GATE — não treine antes de ficar verde)

```bash
python -m pytest tests/ -q
```
Esperado: **todos verdes**. O que cada um cobre:
- `test_geometry.py` — área Monte-Carlo vs analítica < 0.5%, pools fora do corpo
  (já passou na máquina sem JAX).
- `test_lamb.py` — constantes de Lamb (e, α0, k1, Cp(0), Ue_max), ∇²φ≈0 com taxa
  **O(h²)**, BC-1 na superfície, decaimento far-field ~ −3, mapa inverso < 1e−10.
- `test_autodiff.py` — laplaciano da rede por autodiff vs diferenças finitas
  (pega o bug do jacobiano da normalização 1/r*_inf).

Se algum falhar, **pare e me mande a saída** — não prossiga para o treino.

---

## 2. Smoke run (~2 min — confirma que o pipeline treina ponta a ponta)

```bash
python main.py --config=configs/plain.py \
  --config.run_name=smoke --config.training.max_steps=2000 \
  --config.logging.log_every_steps=100 --config.saving.save_every_steps=2000
python main.py --config=configs/plain.py --config.run_name=smoke --config.mode=eval
```
O que olhar (sanidade, NÃO convergência final):
- O treino imprime as perdas a cada 100 passos e elas **descem**.
- Gera `results/smoke/loss_history.csv` e, no eval, `results/smoke/metrics.json`
  + `results/smoke/figures/*.png` sem erro.

Se o smoke rodar sem crashar, a integração JAX/jaxpi está OK e pode ir ao treino real.

> Se der **OOM (out of memory)** na GPU: reduza `--config.training.batch_size_per_device=1024`
> e/ou `--config.sampling.n_surface=2000 --config.sampling.n_collocation=10000`.
> Se houver mais de uma GPU e algo estranho no `pmap`, fixe uma: `export CUDA_VISIBLE_DEVICES=0`.

---

## 3. Treino real + checagem de convergência

Comece pela **física pura** (mais limpa para ver se converge ao Lamb), depois a
configuração completa, depois as técnicas avançadas:

```bash
# (a) física pura (sem dados) — deve convergir ao campo de Lamb
python main.py --config=configs/ablation_A.py
python main.py --config=configs/ablation_A.py --config.mode=eval

# (b) completa (controle, plain) — física + dados de Cp
python main.py --config=configs/plain.py
python main.py --config=configs/plain.py --config.mode=eval

# (c) técnicas avançadas ON (ModifiedMlp+Fourier+RWF+grad-norm)
python main.py --config=configs/default.py
python main.py --config=configs/default.py --config.mode=eval
```

### Como saber se CONVERGIU — abra `results/<run>/metrics.json`

O bloco `"pass"` traz os critérios pré-registrados (`EXPERIMENTS.md`). Convergência
boa = todos `true`:

| métrica em metrics.json | alvo | significado |
|---|---|---|
| `rel_L2_field` | **≤ 0.02** | erro de velocidade vs Lamb em 10k pontos 3D |
| `RMSE_cp_surface` | **≤ 0.0291** (σ) | Cp na superfície vs Lamb (grade densa) |
| `sigma_hat_holdout` | **∈ [0.0146, 0.0437]** | ruído recuperado ≈ σ (denoising) |
| `rms_pde/bc1/bc2_residual` | **≤ 1e−2** | resíduos da física pequenos |

Sinais visuais (`results/<run>/figures/`):
- `cp_eta.png` — a linha **PINN deve cair em cima da linha Lamb**; tomadas como pontos ±σ.
- `loss_curves.png` — todas as componentes **descendo** e estabilizando.
- `error_map.png` — erro pequeno e sem manchas perto da superfície.
- `residual_hist.png` — resíduos das tomadas ~ gaussiana N(0, σ²) sobreposta.

Diagnóstico rápido:
- `sigma_hat ≪ σ` → overfit do ruído (reportar como falha, não “consertar” no olho).
- `sigma_hat ≫ σ` ou `rel_L2_field` alto → subajuste: deixe treinar mais passos
  (`--config.training.max_steps=80000`) ou use `configs/default.py` (técnicas ON).
- **`ablation_D` (só dados) DEVE falhar no campo** — isso é esperado e é o ponto.

---

## 4. Campanha completa (depois que (a)–(c) convergirem)

Ablações físicas + varredura de arquiteturas, **≥ 5 sementes cada** (média ± dp;
nunca reporte só a melhor semente). Exemplo de laço:

```bash
for cfg in ablation_A ablation_B ablation_C ablation_D plain default \
           no_fourier_feature no_rwf no_grad_norm ntk sota; do
  for s in 42 7 13 21 100; do
    python main.py --config=configs/$cfg.py \
      --config.seed=$s --config.run_name=${cfg}_s${s}
    python main.py --config=configs/$cfg.py \
      --config.seed=$s --config.run_name=${cfg}_s${s} --config.mode=eval
  done
done
```
Depois, preencha as tabelas de `REPORT.md` a partir dos `metrics.json`. O claim
honesto do caso sintético é **A ≈ C (os dados não degradam)** — nunca “os dados
melhoraram a solução”.

> **L-BFGS** está implementado mas **desligado por padrão** (`config.training.use_lbfgs`).
> Só ligue (`--config.training.use_lbfgs=True`) depois de confirmar que o Adam
> converge — ele é o único trecho não testado nesta máquina. O checkpoint do Adam
> é salvo antes, então mesmo se o L-BFGS falhar, o resultado do Adam fica intacto.
