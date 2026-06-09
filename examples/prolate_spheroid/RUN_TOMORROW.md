# RUN_TOMORROW.md — passo a passo na máquina GPU do laboratório

Guia completo para o computador do laboratório: montar o **ambiente virtual**,
validar, rodar os testes, fazer um *smoke run* e então treinar o PINN de verdade
e **conferir a convergência**. Tudo é rodado de dentro de `examples/prolate_spheroid/`.

> Integridade: os datasets já estão **congelados** em `data/`
> (`cp_synthetic.csv`, σ=0.029113; e `cp_synthetic_sigma0.csv`, piloto σ=0).
> Determinísticos pelas sementes. Lamb só entra em `data_gen.py` e `evaluate.py`.

---

## 0. Ambiente virtual (SIM, use venv — máquina compartilhada)

Numa máquina de laboratório compartilhada, **crie um ambiente virtual** (não
instale no Python do sistema). Duas opções; escolha uma.

> ⚠️ **JAX + GPU é oficialmente Linux.** Se o computador do lab for **Linux**, o
> caminho abaixo dá GPU direto. Se for **Windows nativo**, o JAX-GPU não tem wheel
> oficial — use **WSL2** (Ubuntu) e siga o caminho Linux lá dentro, OU rode em CPU
> (funciona, só mais lento). Cheque com `nvidia-smi` se há GPU e qual CUDA.

### Opção A — venv + pip (Linux ou WSL2)
```bash
# na RAIZ do repo (onde está setup.py)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .                      # instala o pacote jaxpi + deps base (CPU jax)
pip install -U "jax[cuda12]"          # GPU: casa com CUDA 12.x do lab (veja nvidia-smi)
pip install torch pytest              # torch: usado por jaxpi/samplers.py; pytest p/ os testes
```

### Opção B — conda
```bash
conda create -n pinn python=3.11 -y
conda activate pinn
pip install -e .
pip install -U "jax[cuda12]" torch pytest
```

### Confirme o ambiente
```bash
python -c "import jax; print('devices:', jax.devices())"   # espera [CudaDevice(...)]
python -c "import flax, optax, ml_collections, scipy, torch; print('deps OK')"
cd examples/prolate_spheroid
```
Se `jax.devices()` listar só CPU, o treino roda mas lento — revise a instalação
do `jax[cuda12]` / WSL2.

---

## 1. Testes (GATE — não treine antes de ficar verde)

```bash
python -m pytest tests/ -q
```
Cobre: `test_geometry.py` (área MC < 0.5%, pools fora do corpo — já passou sem
JAX), `test_lamb.py` (constantes de Lamb; ∇²φ≈0 com taxa **O(h²)**; BC-1;
decaimento far-field ~ −3; mapa inverso < 1e−10) e `test_autodiff.py` (laplaciano
da rede por autodiff vs FD — pega o bug do jacobiano da normalização).
Se algum falhar, **pare e me mande a saída**.

---

## 2. Smoke run (~2 min — confirma que treina ponta a ponta)

```bash
python main.py --config=configs/plain.py \
  --config.run_name=smoke --config.training.max_steps=2000 \
  --config.logging.log_every_steps=100 --config.saving.save_every_steps=2000
python main.py --config=configs/plain.py --config.run_name=smoke --config.mode=eval
```
As perdas devem **descer**; gera `results/smoke/loss_history.csv`, `metrics.json`
e `figures/*.png` sem erro. Se rodar sem crashar, a integração JAX/jaxpi está OK.

> **OOM na GPU?** reduza `--config.training.batch_size_per_device=1024` e/ou
> `--config.sampling.n_surface=2000 --config.sampling.n_collocation=10000`.
> Múltiplas GPUs com comportamento estranho no `pmap`? fixe uma: `export CUDA_VISIBLE_DEVICES=0`.

---

## 3. Treino real + checagem de convergência (progressão recomendada)

A ordem vai do mais limpo ao mais realista:

```bash
# (a) PILOTO sigma=0 : fisica + dados 100% coerentes com a analitica (Lamb)
#     -> o dataset noise-free ja esta congelado; se quiser regerar:  python data_gen.py --sigma0
python main.py --config=configs/pilot_sigma0.py
python main.py --config=configs/pilot_sigma0.py --config.mode=eval

# (b) FISICA PURA (sem dados) : o BVP sozinho deve convergir ao Lamb
python main.py --config=configs/ablation_A.py
python main.py --config=configs/ablation_A.py --config.mode=eval

# (c) COMPLETA (controle, plain) : fisica + dados de Cp ruidosos (sigma>0)
python main.py --config=configs/plain.py
python main.py --config=configs/plain.py --config.mode=eval

# (d) TECNICAS AVANCADAS ON (ModifiedMlp+Fourier+RWF+grad-norm)
python main.py --config=configs/default.py
python main.py --config=configs/default.py --config.mode=eval
```

Por que (a) é um passo coerente: o piloto σ=0 é prescrito pelo protocolo
(CLAUDE.md Sec. 9.7) para fixar hiperparâmetros e é a checagem mais limpa de que
o pipeline reproduz o Lamb — com dados perfeitos + física, espere `rel_L2_field`
**bem pequeno** (≪ 2%). É distinto de (b): em (a) o termo de dados está ativo
(dados = analítico); em (b) não há dados, só física.

### Como saber se CONVERGIU — abra `results/<run>/metrics.json`

O bloco `"pass"` traz os critérios pré-registrados (`EXPERIMENTS.md`):

| métrica | alvo (σ>0) | piloto σ=0 |
|---|---|---|
| `rel_L2_field` | **≤ 0.02** | idem (espere muito menor) |
| `RMSE_cp_surface` | **≤ 0.0291** (σ) | **≤ 1e−2** (absoluto) |
| `sigma_hat_holdout` | **∈ [0.0146, 0.0437]** | **≤ 1e−2** (≈ 0) |
| `rms_pde/bc1/bc2_residual` | **≤ 1e−2** | idem |

(No σ=0, `metrics.json` marca `"noise_free_pilot": true` e troca os critérios
relativos a σ por bares absolutos — a banda de σ degenera a zero.)

Sinais visuais (`results/<run>/figures/`): `cp_eta.png` (linha PINN **em cima**
da Lamb), `loss_curves.png` (tudo descendo), `error_map.png` (erro pequeno),
`residual_hist.png` (resíduos ~ N(0,σ²); no piloto a gaussiana é omitida).

Diagnóstico: `sigma_hat ≪ σ` → overfit do ruído (reportar como falha);
`rel_L2_field` alto → treine mais (`--config.training.max_steps=80000`) ou use
`configs/default.py`. **`ablation_D` (só dados) DEVE falhar no campo** — é o ponto.

---

## 4. Campanha completa (depois que (a)–(d) convergirem)

Ablações físicas + varredura de arquiteturas, **≥ 5 sementes cada** (média ± dp;
nunca só a melhor semente):

```bash
for cfg in ablation_A ablation_B ablation_C ablation_D plain default \
           no_fourier_feature no_rwf no_grad_norm ntk sota; do
  for s in 42 7 13 21 100; do
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s}
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s} --config.mode=eval
  done
done
```
Preencha as tabelas de `REPORT.md` a partir dos `metrics.json`. Claim honesto:
**A ≈ C (os dados não degradam)** — nunca “os dados melhoraram a solução”.

> **L-BFGS** está implementado mas **OFF por padrão** (`config.training.use_lbfgs`).
> Só ligue (`--config.training.use_lbfgs=True`) depois que o Adam convergir — é o
> único trecho não testado nesta máquina; o checkpoint do Adam é salvo antes, então
> mesmo se o L-BFGS falhar o resultado do Adam fica intacto.
