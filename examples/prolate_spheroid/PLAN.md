# PLAN.md — `prolate_spheroid`

PINN para **escoamento potencial axissimétrico (α=0°) ao redor de um esferoide
prolato a/b=4**, validada contra a solução analítica de Lamb (1932) com dados
sintéticos de Cp ruidosos. Fontes canônicas: `Calude.md` (protocolo/integridade,
vinculante) e `Hipoteses_loss_Cp.tex` (física: H1–H4b, BC-1/BC-2, Lamb, perda
com Cp). Prioridade: **rigor e reprodutibilidade** > simplicidade **do código** >
velocidade. **Simplicidade refere-se ao código (legível, funções puras), não ao
escopo: o experimento é completo** e inclui comparação de arquiteturas.

## 1. Estrutura do repo encontrada

Raiz: `examples/`, `jaxpi/` (pacote `archs/models/samplers/evaluator/logging/
utils`), `setup.py` (deps: jax≈0.7, flax, optax, ml_collections, scipy,
matplotlib, wandb, absl-py). `jaxpi/archs.py` já implementa **Mlp**,
**ModifiedMlp**, **FourierEmbs**, **PeriodEmbs** e **reparam/RWF** (random weight
factorization) — exatamente as arquiteturas/técnicas que o paper quer comparar.
Cada exemplo (`adv/`, `ns_steady_cylinder/`) segue `main/train/eval/models.py`
(herda `ForwardBVP`/`ForwardIVP`) + `configs/*.py` (ml_collections) + wandb.

## 2. Onde o caso entra

`examples/prolate_spheroid/`, **autossuficiente**:
```
PLAN.md README.md EXPERIMENTS.md REPORT.md DEVIATIONS.md
constants.py config_base.py configs/*.py main.py
lamb.py geometry.py data_gen.py models.py losses.py train.py evaluate.py
tests/ data/ results/<run_id>/
```

## 3. Decisões (registradas)

- **Objetivo de pesquisa ampliado:** além da validação vs Lamb, **comparar
  diferentes arquiteturas e seus hiperparâmetros** (Mlp vs ModifiedMlp; com/sem
  Fourier features; com/sem RWF; profundidade/largura; ponderação fixa vs
  grad-norm/NTK). Isso é parte do paper.
- **Reuso do stack jaxpi (decisão final — ver `DEVIATIONS.md` D2/D3):** o modelo
  herda `jaxpi.models.ForwardBVP` e reutiliza `archs` (zoo de arquiteturas),
  `samplers`, `evaluator`, `utils` e a ponderação grad-norm/NTK já testada do
  repo — mais rigoroso que reimplementar sem poder testar, e entrega as
  arquiteturas a comparar. O núcleo físico (`lamb/geometry/losses/constants/
  data_gen`) permanece standalone e validado por oráculo NumPy + pytest.
  Config via `configs/*.py` + `ml_collections` (convenção do repo).
- **Técnicas avançadas ON por padrão e configuráveis** (flags em `config.py`):
  Fourier features, RWF/reparam, ModifiedMlp, ponderação adaptativa
  grad-norm/NTK. *Causal weighting* é **N/A** (problema estacionário/BVP, sem
  eixo temporal). Cada técnica liga/desliga por flag para a tabela comparativa.
- **Desvio autorizado:** ligar técnicas avançadas por padrão contraria o
  baseline OFF-by-default do `Calude.md` §3.5/§6 → registrado em `DEVIATIONS.md`
  (autorizado pelo usuário em 2026-06-07; motivo: comparação de arquiteturas é
  objetivo do paper). O baseline §6 (MLP `[3,64,64,64,64,1]` tanh, Glorot,
  ponderação fixa, sem Fourier/RWF) é mantido como **configuração de controle**
  na comparação, não eliminado.
- **Simplicidade do código (mantida):** funções puras nos módulos do caso
  (`lamb/geometry/data_gen/losses/train/evaluate`); config em `config.py`
  simples (sem ml_collections); logging local CSV/JSON em `results/<run_id>/`;
  **wandb não obrigatório**. Reaproveito os helpers de ponderação do
  `jaxpi.models` se reduzirem código; caso contrário, implemento grad-norm/NTK
  como funções curtas no loop standalone (decidir na impl., sem pmap/wandb).
- **Correção b\* = 0.5:** com `L=2b`, b*=b/(2b)=0.5 (o "b*=0.25" do CLAUDE
  original é typo). a*=2, r*_∞=5a/(2b)=10. a*/b*=4 preservado.
- **θ_sep=128.2° invariante em Re_b/U∞:** em Thwaites λ=θ²(dUe/dx)/ν tem θ²∝ν e
  Ue∝U∞ ⇒ λ(x) e o ponto λ=−0.09 independem de ν, U∞ e da escala; só dependem da
  forma a/b. (Conferido contra .tex Apêndice F.)
- **Lamb só fora do treino:** aparece exclusivamente em `data_gen.py` e
  `evaluate.py`. FD apenas como oráculo de teste (pytest); treino/perdas/
  avaliação usam **autodiff**.

## 4. Integridade da comparação de arquiteturas (CRÍTICO)

A seleção de arquitetura/hiperparâmetros usa **somente** (a) piloto sem ruído
(σ=0) ou (b) o split 80/20 das tomadas (validação) — **jamais** a métrica de
campo vs Lamb (§2.5 do Calude.md: "sem tuning no teste"). A métrica final
`rel_L2_field` vs Lamb é computada **uma vez**, após congelar cada config.
Cada arquitetura reportada roda com **≥5 sementes** (média ± dp); proibido
reportar a melhor semente. A tabela comparativa (before/after por técnica) é
**pré-registrada** em `EXPERIMENTS.md` antes dos runs finais.

## 5. Parâmetros da campanha + verificação de hipóteses

| Símbolo | Valor (dimensional, só p/ metadata) |
|---|---|
| Ar 20 °C | ρ=1.204 kg/m³, μ=1.825e−5 Pa·s, ν=1.516e−5 m²/s |
| Escoamento | U∞=30 m/s, b=0.075 m, a=4b=0.3 m, a/b=4 |
| H1 | Re_b=U∞·2b/ν ≈ **2.97e5** > 6e4 ✓ (δ*/b≈1.7% na separação) |
| H3 | Ma=U∞/√(γRT) ≈ **0.087** < 0.3 ✓ |
| H4a | θ_sep=128.2° (filtro das tomadas) |

**Adimensional (a rede só vê isto):** entrada (x*,y*,z*), saída φ*_w;
`Cp_w = 1 − ‖∇*φ*_w‖²`. a*=2, b*=0.5, r*_∞=10. **U∞, ν, T não são inputs.**
**Valores de Lamb (a/b=4), conferidos .tex × Calude.md (tol 1e−3):** e≈0.96825,
α0≈0.1508, k1≈0.0816, Cp(0)≈−0.1699, Ue_max/U∞=1+k1≈1.082.

## 6. Dúvidas

**Nenhuma bloqueante.** Ambiguidades pré-resolvidas pelo prompt: σ~TruncNormal(0,
0.025, [0,0.05]) sorteio único (`per_tap_sigma=False` default), N_taps=40 em
hélice (cosseno em θ∈[5°,128°], azimute 137.5°), filtro θ<θ_sep, dataset
congelado. A nota de compatibilidade de escala do .tex (a=0.2/b=0.05) **não** é
conflito (§4 do Calude.md). Único ponto registrado para acompanhamento: a
amplitude do sweep de arquiteturas (quais combos entram) será fixada no
pré-registro `EXPERIMENTS.md` (gate 4f).

## 7. Roadmap com gates (Calude.md §9)

4b `lamb.py`+`tests/test_lamb.py` (autodiff p/ ∇φ; FD oráculo; ref-values,
∇²φ≈0 c/ taxa O(h²), BC-1, decaimento O((a/r)³), mapa inverso <1e−10) → pytest.
4c `geometry.py`+testes de área (MC vs `S=2πb²(1+(a/(b·e))arcsin e)`, <0.5%).
4d `data_gen.py` → congela `data/cp_synthetic.csv`+`metadata.json` ⛔.
4e `model.py` (arquiteturas via `jaxpi.archs`, por flag)/`losses.py`/`train.py`
+ teste autodiff → smoke 1k ⛔. 4f `EXPERIMENTS.md` pré-registro **incluindo a
matriz arquitetura × técnica e o protocolo de seleção** ⛔. 4g piloto σ=0
(fixa hiperparâmetros) + ablações de perda A/B/C/D **×** sweep de arquiteturas,
≥5 sementes cada. 4h `evaluate.py`+`REPORT.md` (tabelas comparativas).
+ `DEVIATIONS.md` criado no gate 4e/4f registrando o desvio do baseline OFF.
