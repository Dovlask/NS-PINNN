# RUN_TOMORROW.md — rodar o PINN (PC do lab: Windows + RTX A2000)

Copia-e-cola, de cima pra baixo. Os datasets já vêm no clone (congelados), então
**não precisa** rodar `data_gen` no lab.

---

## ⚠️ STATUS / qual plano seguir

O JAX só usa a **GPU NVIDIA via WSL2 (Linux)**, e instalar o WSL **exige admin**.
O help desk só autoriza amanhã/depois. Então:

* **PLANO A — HOJE, em CPU (sem admin):** valida que o código roda ponta a ponta
  (testes + smoke + um piloto curto). **Não usa a A2000** e é lento — serve pra
  *provar correção*, não pra rodar a campanha. → **Seção A** abaixo.
* **PLANO B — quando liberarem admin (GPU de verdade):** instala o WSL2 e roda a
  campanha completa, rápido. → **Seção B** abaixo.

> Em CPU, o que importa é o processador, não a placa de vídeo — ou seja, a
> vantagem desse PC (a A2000) fica **parada**. Use o Plano A só pra validar hoje;
> guarde os treinos pesados pro Plano B.

---

# PLANO A — rodar HOJE em CPU (Windows, sem admin)

### A.0 — Pré-requisitos (instaláveis por-usuário, sem admin)
No **PowerShell** confira se já tem Python e Git:
```powershell
python --version
git --version
```
* Se faltar **Python**: baixe em https://www.python.org/downloads/ (3.10–3.12) e,
  no instalador, **DESmarque "Install for all users"** e **marque "Add to PATH"**
  → instala na sua conta, sem admin.
* Se faltar **Git**: https://git-scm.com/download/win → na tela de permissões
  escolha a opção **per-user** (sem admin).

### A.1 — Montar o projeto (PowerShell, dentro de `C:\Users\laskd\NS-PINNN`)
Você já tem o repo aqui, então não precisa clonar:
```powershell
git checkout EP
git pull
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
pip install jax jaxlib torch pytest
cd examples\prolate_spheroid
python -c "import jax; print('JAX devices:', jax.devices())"
```
A última linha vai mostrar **`[CpuDevice(id=0)]`** — isso é o **esperado** em CPU.
Se der erro ao instalar `jaxlib` no Windows, veja Troubleshooting (item 4).

> Toda vez que reabrir o PowerShell: `cd C:\Users\laskd\NS-PINNN; .venv\Scripts\Activate.ps1; cd examples\prolate_spheroid`

### A.2 — Testes (GATE — o entregável de hoje)
```powershell
python -m pytest tests/ -q
```
Todos verdes = **a física e o código estão corretos**. Isso já é a vitória do dia.
Se algo falhar, **me mande a saída**.

### A.3 — Smoke run em CPU (~minutos — confirma que treina ponta a ponta)
```powershell
python main.py --config=configs/plain.py --config.run_name=smoke_cpu --config.training.max_steps=500 --config.training.batch_size_per_device=1024 --config.sampling.n_collocation=8000 --config.sampling.n_surface=2000 --config.logging.log_every_steps=50 --config.saving.save_every_steps=500
```
```powershell
python main.py --config=configs/plain.py --config.run_name=smoke_cpu --config.mode=eval
```
As perdas devem **descer** e gerar `results/smoke_cpu/` com `metrics.json` e figuras.
Não cobre os critérios de aceite (poucos passos) — é só prova de vida.

### A.4 — (Opcional) Piloto σ=0 curto em CPU
Só se a A.3 rodou rápido e você quiser ver mais convergência. Ainda é lento:
```powershell
python main.py --config=configs/pilot_sigma0.py --config.run_name=pilot_cpu --config.training.max_steps=5000 --config.training.batch_size_per_device=1024 --config.sampling.n_collocation=10000 --config.logging.log_every_steps=200 --config.saving.save_every_steps=1000
python main.py --config=configs/pilot_sigma0.py --config.run_name=pilot_cpu --config.mode=eval
```
**NÃO** rode a campanha (Seção B.5) em CPU — levaria dias/semanas.

---

# PLANO B — GPU de verdade via WSL2 (quando liberarem admin)

## B.0 Pôr o JAX na GPU via WSL2 (faça uma vez)

### B.0.1 — Checar se o WSL já existe (PowerShell normal)
```powershell
wsl -l -v
```
- Se listar **Ubuntu** com **VERSION 2** → já está pronto. **Pule para B.0.4.**
- Se der "requer elevação"/erro → faça **B.0.2** (precisa de admin).

### B.0.2 — Instalar o WSL2 + Ubuntu (PowerShell como **Administrador**)
Botão direito no PowerShell → **"Executar como administrador"** (título da janela
tem que dizer "Administrador"), então **digite** (não cole com caracteres estranhos):
```powershell
wsl --install -d Ubuntu
```
Reinicie se pedir. Abra **"Ubuntu"** no menu Iniciar e crie usuário/senha Linux.

### B.0.3 — Driver NVIDIA do Windows
A GPU só aparece no WSL com **driver NVIDIA do Windows** recente. **NÃO** instale
"CUDA toolkit" no Windows nem driver dentro do Ubuntu. Se a B.0.4 não ver a GPU,
atualize em https://www.nvidia.com/download/index.aspx (RTX A2000).

### B.0.4 — Confirmar que o WSL enxerga a GPU (terminal Ubuntu)
```bash
nvidia-smi
```
Tem que mostrar **"NVIDIA RTX A2000"**. Se falhar: veja B.0.3, depois no
PowerShell `wsl --shutdown` e reabra o Ubuntu.

## B.1 Montar o projeto na GPU (terminal Ubuntu — bloco INTEIRO)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone https://github.com/Dovlask/NS-PINNN.git
cd NS-PINNN
git checkout EP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pip install -U "jax[cuda12]" torch pytest
python -c "import jax; print('JAX devices:', jax.devices())"
cd examples/prolate_spheroid
```
**A penúltima linha TEM que mostrar `JAX devices: [CudaDevice(id=0)]`.**
Se mostrar CPU, veja o **Troubleshooting**.

> Reabrindo o Ubuntu noutro dia: `cd ~/NS-PINNN && source .venv/bin/activate && cd examples/prolate_spheroid`

## B.2 Testes (GATE)
```bash
python -m pytest tests/ -q
```

## B.3 Smoke run (~2 min)
```bash
python main.py --config=configs/plain.py --config.run_name=smoke --config.training.max_steps=2000 --config.logging.log_every_steps=100 --config.saving.save_every_steps=2000
python main.py --config=configs/plain.py --config.run_name=smoke --config.mode=eval
```
> **OOM (a A2000 tem pouca VRAM)?**
> ```bash
> python main.py --config=configs/plain.py --config.run_name=smoke --config.training.max_steps=2000 --config.training.batch_size_per_device=1024 --config.sampling.n_surface=2000 --config.sampling.n_collocation=10000
> ```

## B.4 Treino real + convergência (em ordem)
```bash
python main.py --config=configs/pilot_sigma0.py
python main.py --config=configs/pilot_sigma0.py --config.mode=eval
python main.py --config=configs/ablation_A.py
python main.py --config=configs/ablation_A.py --config.mode=eval
python main.py --config=configs/plain.py
python main.py --config=configs/plain.py --config.mode=eval
python main.py --config=configs/default.py
python main.py --config=configs/default.py --config.mode=eval
```
Progressão: (pilot σ=0) física + dados perfeitos → (ablation_A) física pura →
(plain) física + dados ruidosos → (default) técnicas avançadas.

### Convergiu? abra `results/<run>/metrics.json`, bloco `"pass"`:

| métrica | alvo (σ>0) | piloto σ=0 |
|---|---|---|
| `rel_L2_field` | **≤ 0.02** | idem (espere bem menor) |
| `RMSE_cp_surface` | **≤ 0.0291** (σ) | **≤ 1e−2** |
| `sigma_hat_holdout` | **∈ [0.0146, 0.0437]** | **≤ 1e−2** (≈ 0) |
| `rms_pde/bc1/bc2_residual` | **≤ 1e−2** | idem |

Figuras em `results/<run>/figures/`: `cp_eta.png`, `loss_curves.png`,
`error_map.png`, `residual_hist.png`. `ablation_D` (só dados) **deve falhar no
campo** — é o esperado.

## B.5 Campanha completa (só na GPU — NÃO em CPU) — ≥ 5 sementes
```bash
for cfg in ablation_A ablation_B ablation_C ablation_D plain default no_fourier_feature no_rwf no_grad_norm ntk sota; do
  for s in 42 7 13 21 100; do
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s}
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s} --config.mode=eval
  done
done
```
Depois preencha `REPORT.md`. Claim honesto: **A ≈ C (os dados não degradam)**.

> **L-BFGS** OFF por padrão; só ligue (`--config.training.use_lbfgs=True`) depois
> que o Adam convergir.

---

## Troubleshooting
1. **`jax.devices()` mostra CPU mas você quer GPU:** no Ubuntu rode `nvidia-smi`.
   Falhou → driver (B.0.3) + `wsl --shutdown`. Funciona mas JAX vê CPU →
   `pip uninstall -y jax jaxlib && pip install -U "jax[cuda12]"`.
2. Ambiente ativo? Tem que aparecer `(.venv)` no início da linha.
3. CUDA do driver: `nvidia-smi` mostra "CUDA Version". Se for 11.x →
   `pip install -U "jax[cuda11]"`.
4. **Erro instalando `jaxlib` no Windows (Plano A):** garanta Python 64-bit 3.10–3.12
   e `pip` atualizado (`python -m pip install -U pip`); então
   `pip install --upgrade jax jaxlib`. Se persistir, me mande a mensagem de erro.
5. **`pip install -e .` falhar:** rode-o na raiz `NS-PINNN` (onde está o `setup.py`)
   e os treinos dentro de `examples/prolate_spheroid`.
