# RUN_TOMORROW.md — rodar o PINN na GPU (PC do lab: Windows + RTX A2000)

Copia-e-cola, de cima pra baixo. Os datasets já vêm no clone (congelados), então
**não precisa** rodar `data_gen` no lab.

**TL;DR:** o PC é Windows com GPU NVIDIA. O JAX só usa a GPU por **Linux/WSL2**
(um Linux dentro do Windows que usa a MESMA RTX A2000). Então: instala/abre o
**WSL2 (Seção 0)** → roda os blocos no terminal do Ubuntu. Se não tiver admin pra
instalar o WSL, vá para o **Apêndice CPU** no fim.

---

## 0. Pôr o JAX na GPU via WSL2 (faça uma vez)

### 0.1 — Checar se o WSL já existe (PowerShell normal)
Abra o **PowerShell** (tecla Windows, digite "PowerShell", Enter) e rode:
```powershell
wsl -l -v
```
- Se listar uma distro **Ubuntu** com **VERSION 2** → o WSL já está pronto. **Pule para 0.4.**
- Se der erro/"não reconhecido" ou não listar nada → faça **0.2**.

### 0.2 — Instalar o WSL2 + Ubuntu (PowerShell como **Administrador**)
Feche o PowerShell, abra de novo com botão direito → **"Executar como administrador"**:
```powershell
wsl --install -d Ubuntu
```
Reinicie o PC se ele pedir. Depois abra **"Ubuntu"** no menu Iniciar e crie um
**usuário e senha** Linux (qualquer um; a senha some na tela enquanto digita —
normal). Isso te joga num terminal Linux (o "Ubuntu").

### 0.3 — Driver NVIDIA do Windows
A GPU só aparece no WSL se o **driver NVIDIA do Windows** for recente. Se o PC é de
trabalho com a A2000, provavelmente já está ok. Se na etapa 0.4 a GPU não aparecer,
atualize o driver em https://www.nvidia.com/download/index.aspx (escolha RTX A2000).
**NÃO** instale "CUDA toolkit" no Windows nem driver NVIDIA dentro do Ubuntu — o
driver do Windows já entrega a GPU pro WSL.

### 0.4 — Confirmar que o Ubuntu/WSL enxerga a GPU
No terminal **Ubuntu** (não no PowerShell), rode:
```bash
nvidia-smi
```
Tem que aparecer uma tabela com **"NVIDIA RTX A2000"**. Se aparecer, a GPU está
visível no WSL. Se der "command not found" ou "No devices", veja **0.3** (driver) e
depois, no PowerShell: `wsl --shutdown`, e reabra o Ubuntu.

---

## 1. Montar o projeto na GPU (terminal Ubuntu — cola o bloco INTEIRO)
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
Se mostrar `[CpuDevice...]`, pare e veja o **Troubleshooting** no fim — não adianta
treinar em CPU sem querer.

> Toda vez que abrir o Ubuntu de novo (outro dia), reative o ambiente:
> ```bash
> cd ~/NS-PINNN && source .venv/bin/activate && cd examples/prolate_spheroid
> ```

---

## 2. Testes (GATE — não treine antes de ficar verde)
```bash
python -m pytest tests/ -q
```
Esperado: todos verdes. Se algum falhar, **pare e me mande a saída**.

---

## 3. Smoke run (~2 min — confirma que treina ponta a ponta)
```bash
python main.py --config=configs/plain.py --config.run_name=smoke --config.training.max_steps=2000 --config.logging.log_every_steps=100 --config.saving.save_every_steps=2000
```
```bash
python main.py --config=configs/plain.py --config.run_name=smoke --config.mode=eval
```
As perdas devem **descer** e gerar `results/smoke/` com `metrics.json` e figuras.

> **OOM (a A2000 tem pouca VRAM)?** rode com lote menor:
> ```bash
> python main.py --config=configs/plain.py --config.run_name=smoke --config.training.max_steps=2000 --config.training.batch_size_per_device=1024 --config.sampling.n_surface=2000 --config.sampling.n_collocation=10000
> ```

---

## 4. Treino real + checagem de convergência (rode em ordem)
```bash
python main.py --config=configs/pilot_sigma0.py
```
```bash
python main.py --config=configs/pilot_sigma0.py --config.mode=eval
```
```bash
python main.py --config=configs/ablation_A.py
```
```bash
python main.py --config=configs/ablation_A.py --config.mode=eval
```
```bash
python main.py --config=configs/plain.py
```
```bash
python main.py --config=configs/plain.py --config.mode=eval
```
```bash
python main.py --config=configs/default.py
```
```bash
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

Figuras em `results/<run>/figures/`: `cp_eta.png` (PINN em cima da Lamb),
`loss_curves.png` (descendo), `error_map.png`, `residual_hist.png`.
`ablation_D` (só dados) **deve falhar no campo** — é o esperado.

---

## 5. Campanha completa (depois que a Seção 4 convergir) — ≥ 5 sementes
```bash
for cfg in ablation_A ablation_B ablation_C ablation_D plain default no_fourier_feature no_rwf no_grad_norm ntk sota; do
  for s in 42 7 13 21 100; do
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s}
    python main.py --config=configs/$cfg.py --config.seed=$s --config.run_name=${cfg}_s${s} --config.mode=eval
  done
done
```
Depois, preencha `REPORT.md` a partir dos `metrics.json`. Claim honesto:
**A ≈ C (os dados não degradam)** — nunca “os dados melhoraram a solução”.

> **L-BFGS** está OFF por padrão; só ligue (`--config.training.use_lbfgs=True`)
> depois que o Adam convergir.

---

## Troubleshooting — `jax.devices()` mostrou CPU em vez de GPU
1. No Ubuntu, rode `nvidia-smi`. Se **falhar** → é o driver NVIDIA do Windows
   (Seção 0.3): atualize, depois no PowerShell `wsl --shutdown` e reabra o Ubuntu.
2. Se `nvidia-smi` **funciona** mas o JAX vê CPU → reinstale o JAX de GPU no venv ativo:
   ```bash
   pip uninstall -y jax jaxlib && pip install -U "jax[cuda12]"
   python -c "import jax; print(jax.devices())"
   ```
3. Confira que o ambiente está ativo (aparece `(.venv)` no início da linha). Se não:
   `source ~/NS-PINNN/.venv/bin/activate`.
4. Versão do CUDA do driver: `nvidia-smi` mostra "CUDA Version" no topo. Se for
   **11.x** (driver antigo), troque o pacote: `pip install -U "jax[cuda11]"`.

---

## Apêndice — Fallback SEM GPU (Windows nativo, CPU)
Só se não tiver admin pra instalar o WSL. Instale **Python 3.11** (marque "Add to
PATH") e **Git for Windows**; no **PowerShell**:
```powershell
git clone https://github.com/Dovlask/NS-PINNN.git
cd NS-PINNN
git checkout EP
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
pip install jax jaxlib torch pytest
cd examples\prolate_spheroid
```
Roda em CPU (lento, não usa a A2000). Os comandos das Seções 2–4 funcionam; o laço
da Seção 5 precisa da versão PowerShell:
```powershell
foreach ($cfg in "ablation_A","ablation_B","ablation_C","ablation_D","plain","default","no_fourier_feature","no_rwf","no_grad_norm","ntk","sota") {
  foreach ($s in 42,7,13,21,100) {
    python main.py --config="configs/$cfg.py" --config.seed=$s --config.run_name="${cfg}_s$s"
    python main.py --config="configs/$cfg.py" --config.seed=$s --config.run_name="${cfg}_s$s" --config.mode=eval
  }
}
```
