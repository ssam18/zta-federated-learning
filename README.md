# Zero-Trust Federated Learning for IIoT Intrusion Detection

A privacy-preserving, Byzantine-resilient federated learning framework for
Industrial IoT (IIoT) intrusion detection. The system integrates hardware-rooted
device attestation, SHAP-weighted robust aggregation, and on-device adversarial
training into a three-tier edge–fog–cloud architecture.

---

## Overview

Traditional centralised intrusion detection systems cannot scale to IIoT
deployments spanning thousands of heterogeneous edge devices across multiple
industrial sites.  Federated learning allows devices to collaboratively train a
shared intrusion-detection model without sharing raw traffic data.  However,
standard FL is vulnerable to:

- **Byzantine poisoning** compromised devices injecting malicious model updates
- **Adversarial evasion** crafted inputs that fool the trained classifier
- **Sybil attacks** unauthenticated agents impersonating legitimate devices

This project addresses all three threats through a unified zero-trust design.

---

## System Architecture

```mermaid
flowchart TB
    subgraph CLOUD["☁️  Cloud Layer"]
        direction TB
        GA["Global Aggregator\nθ⁽ᵗ⁺¹⁾ = Σ wᵢ · θᵢ⁽ᵗ⁾"]
    end

    subgraph FOG["🌫️  Fog Layer"]
        direction LR
        FN1["Fog Node 1\n──────────\n• TPM Verification\n• SHAP Stability Score\n• Byzantine Filter\n• FedAvg / FLTrust / FLAME"]
        FN2["Fog Node 2\n──────────\n• TPM Verification\n• SHAP Stability Score\n• Byzantine Filter\n• Krum / Trimmed Mean"]
    end

    subgraph EDGE["🏭  Edge Layer  (IIoT Devices)"]
        direction LR
        E1["Agent 1\n─────────\nCNN-LSTM IDS\nAdv. Training\nTPM Attestation"]
        E2["Agent 2\n─────────\nCNN-LSTM IDS\nAdv. Training\nTPM Attestation"]
        E3["Agent 3\n─────────\nCNN-LSTM IDS\nAdv. Training\nTPM Attestation"]
        EN["Agent N\n─────────\nCNN-LSTM IDS\nAdv. Training\nTPM Attestation"]
    end

    subgraph ATTACK["⚠️  Threat Model"]
        direction TB
        BYZ["Byzantine Agents\n(Label Flip / Grad Scale)"]
        ADV["Adversarial Inputs\n(FGSM / PGD-7 / PGD-20)"]
        SYB["Sybil / Replay\nAttackers"]
    end

    %% Data flow — upward
    E1 -->|"Signed\nmodel update"| FN1
    E2 -->|"Signed\nmodel update"| FN1
    E3 -->|"Signed\nmodel update"| FN2
    EN -->|"Signed\nmodel update"| FN2

    FN1 -->|"Aggregated\nfog update"| GA
    FN2 -->|"Aggregated\nfog update"| GA

    %% Global model — downward
    GA -->|"Global model θ⁽ᵗ⁺¹⁾"| FN1
    GA -->|"Global model θ⁽ᵗ⁺¹⁾"| FN2
    FN1 -->|"Updated model"| E1
    FN1 -->|"Updated model"| E2
    FN2 -->|"Updated model"| E3
    FN2 -->|"Updated model"| EN

    %% Threats
    BYZ -. "Poisoned update" .-> FN1
    ADV -. "Evasion attempt" .-> E2
    SYB -. "Fake token" .-> FN2

    %% Styles
    classDef cloudStyle  fill:#1a6496,stroke:#0d3d5c,color:#fff,rx:8
    classDef fogStyle    fill:#2e7d32,stroke:#1b5e20,color:#fff,rx:6
    classDef edgeStyle   fill:#4a4a8a,stroke:#2c2c6a,color:#fff,rx:5
    classDef threatStyle fill:#b71c1c,stroke:#7f0000,color:#fff,rx:5,stroke-dasharray:4 4

    class GA cloudStyle
    class FN1,FN2 fogStyle
    class E1,E2,E3,EN edgeStyle
    class BYZ,ADV,SYB threatStyle
```

### Key Components

| Component | Description |
|-----------|-------------|
| **CNN-LSTM Classifier** | 1-D CNN feature extractor + stacked BiLSTM (~487 K params, INT8 quantisable) |
| **TPM Attestation** | RSA-2048 token signing with PCR measurements; FAR < 10⁻⁷ |
| **SHAP Aggregation** | RBF-kernel stability scores down-weight divergent client updates |
| **Adversarial Training** | On-device FGSM/PGD augmentation (70 % clean / 30 % adversarial per batch) |
| **Byzantine Defence** | FLTrust cosine-similarity filtering; FLAME norm-clipping + outlier rejection |
| **Agentic Layer** | TrustDB (paper §V.A) + 5-module Edge Agent (§IV) + Fog pipeline (§V.B): attestation → SHAP filter at μ−2σ → weighted FedAvg → rollback |

### Agentic Architecture (paper §IV–V)

The framework implements every clause of the paper's "agentic" specification.
Each Edge Agent is a 5-module IIoT device, each Fog Agent runs the
attestation→SHAP-filter→weighted-aggregation→rollback pipeline, and the
TrustDB tracks per-agent trust scores `τ ∈ [0,1]` with the paper's exact
update rules. Live in [src/agentic/](src/agentic/):

| File | Paper section | Responsibility |
|------|--------------|----------------|
| [trust_db.py](src/agentic/trust_db.py) | §V.A | `τ ← min(1, τ+0.02)` on positive round; `τ ← τ × 0.5` on penalty; quarantine when `τ < 0.6`; **5 consecutive clean attestations** before rejoining |
| [edge_agent.py](src/agentic/edge_agent.py) | §IV | 5 named modules: `perception`, `local_ids`, `adv_training`, `attestation`, `secure_comm` |
| [fog_agent.py](src/agentic/fog_agent.py) | §V.B | Verify TPM tokens → compute `s_i = 1 − ‖φ_i − φ_ref‖₂ / (‖φ_ref‖₂+ε)` → filter `s_i < μ_s − 2σ_s` → aggregate with `w_i ∝ s_i · acc_i · √|D_i|` → rollback if `acc < 0.8 × acc_prev` |
| [config.py](src/agentic/config.py) | §V config | Single source of truth for every paper constant (`τ_min`, `Δt_max`, etc.) |
| [observability.py](src/agentic/observability.py) | n/a | Structured logger + JSONL audit trail + JSONL metrics stream for production observability |
| [policies.py](src/agentic/policies.py) | n/a (extension) | Pluggable decision interface: `ThresholdPolicy` (paper-equivalent), `LearnedPolicy`, `LLMPolicy` — for research follow-up beyond the paper |

---

## Project Structure

```
zta-federated-learning/
├── data/
│   ├── edge_iiotset/          # IIoT device traffic (Modbus/CoAP/MQTT) — 15 classes
│   ├── cic_ids2017/           # Network flow records from CIC testbed — 10 classes
│   └── unsw_nb15/             # Network intrusion records from UNSW cyber range — 10 classes
├── src/
│   ├── models/
│   │   └── cnn_lstm.py        # CNN-LSTM intrusion detection model + quantisation helper
│   ├── federation/
│   │   └── aggregation.py     # FedAvg, FedProx, Krum, Trimmed Mean, FLTrust, FLAME, SHAP-weighted
│   ├── security/
│   │   ├── attestation.py     # TPM device attestation & trust management
│   │   ├── adversarial.py     # FGSM / PGD attack generation, adversarial training, robustness eval
│   │   └── backdoor.py        # BadNet trigger pattern + ASR computation
│   ├── agentic/               # Agentic decision layer (paper §IV-V)
│   │   ├── trust_db.py        # TrustDB with paper-exact tau update rules
│   │   ├── edge_agent.py      # 5-module Edge Agent (perception/IDS/adv/attest/comm)
│   │   ├── fog_agent.py       # Fog: attestation -> SHAP filter -> weighted FedAvg -> rollback
│   │   ├── config.py          # AgenticConfig (single source of truth for all params)
│   │   ├── observability.py   # Structured logging + JSONL audit + metrics
│   │   ├── signals.py         # Per-client signal extraction (extension)
│   │   └── policies.py        # Pluggable Threshold/Learned/LLM policies (extension)
│   └── utils/
│       ├── data_loader.py     # Dataset loaders, non-IID partitioning, preprocessing
│       └── metrics.py         # Accuracy, macro-F1, SHAP stability score
├── experiments/
│   ├── baseline_comparison.py # ZTA-FL vs FedAvg / FedProx / Krum / Trimmed Mean / Adv-FL
│   ├── byzantine_robustness.py# Accuracy under β ∈ {0.1, 0.2, 0.3} Byzantine fraction
│   ├── adversarial_eval.py    # Robustness at ε ∈ {0.05, 0.1, 0.15, 0.2}
│   └── ablation_study.py      # Component contribution analysis
├── scripts/
│   ├── run_experiments.py        # Main experiment runner (all results in one pass)
│   ├── run_agentic_experiment.py # Agentic ZTA-FL pipeline runner (paper §IV-V)
│   ├── generate_figures.py       # Publication figures & tables from experiment_results.json
│   ├── verify_pipeline.py        # SHA256 + per-seed array verification
│   └── analyze_results.py        # Summary statistics and quick CSV export
├── notebooks/
│   └── federated_ids_analysis.ipynb  # End-to-end walkthrough notebook
├── tests/
│   ├── test_federation.py        # Unit tests — aggregation & partitioning
│   ├── test_security.py          # Unit tests — attestation & trust management
│   ├── test_agentic.py           # Property tests for the agentic layer (19 tests)
│   └── test_pipeline_integrity.py# Verifies metrics are computed not hardcoded
└── results/
    ├── experiment_results.json # Structured results (auto-generated)
    └── figures/               # All generated plots (auto-generated)
```

---

## Installation

```bash
git clone https://github.com/yourorg/zta-federated-learning.git
cd zta-federated-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU support (CUDA 12.8, recommended for RTX / A-series GPUs):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Datasets

Three publicly available network intrusion datasets are included under `data/`:

| Dataset | Classes | Raw Features | Source |
|---------|---------|--------------|--------|
| [Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot) | 15 | 61 | IIoT devices (PLC, SCADA, Smart Sensor) |
| [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | 10 | 78 | General enterprise network |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 10 | 49 | UNSW cyber range |

All three are preprocessed to a common 40-feature representation via PCA.

---

## Reproducing Results

### Step 1 - Run all experiments

```bash
source .venv/bin/activate

python3 scripts/run_experiments.py \
    --dataset all \
    --rounds  20  \
    --agents  10  \
    --seeds   2   \
    --gpu         \
    --output  results/experiment_results.json
```

> **Quick smoke test** (5 agents, 10 rounds, Edge-IIoTset only — completes in ~2 min):
> ```bash
> python3 scripts/run_experiments.py --quick --gpu
> ```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `all` | Dataset to use: `edge`, `cic`, `unsw`, or `all` |
| `--rounds` | `30` | Global FL communication rounds |
| `--agents` | `20` | Number of edge agents |
| `--seeds` | `3` | Independent runs (results are mean ± std) |
| `--gpu` | off | Use CUDA GPU if available |
| `--cpu` | off | Force CPU (overrides `--gpu`) |
| `--quick` | off | Smoke-test mode: 5 agents, 10 rounds, 1 seed, Edge only |
| `--output` | `results/experiment_results.json` | Output path for structured results |

### Step 2 - Generate publication figures and tables

```bash
python3 scripts/generate_figures.py
```

This reads `results/experiment_results.json` and writes all figures (Figures 3–7)
and comparison tables (Tables II–VI) to `results/figures/`.

### Step 3 - (Optional) Run individual experiment modules

```bash
# Baseline method comparison
python3 experiments/baseline_comparison.py --dataset edge --rounds 30 --agents 10

# Byzantine robustness under label flipping and gradient manipulation
python3 experiments/byzantine_robustness.py --dataset edge --rounds 30 --agents 20

# Adversarial robustness at multiple ε budgets
python3 experiments/adversarial_eval.py --dataset edge --rounds 20 --agents 10

# Ablation study (component contributions)
python3 experiments/ablation_study.py --dataset edge --rounds 30 --agents 10
```

### Step 3b - (Optional) Run the agentic ZTA-FL pipeline

The full agentic stack (paper §IV-V) runs as a separate experiment with its own
runner. This is the script that exercises the TrustDB, the 5-module Edge Agent,
and the Fog Agent's four-step pipeline against a Byzantine-poisoned client
population:

```bash
# Smoke test (5 agents, 6 rounds, ~3 min on RTX 4060)
python3 scripts/run_agentic_experiment.py --quick --gpu

# Moderate scale (10 agents, 20 rounds, 2 seeds, β=0.3 Byzantine, ~40 min)
python3 scripts/run_agentic_experiment.py \
    --rounds 20 --agents 10 --seeds 2 --byz-fraction 0.3 --gpu

# Paper scale (100 agents, 100 rounds, 5 seeds; needs full Edge-IIoTset)
python3 scripts/run_agentic_experiment.py --paper-scale --gpu
```

Three artefacts are produced:

| File | Purpose |
|------|---------|
| `results/agentic_results.json` | Per-config metrics + TrustDB status counts + Byzantine catch rate |
| `results/agentic_audit.jsonl` | Append-only TrustDB event log (every τ update, every state transition) |
| `results/agentic_metrics.jsonl` | Per-round operational telemetry (round timings, SHAP filter μ/σ, weights, rollback flags) |

All three are streamed in real time and reproducible. Every event references a
specific paper section in its log line so an external auditor can replay the
decision trail offline.

### Step 4 - (Optional) Interactive notebook

```bash
jupyter lab notebooks/federated_ids_analysis.ipynb
```

The notebook covers data loading, model architecture, attestation, adversarial
training, and FL convergence in a self-contained walkthrough.

### Step 5 - Run unit tests

```bash
python3 -m pytest tests/ -v
```

---

## Results Summary

Results written to `results/experiment_results.json` after Step 1.
Figures and tables are generated in `results/figures/` after Step 2.

### Clean Performance (Table II)

| Method | Edge-IIoTset | CIC-IDS2017 | UNSW-NB15 |
|--------|-------------|-------------|-----------|
| FedAvg | 94.2 ± 0.8 % | 92.8 ± 0.6 % | 91.4 ± 0.7 % |
| FedProx | 94.5 ± 0.7 % | 93.1 ± 0.5 % | 91.8 ± 0.6 % |
| Krum | 93.8 ± 1.1 % | 92.1 ± 0.9 % | 90.7 ± 1.0 % |
| Trimmed Mean | 96.1 ± 0.5 % | 94.8 ± 0.4 % | 93.5 ± 0.5 % |
| Adv-FL | 96.4 ± 0.4 % | 95.1 ± 0.3 % | 93.9 ± 0.4 % |
| **ZTA-FL (ours)** | **97.8 ± 0.3 %** | **96.4 ± 0.2 %** | **95.2 ± 0.3 %** |

### Byzantine Robustness at β = 0.3 (Table III)

| Method | Label Flip Acc | Grad Manip Acc |
|--------|---------------|----------------|
| FedAvg | 67.8 % | 61.2 % |
| Krum | 82.4 % | 78.9 % |
| Trimmed Mean | 89.4 % | 85.1 % |
| FLTrust | 91.2 % | 88.7 % |
| FLAME | 90.8 % | 87.3 % |
| **ZTA-FL (ours)** | **93.2 %** | **91.4 %** |

### Adversarial Robustness under PGD-7 at ε = 0.1 (Table IV)

| Method | Clean Acc | Adv Acc | Acc Drop |
|--------|----------|---------|----------|
| FedAvg | 94.2 % | 71.2 % | 23.0 % |
| Adv-FL | 96.4 % | 84.3 % | 12.1 % |
| **ZTA-FL (ours)** | **97.8 %** | **89.3 %** | **8.5 %** |

---

## Publication

> **Zero-Trust Agentic Federated Learning for Secure Internet of Things (IoT) Defense Systems**  
> Samaresh Kumar Singh, Joyjit Roy, Martin So  
> *2026 IEEE 2nd International Conference on Secure IoT, Assured and Trusted Computing (SATC)*, 2026  
>  
> **IEEE Xplore:**  
> [https://ieeexplore.ieee.org/document/11542411](https://ieeexplore.ieee.org/document/11542411)  
>  
> **DOI:**  
> [https://doi.org/10.1109/SATC69565.2026.11542411](https://doi.org/10.1109/SATC69565.2026.11542411)

## Preprint

> **Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems**  
> Samaresh Kumar Singh, Joyjit Roy, Martin So  
> arXiv preprint arXiv:2512.23809, 2025  
> [https://arxiv.org/abs/2512.23809](https://arxiv.org/abs/2512.23809)

---

## Citation

```bibtex
@inproceedings{singh2026ztafl,
  author    = {Singh, Samaresh Kumar and Roy, Joyjit and So, Martin},
  title     = {Zero-Trust Agentic Federated Learning for Secure Internet of Things (IoT) Defense Systems},
  booktitle = {2026 IEEE 2nd International Conference on Secure IoT, Assured and Trusted Computing (SATC)},
  year      = {2026},
  pages     = {1--10},
  doi       = {10.1109/SATC69565.2026.11542411},
  url       = {https://ieeexplore.ieee.org/document/11542411}
}
---
---

## License

MIT License, see [LICENSE](LICENSE) for details.
