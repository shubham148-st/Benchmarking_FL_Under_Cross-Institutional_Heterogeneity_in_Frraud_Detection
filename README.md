# Multi-Domain Cross-Silo FL Benchmark for Federated Fraud Detection

Benchmarking federated learning algorithms under real-world cross-institutional heterogeneity in fraud detection.

## Quick Start

```bash
# 1. Create conda environment
conda create -n FLFD python==3.10
conda activate FLFD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets as specified below and place in data/ folder

# 4. Run a single experiment
python main_simulation.py --config configs/stage1_natural.yaml --algorithm fedavg

# 5. Run full benchmark (all stages × all algorithms)
python main_simulation.py --run-all
```

## Datasets

| Dataset | Source | Fraud Rate | Role |
|---------|--------|-----------|------|
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) | Kaggle Competition | ~3.5% | Client 0 (Institution 1) |
| [Feedzai BAF](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | NeurIPS 2022 | ~1.1% | Client 1 (Institution 2) |
| [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) | Synthetic Mobile Money | ~0.13% | Client 2 (Institution 3) |

Download all three and place CSVs in the respective `data/` subdirectories.

## 3-Stage Ablation Framework

| Stage | Description | Goal |
|-------|-------------|------|
| **Stage 1** | Natural heterogeneity (real fraud rates) | Baseline under combined domain shift + class imbalance |
| **Stage 2** | Base-rate balanced (all clients → 1% fraud) | Isolate feature/domain heterogeneity from class imbalance |
| **Stage 3** | Synthetic Dirichlet (IEEE-CIS split into 3 clients, α=0.5) | Compare real federation vs. standard single-dataset FL setup |

## FL Algorithms

| Algorithm | Key Mechanism |
|-----------|---------------|
| **FedAvg** | Weighted averaging (baseline) |
| **FedProx** | Proximal regularization (μ/2 · ‖θ − θ_global‖²) |
| **SCAFFOLD** | Control variates for gradient drift correction |
| **Ditto** | Personalized local models with global regularization |

## Evaluation Metrics

- **AUC-PR** (primary) — robust to class imbalance
- **F1 @ top 1%/5%** — precision at realistic decision thresholds
- **Rounds to convergence** — AUC-PR variance < 0.005 over 10 rounds

## Project Structure

```
fl_fraud_benchmark/
├── data/                     # Raw datasets (gitignored)
├── src/
│   ├── data_prep/            # Harmonization & splitting
│   ├── models/               # TabularMLP architecture
│   ├── fl_core/              # Flower client, strategies, server
│   └── utils/                # Metrics & logging
├── configs/                  # YAML configurations
├── main_simulation.py        # CLI entry point
└── requirements.txt          # Dependencies (Python 3.10)
```

## CLI Reference

```bash
# Single experiment
python main_simulation.py --config configs/stage1_natural.yaml --algorithm fedavg

# All algorithms for one stage
python main_simulation.py --config configs/stage2_balanced.yaml --run-all-algorithms

# Full benchmark
python main_simulation.py --run-all

# Smoke test (2 rounds)
python main_simulation.py --algorithm fedavg --smoke-test
```
