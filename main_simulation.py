"""
FL Fraud Detection Benchmark — Main Simulation Entry Point.

Usage:
    # Run a single experiment
    python main_simulation.py --config configs/stage1_natural.yaml --algorithm fedavg

    # Run all algorithms for a stage
    python main_simulation.py --config configs/stage1_natural.yaml --run-all-algorithms

    # Run full benchmark (all stages × all algorithms)
    python main_simulation.py --run-all

    # Smoke test (2 rounds, small data)
    python main_simulation.py --config configs/base.yaml --algorithm fedavg --num-rounds 2 --smoke-test
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_prep.client_splitters import (
    split_stage1_natural,
    split_stage2_balanced,
    split_stage3_dirichlet,
)
from src.fl_core.server import run_simulation, run_full_benchmark
from src.utils.logging_utils import (
    init_logger,
    log_final_results,
    save_results_csv,
    finish_logging,
)


# ── Config loading ──────────────────────────────────────────────────────────

def load_config(base_path: str, override_path: str | None = None) -> dict:
    """Load base config and optionally merge a stage override."""
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path, "r") as f:
            override = yaml.safe_load(f)
        config = _deep_merge(config, override)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Data splitting ──────────────────────────────────────────────────────────

def prepare_data(config: dict, data_root: str | Path) -> list[dict]:
    """Prepare client data based on the stage configuration."""
    stage_cfg = config.get("stage", {})
    fl_cfg = config.get("fl", {})
    split_strategy = stage_cfg.get("split_strategy", "natural")
    seed = config.get("seed", 42)
    max_samples = fl_cfg.get("max_samples_per_client", None)

    print(f"\nPreparing data: strategy='{split_strategy}'")
    if max_samples:
        print(f"  Max samples per client: {max_samples:,}")

    if split_strategy == "natural":
        return split_stage1_natural(
            data_root, seed=seed,
            max_samples_per_client=max_samples,
        )

    elif split_strategy == "balanced":
        target_rate = stage_cfg.get("target_fraud_rate", 0.01)
        return split_stage2_balanced(
            data_root, target_fraud_rate=target_rate, seed=seed,
            max_samples_per_client=max_samples,
        )

    elif split_strategy == "dirichlet":
        source = stage_cfg.get("source_dataset", "ieee_cis")
        n_clients = stage_cfg.get("num_clients", 3)
        alpha = stage_cfg.get("alpha", 0.5)
        return split_stage3_dirichlet(
            data_root, source_dataset=source,
            num_clients=n_clients, alpha=alpha, seed=seed,
        )

    else:
        raise ValueError(f"Unknown split strategy: '{split_strategy}'")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FL Fraud Detection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to stage config YAML (merged with base.yaml)",
    )
    parser.add_argument(
        "--algorithm", type=str, default="fedavg",
        choices=["fedavg", "fedprox", "scaffold", "ditto"],
        help="FL algorithm to run",
    )
    parser.add_argument(
        "--run-all-algorithms", action="store_true",
        help="Run all 4 algorithms for the given stage config",
    )
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run full benchmark: all stages × all algorithms",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=None,
        help="Override number of FL rounds",
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output CSV results",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test with 2 rounds and reduced data",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent
    data_root = project_root / args.data_root
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = project_root / "configs"
    base_config_path = configs_dir / "base.yaml"

    if args.run_all:
        _run_full_benchmark(base_config_path, configs_dir, data_root, output_dir, args)
    elif args.run_all_algorithms:
        config = load_config(str(base_config_path), args.config)
        _apply_overrides(config, args)
        _run_stage(config, data_root, output_dir, algorithms=None)
    else:
        config = load_config(str(base_config_path), args.config)
        _apply_overrides(config, args)
        _run_stage(config, data_root, output_dir, algorithms=[args.algorithm])


def _apply_overrides(config: dict, args) -> None:
    """Apply CLI overrides to config."""
    if args.num_rounds is not None:
        config.setdefault("fl", {})["num_rounds"] = args.num_rounds
    if args.smoke_test:
        config.setdefault("fl", {})["num_rounds"] = 2
        config.setdefault("training", {})["local_epochs"] = 1


def _run_stage(
    config: dict,
    data_root: Path,
    output_dir: Path,
    algorithms: list[str] | None = None,
) -> list[dict]:
    """Run one stage with specified algorithms."""
    # Set seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Init logging
    logging_cfg = config.get("logging", {})
    init_logger(config, use_wandb=logging_cfg.get("use_wandb", False))

    # Prepare data
    client_data = prepare_data(config, data_root)

    # Run
    stage_name = config.get("stage", {}).get("name", "default")

    if algorithms is None:
        algorithms = ["fedavg", "fedprox", "scaffold", "ditto"]

    results = run_full_benchmark(config, client_data, algorithms=algorithms)

    # Add stage info
    for row in results:
        row["stage"] = stage_name

    # Log & save
    log_final_results(results)
    csv_path = output_dir / f"results_{stage_name}.csv"
    save_results_csv(results, csv_path)

    finish_logging()
    return results


def _run_full_benchmark(
    base_config_path: Path,
    configs_dir: Path,
    data_root: Path,
    output_dir: Path,
    args,
) -> None:
    """Run all stages × all algorithms."""
    stage_configs = [
        configs_dir / "stage1_natural.yaml",
        configs_dir / "stage2_balanced.yaml",
        configs_dir / "stage3_dirichlet.yaml",
    ]

    all_results = []

    for stage_path in stage_configs:
        if not stage_path.exists():
            print(f"⚠ Stage config not found: {stage_path}, skipping.")
            continue

        config = load_config(str(base_config_path), str(stage_path))
        _apply_overrides(config, args)

        results = _run_stage(config, data_root, output_dir)
        all_results.extend(results)

    # Save combined results
    if all_results:
        save_results_csv(all_results, output_dir / "results_full_benchmark.csv")
        print(f"\n✓ Full benchmark complete. Results: {output_dir / 'results_full_benchmark.csv'}")


if __name__ == "__main__":
    main()
