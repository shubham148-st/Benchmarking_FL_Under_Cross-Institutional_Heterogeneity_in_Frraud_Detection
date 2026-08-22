"""
Flower simulation runner.

Orchestrates federated learning experiments by:
    1. Building the strategy for the chosen algorithm
    2. Creating client factory functions with appropriate data loaders
    3. Running flwr.simulation.start_simulation()
"""

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters

from ..models.tabular_mlp import TabularMLP, build_model
from ..data_prep.datasets import FraudDataset, create_dataloader
from .strategies import create_strategy
from .client import FraudClient


def run_simulation(
    config: dict,
    client_data: list[dict],
    algorithm: str = "fedavg",
) -> fl.server.history.History:
    """
    Run a federated learning simulation.

    Parameters
    ----------
    config : dict
        Full experiment configuration.
    client_data : list[dict]
        List of client data dicts from the splitting functions.
        Each dict has keys: 'name', 'train_df', 'test_df', 'fraud_rate'.
    algorithm : str
        Algorithm name: 'fedavg', 'fedprox', 'scaffold', 'ditto'.

    Returns
    -------
    flwr.server.history.History
        Training history with per-round metrics.
    """
    fl_cfg = config.get("fl", {})
    num_rounds = fl_cfg.get("num_rounds", 50)
    num_clients = len(client_data)

    # Build initial model and get initial parameters
    initial_model = build_model(config)
    initial_params = initial_model.get_parameters()
    initial_parameters = ndarrays_to_parameters(initial_params)

    # Create strategy
    strategy = create_strategy(
        algorithm=algorithm,
        config=config,
        initial_parameters=initial_parameters,
    )

    # Build client factory
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 256)

    def client_fn(cid: str) -> FraudClient:
        """Create a FraudClient for the given client ID."""
        client_idx = int(cid)
        data = client_data[client_idx]

        # Create data loaders
        train_loader = create_dataloader(
            data["train_df"],
            batch_size=batch_size,
            shuffle=True,
            use_weighted_sampling=False,
        )
        test_loader = create_dataloader(
            data["test_df"],
            batch_size=batch_size,
            shuffle=False,
        )

        # Each client gets a fresh copy of the model architecture
        model = build_model(config)

        return FraudClient(
            client_id=data["name"],
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            config=config,
            algorithm=algorithm,
        )

    # Configure Ray resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        client_resources = {"num_cpus": 1, "num_gpus": 0.33}  # 3 clients share 1 GPU
    else:
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # Run simulation
    print(f"\n{'='*60}")
    print(f"  FL Simulation: {algorithm.upper()}")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    return history


def run_full_benchmark(
    config: dict,
    client_data: list[dict],
    algorithms: list[str] | None = None,
) -> list[dict]:
    """
    Run the full benchmark suite: all algorithms on the given client data.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    client_data : list[dict]
        Client data from splitting functions.
    algorithms : list[str] | None
        Algorithms to benchmark. Defaults to all four.

    Returns
    -------
    list[dict]
        Results table with one row per algorithm.
    """
    if algorithms is None:
        algorithms = ["fedavg", "fedprox", "scaffold", "ditto"]

    results = []

    for algo in algorithms:
        print(f"\n{'#'*60}")
        print(f"  Running: {algo.upper()}")
        print(f"{'#'*60}")

        history = run_simulation(config, client_data, algorithm=algo)

        # Extract final metrics from history
        result_row = {
            "algorithm": algo,
        }

        # Get distributed evaluation metrics if available
        if history.metrics_distributed:
            for metric_name, metric_values in history.metrics_distributed.items():
                if metric_values:
                    # Last round's value
                    _, last_value = metric_values[-1]
                    result_row[metric_name] = last_value

        # Centralized metrics
        if history.metrics_centralized:
            for metric_name, metric_values in history.metrics_centralized.items():
                if metric_values:
                    _, last_value = metric_values[-1]
                    result_row[f"centralized_{metric_name}"] = last_value

        # Loss history
        if history.losses_distributed:
            losses = [v for _, v in history.losses_distributed]
            result_row["final_loss"] = losses[-1] if losses else None
            result_row["rounds_to_converge"] = len(losses)

        results.append(result_row)
        
        # ── Save Detailed Per-Round History to CSV ──
        try:
            import pandas as pd
            history_data = []
            rounds = set()
            if history.losses_distributed:
                rounds.update([r for r, _ in history.losses_distributed])
            if history.metrics_distributed:
                for m in history.metrics_distributed.values():
                    rounds.update([r for r, _ in m])
                    
            stage_name = config.get("stage", {}).get("name", "default")
            
            for r in sorted(list(rounds)):
                row = {"round": r, "algorithm": algo, "stage": stage_name}
                if history.losses_distributed:
                    for rnd, val in history.losses_distributed:
                        if rnd == r: row["loss"] = val
                if history.metrics_distributed:
                    for metric_name, metric_values in history.metrics_distributed.items():
                        for rnd, val in metric_values:
                            if rnd == r: row[metric_name] = val
                history_data.append(row)
                
            if history_data:
                output_dir = Path("results")
                output_dir.mkdir(parents=True, exist_ok=True)
                history_csv = output_dir / f"history_{stage_name}_{algo}.csv"
                pd.DataFrame(history_data).to_csv(history_csv, index=False)
                print(f"  [History] Saved detailed per-round history to {history_csv}")
        except Exception as e:
            print(f"  [History] Could not save history CSV: {e}")

    return results
