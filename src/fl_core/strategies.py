"""
Custom Flower server strategies for the FL benchmark.

Implements:
    FedAvgStrategy    – Standard weighted averaging (baseline)
    FedProxStrategy   – FedAvg + passes μ to clients for proximal term
    SCAFFOLDStrategy  – Maintains server control variate, corrects drift
    DittoStrategy     – FedAvg server-side; clients handle personalization
"""

import json
import pickle
from typing import Optional, Union

import numpy as np
from logging import WARNING

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ── FedAvg (Baseline) ───────────────────────────────────────────────────────

def weighted_average_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    """Aggregate evaluation metrics weighted by number of samples."""
    if not metrics:
        return {}
    
    # We only average scalar values
    first_metrics = metrics[0][1]
    keys = [k for k, v in first_metrics.items() if isinstance(v, (int, float))]
    
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    aggregated = {}
    for key in keys:
        weighted_sum = sum([num_examples * float(m.get(key, 0.0)) for num_examples, m in metrics])
        aggregated[key] = weighted_sum / total_examples
        
    return aggregated


def weighted_average_fit_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    """Aggregate fit (training) metrics weighted by number of samples."""
    if not metrics:
        return {}
    
    # Only average numeric scalars (skip serialized strings like delta_c)
    first_metrics = metrics[0][1]
    keys = [k for k, v in first_metrics.items() if isinstance(v, (int, float))]
    
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    aggregated = {}
    for key in keys:
        weighted_sum = sum([num_examples * float(m.get(key, 0.0)) for num_examples, m in metrics])
        aggregated[key] = weighted_sum / total_examples
        
    return aggregated


class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg strategy.

    θ_global = Σ (n_k / N) · θ_k

    No modifications to the base FedAvg — serves as baseline.
    """

    def __init__(self, **kwargs):
        if "evaluate_metrics_aggregation_fn" not in kwargs:
            kwargs["evaluate_metrics_aggregation_fn"] = weighted_average_metrics
        super().__init__(**kwargs)


# ── FedProx ──────────────────────────────────────────────────────────────────

class FedProxStrategy(FedAvg):
    """
    FedProx strategy.

    Server-side is identical to FedAvg (weighted averaging).
    Passes μ (proximal weight) to clients so they can add the
    proximal term to their local loss function.

    Parameters
    ----------
    mu : float
        Proximal term weight. Typical values: {0.01, 0.1, 1.0}.
    """

    def __init__(self, mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure clients with the proximal μ parameter."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Inject μ into each client's config
        updated_pairs = []
        for client_proxy, fit_ins in client_config_pairs:
            config = dict(fit_ins.config)
            config["mu"] = self.mu
            updated_pairs.append((
                client_proxy,
                FitIns(fit_ins.parameters, config),
            ))

        return updated_pairs


# ── SCAFFOLD ─────────────────────────────────────────────────────────────────

class SCAFFOLDStrategy(FedAvg):
    """
    SCAFFOLD strategy with server control variate.

    Maintains a global control variate c that is updated each round
    from client-reported delta control variates (Δc_i).

    The server control variate is sent to clients via configure_fit
    so they can correct their local gradients.
    """

    def __init__(self, n_clients: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_clients = n_clients
        self.server_c: list[np.ndarray] | None = None  # Initialized lazily

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Send server control variate to clients."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Initialize server_c on first round from trainable params only
        if self.server_c is None:
            params = parameters_to_ndarrays(parameters)
            # Only use parameter tensors (weights + biases), skip
            # non-trainable entries. With LayerNorm the state_dict has
            # weight and bias for each LN layer which ARE trainable,
            # so we initialize from all params here. The client will
            # match because it also uses model.parameters().
            self.server_c = [np.zeros_like(p) for p in params]

        # Serialize server_c as pickle bytes and inject into config
        server_c_bytes = pickle.dumps(
            [c.astype(np.float32) for c in self.server_c]
        )

        updated_pairs = []
        for client_proxy, fit_ins in client_config_pairs:
            config = dict(fit_ins.config)
            config["server_c"] = server_c_bytes
            updated_pairs.append((
                client_proxy,
                FitIns(fit_ins.parameters, config),
            ))

        return updated_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregate model updates and control variate deltas.

        1. Standard weighted averaging of model parameters
        2. Update server_c: c ← c + (1/N) Σ Δc_i
        """
        if not results:
            return None, {}

        # Standard FedAvg aggregation for model parameters
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Update server control variate from client delta_c
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            if "delta_c" in metrics:
                delta_c = pickle.loads(metrics["delta_c"])

                if self.server_c is None:
                    self.server_c = [np.zeros_like(d) for d in delta_c]

                for i in range(len(self.server_c)):
                    self.server_c[i] += delta_c[i] / self.n_clients

        return aggregated_params, aggregated_metrics


# ── Ditto ────────────────────────────────────────────────────────────────────

class DittoStrategy(FedAvg):
    """
    Ditto strategy for personalized FL.

    Server-side is identical to FedAvg (only aggregates the global model).
    Passes λ_ditto to clients so they can train their personalized model
    with proximal regularization.

    Evaluation collects metrics from both global and personal models.

    Parameters
    ----------
    lambda_ditto : float
        Regularization strength for personalized model.
    """

    def __init__(self, lambda_ditto: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ditto = lambda_ditto

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Pass lambda_ditto to clients."""
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        updated_pairs = []
        for client_proxy, fit_ins in client_config_pairs:
            config = dict(fit_ins.config)
            config["lambda_ditto"] = self.lambda_ditto
            updated_pairs.append((
                client_proxy,
                FitIns(fit_ins.parameters, config),
            ))

        return updated_pairs


# ── Strategy Factory ─────────────────────────────────────────────────────────

def create_strategy(algorithm: str, config: dict, **kwargs) -> FedAvg:
    """
    Create a strategy instance based on algorithm name and config.

    Parameters
    ----------
    algorithm : str
        One of 'fedavg', 'fedprox', 'scaffold', 'ditto'.
    config : dict
        Experiment configuration.

    Returns
    -------
    Strategy instance.
    """
    fl_cfg = config.get("fl", {})
    common_kwargs = {
        "fraction_fit": fl_cfg.get("fraction_fit", 1.0),
        "fraction_evaluate": fl_cfg.get("fraction_evaluate", 1.0),
        "min_fit_clients": fl_cfg.get("num_clients", 3),
        "min_evaluate_clients": fl_cfg.get("num_clients", 3),
        "min_available_clients": fl_cfg.get("num_clients", 3),
        "evaluate_metrics_aggregation_fn": weighted_average_metrics,
        "fit_metrics_aggregation_fn": weighted_average_fit_metrics,
        **kwargs,
    }

    algorithm = algorithm.lower()

    if algorithm == "fedavg":
        return FedAvgStrategy(**common_kwargs)
    elif algorithm == "fedprox":
        mu = config.get("fedprox", {}).get("mu", 0.1)
        return FedProxStrategy(mu=mu, **common_kwargs)
    elif algorithm == "scaffold":
        n_clients = fl_cfg.get("num_clients", 3)
        return SCAFFOLDStrategy(n_clients=n_clients, **common_kwargs)
    elif algorithm == "ditto":
        lambda_ditto = config.get("ditto", {}).get("lambda_reg", 0.1)
        return DittoStrategy(lambda_ditto=lambda_ditto, **common_kwargs)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: fedavg, fedprox, scaffold, ditto"
        )
