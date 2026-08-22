"""
Flower client for federated fraud detection.

Implements a unified FraudClient(NumPyClient) that adapts its local
training loop based on the active FL algorithm:

    FedAvg    → Standard BCE loss
    FedProx   → BCE + proximal term μ/2 · ||θ - θ_global||²
    SCAFFOLD  → Gradient correction: grad += (server_c - client_c)
    Ditto     → Two-model training (global + personalized)
"""

import copy
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl

from ..models.tabular_mlp import TabularMLP, build_model
from ..data_prep.datasets import FraudDataset, create_dataloader
from ..utils.metrics import compute_all_metrics


class FraudClient(fl.client.NumPyClient):
    """
    Unified Flower NumPyClient for all four FL algorithms.

    Parameters
    ----------
    client_id : str
        Unique client identifier.
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Test/validation data loader.
    model : TabularMLP
        Model instance.
    config : dict
        Experiment configuration.
    algorithm : str
        One of 'fedavg', 'fedprox', 'scaffold', 'ditto'.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: TabularMLP,
        config: dict,
        algorithm: str = "fedavg",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.config = config
        self.algorithm = algorithm.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.local_epochs = train_cfg.get("local_epochs", 5)
        self.lr = train_cfg.get("learning_rate", 0.001)

        # Compute pos_weight from training data for class imbalance
        dataset = train_loader.dataset
        if isinstance(dataset, FraudDataset):
            self.pos_weight = dataset.pos_weight.to(self.device)
        else:
            self.pos_weight = torch.tensor(1.0).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # ── Algorithm-specific state ──
        # FedProx
        self.mu = config.get("fedprox", {}).get("mu", 0.1)

        # SCAFFOLD: control variates (initialized to zeros)
        self.client_c: list[np.ndarray] | None = None  # Client control variate
        self.server_c: list[np.ndarray] | None = None  # Server control variate

        # Ditto: personalized model
        self.lambda_ditto = config.get("ditto", {}).get("lambda_reg", 0.1)
        self.personal_epochs = config.get("ditto", {}).get("personal_epochs", 5)
        self.personal_model: TabularMLP | None = None
        if self.algorithm == "ditto":
            self.personal_model = copy.deepcopy(self.model)
            self.personal_model.to(self.device)

    # ── Flower interface ─────────────────────────────────────────────────

    def get_parameters(self, config: dict = {}) -> list[np.ndarray]:
        """Return current model parameters."""
        return self.model.get_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set model parameters from server."""
        self.model.set_parameters(parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict]:
        """
        Local training round.

        Returns updated parameters, number of training samples, and metrics dict.
        """
        self.set_parameters(parameters)

        # Extract server-side info from config
        if self.algorithm == "scaffold":
            # Server control variate passed via config (serialized as bytes)
            if "server_c" in config:
                self.server_c = _deserialize_cv(config["server_c"])
            if self.client_c is None:
                # Initialize from trainable params only (not state_dict)
                self.client_c = [
                    np.zeros_like(p.detach().cpu().numpy())
                    for p in self.model.parameters()
                ]
            if self.server_c is None:
                self.server_c = [
                    np.zeros_like(p.detach().cpu().numpy())
                    for p in self.model.parameters()
                ]

        # Store global params (needed for FedProx and Ditto proximal terms)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # ── Train ──
        if self.algorithm == "fedavg":
            metrics = self._train_fedavg()
        elif self.algorithm == "fedprox":
            metrics = self._train_fedprox(global_params)
        elif self.algorithm == "scaffold":
            metrics = self._train_scaffold(parameters)
        elif self.algorithm == "ditto":
            metrics = self._train_ditto(global_params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Get updated parameters
        updated_params = self.model.get_parameters()
        n_samples = len(self.train_loader.dataset)

        return updated_params, n_samples, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, int, dict]:
        """
        Evaluate the model on local test data.

        For Ditto, evaluates the personalized model instead.
        """
        self.set_parameters(parameters)

        eval_model = self.model
        if self.algorithm == "ditto" and self.personal_model is not None:
            eval_model = self.personal_model

        eval_model.eval()
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                logits = eval_model(features)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.extend(probs)
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)

        metrics = compute_all_metrics(y_true, y_scores)

        # Loss for Flower's aggregation
        loss = float(-metrics["auc_pr"])  # Negative so "lower is better"

        return loss, len(self.test_loader.dataset), metrics

    # ── Training loops ───────────────────────────────────────────────────

    def _train_fedavg(self) -> dict:
        """Standard FedAvg training: BCE loss only."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.local_epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def _train_fedprox(self, global_params: list[torch.Tensor]) -> dict:
        """FedProx: BCE + proximal term μ/2 · ||θ - θ_global||²."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.local_epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, labels)

                # Proximal term
                proximal = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    proximal += torch.sum((local_p - global_p) ** 2)
                loss += (self.mu / 2.0) * proximal

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def _train_scaffold(self, global_params_np: list[np.ndarray]) -> dict:
        """
        SCAFFOLD: correct local gradients with control variates.

        Updates:
            y ← y - η (∇f(y) - c_i + c)
            c_i_new = c_i - c + (1/Kη)(x - y)
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        n_batches = 0
        total_steps = 0

        # Snapshot trainable params before training (for control variate update)
        params_before = [
            p.clone().detach() for p in self.model.parameters()
        ]

        for epoch in range(self.local_epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                loss.backward()

                # Apply SCAFFOLD correction to gradients (trainable params only)
                for param, c_i, s_c in zip(
                    self.model.parameters(), self.client_c, self.server_c
                ):
                    if param.grad is not None:
                        correction = torch.tensor(
                            s_c - c_i, dtype=param.grad.dtype, device=self.device
                        )
                        param.grad.data.add_(correction)

                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                total_steps += 1

        # Update client control variate using trainable params
        K_eta = total_steps * self.lr

        new_client_c = []
        delta_c = []
        for c_i, s_c, x, y in zip(
            self.client_c, self.server_c,
            params_before,  # x = params before training
            list(self.model.parameters()),  # y = params after training
        ):
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            c_i_new = c_i - s_c + (x_np - y_np) / K_eta
            new_client_c.append(c_i_new)
            delta_c.append(c_i_new - c_i)

        self.client_c = new_client_c

        # Send delta_c back to server via metrics (pickle bytes)
        metrics = {
            "train_loss": total_loss / max(n_batches, 1),
            "delta_c": _serialize_cv(delta_c),
        }
        return metrics

    def _train_ditto(self, global_params: list[torch.Tensor]) -> dict:
        """
        Ditto: train global model normally (FedAvg), then update
        personalized model with proximal regularization.
        """
        # Step 1: Train global model (standard FedAvg)
        global_metrics = self._train_fedavg()

        # Step 2: Update personalized model
        # Copy current global params as the reference
        global_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.personal_model.train()
        optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.lr)
        total_loss = 0.0
        n_batches = 0

        global_param_list = [
            p.clone().detach().to(self.device)
            for p in self.model.parameters()
        ]

        for epoch in range(self.personal_epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.personal_model(features)
                loss = self.criterion(logits, labels)

                # Proximal term: λ/2 · ||v_k - θ_global||²
                proximal = 0.0
                for v_p, g_p in zip(self.personal_model.parameters(), global_param_list):
                    proximal += torch.sum((v_p - g_p) ** 2)
                loss += (self.lambda_ditto / 2.0) * proximal

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return {
            "train_loss": global_metrics["train_loss"],
            "personal_train_loss": total_loss / max(n_batches, 1),
        }


# ── Serialization helpers for SCAFFOLD control variates ──────────────────────

def _serialize_cv(arrays: list[np.ndarray]) -> bytes:
    """Serialize list of numpy arrays to pickle bytes (compact + fast)."""
    return pickle.dumps([a.astype(np.float32) for a in arrays])


def _deserialize_cv(data: bytes) -> list[np.ndarray]:
    """Deserialize pickle bytes back to list of numpy arrays."""
    return pickle.loads(data)
