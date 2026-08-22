"""
Configurable tabular MLP for fraud detection.

Architecture:
    Input(n_features)
      → [Linear → BatchNorm → ReLU → Dropout] × N hidden layers
      → Linear(1)

Design decisions:
    - BatchNorm for training stability across heterogeneous FL clients
    - Moderate dropout (configurable) to prevent overfitting on small partitions
    - Single sigmoid output for binary fraud/legitimate classification
    - get/set_parameters helpers for Flower integration
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class TabularMLP(nn.Module):
    """
    Multi-layer perceptron for tabular fraud detection.

    Parameters
    ----------
    input_dim : int
        Number of input features (after harmonization).
    hidden_dims : list[int]
        Sizes of hidden layers (e.g., [128, 64, 32]).
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append((f"linear_{i}", nn.Linear(prev_dim, h_dim)))
            # LayerNorm instead of BatchNorm: no running statistics to
            # corrupt during FL parameter averaging across clients.
            layers.append((f"ln_{i}", nn.LayerNorm(h_dim)))
            layers.append((f"relu_{i}", nn.ReLU()))
            layers.append((f"dropout_{i}", nn.Dropout(dropout)))
            prev_dim = h_dim

        self.hidden = nn.Sequential(OrderedDict(layers))
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch_size, 1). Apply sigmoid for probabilities.
        """
        h = self.hidden(x)
        return self.output(h).squeeze(-1)

    def get_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays (for Flower)."""
        return [
            val.cpu().numpy()
            for _, val in self.state_dict().items()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load model parameters from a list of NumPy arrays (from Flower)."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.load_state_dict(state_dict, strict=True)


def build_model(config: dict) -> TabularMLP:
    """
    Build a TabularMLP from a config dict.

    Expected config keys:
        model.input_dim : int
        model.hidden_dims : list[int]
        model.dropout : float
    """
    model_cfg = config.get("model", {})
    return TabularMLP(
        input_dim=model_cfg.get("input_dim", 8),
        hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
        dropout=model_cfg.get("dropout", 0.3),
    )
