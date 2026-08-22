# Federated Learning core subpackage
from .client import FraudClient
from .strategies import (
    FedAvgStrategy,
    FedProxStrategy,
    SCAFFOLDStrategy,
    DittoStrategy,
)
from .server import run_simulation
