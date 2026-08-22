"""
Evaluation metrics for fraud detection under extreme class imbalance.

Primary metrics (never use plain accuracy or ROC-AUC as primary):
    - AUC-PR (Area Under the Precision-Recall Curve)
    - F1 at fixed top-k% risk thresholds

Efficiency metrics:
    - Convergence detection (stable AUC-PR over a rolling window)
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


def compute_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_scores : np.ndarray
        Predicted probabilities or decision scores.

    Returns
    -------
    float
        AUC-PR score.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_scores))


def compute_f1_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: float = 0.01,
) -> float:
    """
    Compute F1 score at a fixed top-k% risk threshold.

    Classifies the top k% of samples (by predicted score) as positive.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_scores : np.ndarray
        Predicted probabilities.
    k : float
        Fraction of top-scoring samples to classify as positive (e.g., 0.01 = top 1%).

    Returns
    -------
    float
        F1 score at the given threshold.
    """
    n = len(y_true)
    n_positive = max(1, int(n * k))

    # Get threshold: score of the (n_positive)-th highest scoring sample
    sorted_scores = np.sort(y_scores)[::-1]
    threshold = sorted_scores[min(n_positive - 1, n - 1)]

    y_pred = (y_scores >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float]:
    """
    Compute the full metrics suite.

    Returns
    -------
    dict
        Keys: auc_pr, f1_at_1pct, f1_at_5pct, precision_at_1pct, recall_at_1pct
    """
    # Convert to numpy if needed
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    auc_pr = compute_auc_pr(y_true, y_scores)
    f1_1 = compute_f1_at_k(y_true, y_scores, k=0.01)
    f1_5 = compute_f1_at_k(y_true, y_scores, k=0.05)

    # Precision and recall at 1% threshold
    n_pos = max(1, int(len(y_true) * 0.01))
    sorted_scores = np.sort(y_scores)[::-1]
    threshold = sorted_scores[min(n_pos - 1, len(y_true) - 1)]
    y_pred = (y_scores >= threshold).astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0.0))
    rec = float(recall_score(y_true, y_pred, zero_division=0.0))

    return {
        "auc_pr": auc_pr,
        "f1_at_1pct": f1_1,
        "f1_at_5pct": f1_5,
        "precision_at_1pct": prec,
        "recall_at_1pct": rec,
    }


def check_convergence(
    history: list[float],
    window: int = 10,
    threshold: float = 0.005,
) -> bool:
    """
    Check if training has converged based on AUC-PR stability.

    Convergence is defined as variance < threshold over the last
    `window` consecutive rounds.

    Parameters
    ----------
    history : list[float]
        AUC-PR values per round.
    window : int
        Number of consecutive rounds to check.
    threshold : float
        Maximum allowed variance.

    Returns
    -------
    bool
        True if converged.
    """
    if len(history) < window:
        return False

    recent = history[-window:]
    return float(np.var(recent)) < threshold
