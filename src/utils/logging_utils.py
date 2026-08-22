"""
Logging utilities for experiment tracking.

Supports:
    - Weights & Biases (wandb) when available and configured
    - Console-only fallback logging
    - CSV export for paper result tables
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

# Set up module logger
logger = logging.getLogger("fl_fraud_benchmark")


# ── wandb integration (optional) ────────────────────────────────────────────

_wandb_run = None


def init_logger(
    config: dict,
    project_name: str = "fl-fraud-benchmark",
    use_wandb: bool = True,
) -> None:
    """
    Initialize experiment logging.

    Parameters
    ----------
    config : dict
        Full experiment config (logged as run config).
    project_name : str
        wandb project name.
    use_wandb : bool
        Whether to attempt wandb initialization.
    """
    global _wandb_run

    # Console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_wandb:
        try:
            import wandb
            _wandb_run = wandb.init(
                project=project_name,
                config=config,
                reinit=True,
            )
            logger.info(f"wandb initialized: {_wandb_run.url}")
        except Exception as e:
            logger.warning(f"wandb init failed ({e}), falling back to console-only logging.")
            _wandb_run = None
    else:
        logger.info("Logging to console only (wandb disabled).")


def log_round_metrics(
    round_num: int,
    client_metrics: dict[str, dict[str, float]],
    global_metrics: Optional[dict[str, float]] = None,
) -> None:
    """
    Log per-round metrics for each client and (optionally) the global model.

    Parameters
    ----------
    round_num : int
        Current FL round number.
    client_metrics : dict
        Mapping client_name → {metric_name: value}.
    global_metrics : dict | None
        Global model metrics (if applicable).
    """
    log_data = {"round": round_num}

    for client_name, metrics in client_metrics.items():
        for metric_name, value in metrics.items():
            key = f"client/{client_name}/{metric_name}"
            log_data[key] = value
            logger.info(
                f"  Round {round_num} | {client_name} | {metric_name}: {value:.4f}"
            )

    if global_metrics:
        for metric_name, value in global_metrics.items():
            key = f"global/{metric_name}"
            log_data[key] = value
            logger.info(
                f"  Round {round_num} | GLOBAL | {metric_name}: {value:.4f}"
            )

    if _wandb_run is not None:
        import wandb
        wandb.log(log_data, step=round_num)


def log_final_results(results_table: list[dict]) -> None:
    """
    Log the headline results table.

    Parameters
    ----------
    results_table : list[dict]
        List of result rows, each with keys like:
        'algorithm', 'stage', 'auc_pr', 'f1_at_1pct', 'rounds_to_converge', etc.
    """
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    if results_table:
        headers = list(results_table[0].keys())
        header_line = " | ".join(f"{h:>20s}" for h in headers)
        logger.info(header_line)
        logger.info("-" * len(header_line))

        for row in results_table:
            values = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    values.append(f"{v:>20.4f}")
                else:
                    values.append(f"{str(v):>20s}")
            logger.info(" | ".join(values))

    if _wandb_run is not None:
        import wandb
        table = wandb.Table(
            columns=list(results_table[0].keys()) if results_table else [],
            data=[list(row.values()) for row in results_table],
        )
        wandb.log({"final_results": table})


def save_results_csv(results: list[dict], path: str | Path) -> None:
    """
    Export results to a CSV file for paper tables.

    Parameters
    ----------
    results : list[dict]
        List of result rows.
    path : str | Path
        Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to save.")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {path}")


def finish_logging() -> None:
    """Finalize logging (close wandb run if active)."""
    global _wandb_run
    if _wandb_run is not None:
        import wandb
        wandb.finish()
        _wandb_run = None
