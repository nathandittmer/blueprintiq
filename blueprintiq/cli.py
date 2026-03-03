# blueprintiq/cli.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


app = typer.Typer(help="BlueprintIQ CLI")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@app.command()
def run(config: str = typer.Option("blueprintiq/config/default.yaml", help="Path to YAML config")) -> None:
    cfg_path = Path(config)
    if not cfg_path.exists():
        raise typer.BadParameter(f"Config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)
    seed = int(cfg["project"]["seed"])
    run_root = Path(cfg["project"].get("run_dir", "runs"))
    run_id = now_stamp()
    run_dir = run_root / run_id

    ensure_dir(run_dir)

    # Save config snapshot
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Basic run metadata
    meta = {
        "run_id": run_id,
        "utc_timestamp": run_id,
        "seed": seed,
        "cwd": os.getcwd(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Determinism
    set_seed(seed)

    # MLflow logging (local)
    if cfg.get("mlflow", {}).get("enabled", False) and mlflow is not None:
        tracking_uri = cfg["mlflow"].get("tracking_uri", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(cfg["mlflow"].get("experiment_name", "blueprintiq-dev"))

        with mlflow.start_run(run_name=f"spine-{run_id}"):
            # log a few key params (don’t spam everything yet)
            mlflow.log_param("seed", seed)
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("image_size", cfg.get("data_gen", {}).get("image_size"))
            mlflow.log_param("n_samples", cfg.get("data_gen", {}).get("n_samples"))

            # log config file as artifact
            mlflow.log_artifact(str(run_dir / "config.yaml"))
            mlflow.log_artifact(str(run_dir / "meta.json"))

    typer.echo(f"✅ Run initialized: {run_dir}")


if __name__ == "__main__":
    app()