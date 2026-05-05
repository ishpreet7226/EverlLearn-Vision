"""
EverLearn Vision – Retrieve Best Model from MLflow
=====================================================
Query the local MLflow tracking store to find the run with the
highest validation accuracy and download its model artifact.

Usage:
    python retrieve_best_model.py
    python retrieve_best_model.py --output-dir best_model

How it works:
    1. Connects to the local MLflow tracking store (./mlruns)
    2. Finds the "EverLearn-Vision" experiment
    3. Searches all runs, sorted by best_val_acc descending
    4. Prints the winning run's parameters and metrics
    5. Downloads the model.pth artifact to a local directory
"""

import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main(output_dir: str) -> None:
    client = MlflowClient()

    # ── Find the experiment ───────────────────────────────────────────────────
    experiment = client.get_experiment_by_name("EverLearn-Vision")
    if experiment is None:
        print("❌  No 'EverLearn-Vision' experiment found.")
        print("    Run `python train.py` first to create an MLflow experiment.")
        sys.exit(1)

    experiment_id = experiment.experiment_id
    print(f"📋  Experiment: {experiment.name} (id={experiment_id})")

    # ── Search for the best run ───────────────────────────────────────────────
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.best_val_acc DESC"],
        max_results=1,
    )

    if not runs:
        print("❌  No runs found in the experiment.")
        print("    Run `python train.py` first to log a training run.")
        sys.exit(1)

    best_run = runs[0]
    run_id = best_run.info.run_id
    run_name = best_run.info.run_name or "unnamed"

    print(f"\n🏆  Best run: {run_name} (id={run_id})")
    print(f"    Status: {best_run.info.status}")
    print()

    # ── Print parameters ──────────────────────────────────────────────────────
    print("    Parameters:")
    for key, value in sorted(best_run.data.params.items()):
        print(f"      {key:20s} = {value}")

    # ── Print metrics ─────────────────────────────────────────────────────────
    print()
    print("    Metrics:")
    for key, value in sorted(best_run.data.metrics.items()):
        if isinstance(value, float):
            print(f"      {key:20s} = {value:.4f}")
        else:
            print(f"      {key:20s} = {value}")

    # ── Download artifacts ────────────────────────────────────────────────────
    print()
    os.makedirs(output_dir, exist_ok=True)

    artifacts = client.list_artifacts(run_id)
    if not artifacts:
        print("⚠️  No artifacts found for this run.")
        return

    print(f"📦  Downloading artifacts to '{output_dir}/':")
    for artifact in artifacts:
        print(f"      → {artifact.path}")

    local_path = client.download_artifacts(run_id, "", dst_path=output_dir)
    print(f"\n✅  Artifacts saved to: {local_path}")
    print(f"    Best val accuracy: {best_run.data.metrics.get('best_val_acc', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve the best model from MLflow tracking"
    )
    parser.add_argument(
        "--output-dir",
        default="best_model",
        help="Directory to download the model artifact to (default: best_model)",
    )
    args = parser.parse_args()
    main(args.output_dir)
