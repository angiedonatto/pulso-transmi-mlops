"""Check that the shadow runtime reproduces the K v2.1 experiment (read-only, needs PyTorch).

Retrains the 2026-09-18 05:00Z refit with `candidate_k_runtime.FrozenKPredictor` (frozen artifact,
no cache) and predicts the 14 reference cycles (05:00Z..18:00Z) whose per-seed experiment
predictions are pinned in artifacts/shadow/candidate_k_v2_1/parity_reference.csv. Reports the
prediction differences and the leaderboard metric of both on those cycles.

Bitwise identity is only expected on the platform that ran the experiment (macOS x86, torch 2.2.2)
with the experiment snapshot; on Linux CI, or with observations revised since, small numerical
differences are expected and reported. Never writes anything (unless --write-artifact, local only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_k_runtime as ckr  # noqa: E402
import candidate_k_shadow as cks  # noqa: E402
from drift_monitor import accuracy_summary  # noqa: E402

SNAPSHOT = ROOT / "data" / "interim" / "candidate_k" / "snapshot" / "observations.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observations-csv", type=Path, default=SNAPSHOT)
    parser.add_argument("--from-supabase", action="store_true", help="Read observations from Supabase instead of the experiment snapshot.")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--write-artifact", action="store_true", help="Store the result as artifacts/shadow/candidate_k_v2_1/parity_local.json.")
    args = parser.parse_args()
    started = time.perf_counter()

    reference = pd.read_csv(cks.artifact_dir() / "parity_reference.csv", dtype={"station_id": str}, parse_dates=["data_cutoff", "target_at"])
    cutoffs = sorted(reference["data_cutoff"].unique())
    if args.from_supabase:
        if args.env_file:
            for line in args.env_file.read_text().splitlines():
                key, _, value = line.strip().partition("=")
                if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                    os.environ.setdefault(key, value)
        with cks.SupabaseCandidateKShadowStore() as store:
            observations = store.observations_until(max(cutoffs) + pd.Timedelta(hours=1))
        source = "supabase"
    else:
        observations = pd.read_csv(args.observations_csv, dtype={"station_id": str})
        source = str(args.observations_csv.relative_to(ROOT)) if args.observations_csv.is_relative_to(ROOT) else str(args.observations_csv)
    observations = cks._normalize_observations(observations)
    observations = observations[observations["observed_at"] >= pd.Timestamp(cks.HISTORY_START)]

    predictor = ckr.FrozenKPredictor(cks.load_artifact(), cks.artifact_dir(), None)
    expected = reference.groupby(["data_cutoff", "station_id", "horizon_minutes"])["prediction"].median().rename("experiment").reset_index()
    frames = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        targets = expected[expected["data_cutoff"] == cutoff][["station_id", "horizon_minutes"]].assign(target_at=lambda f: cutoff + pd.to_timedelta(f["horizon_minutes"], unit="min"))
        frame, info = predictor.predict(observations[observations["observed_at"] <= cutoff], cutoff, targets)
        frames.append(frame.assign(data_cutoff=cutoff))
    runtime = pd.concat(frames)
    joined = expected.merge(runtime[["data_cutoff", "station_id", "horizon_minutes", "target_at", "k_prediction"]], on=["data_cutoff", "station_id", "horizon_minutes"])
    actual = observations.rename(columns={"observed_at": "target_at", "demand": "actual"})
    joined = joined.merge(actual, on=["station_id", "target_at"], how="left")
    rel = (joined["k_prediction"] - joined["experiment"]).abs() / joined["experiment"].abs().clip(lower=1.0)

    def accuracy(column: str) -> float:
        scored = joined.dropna(subset=["actual"]).assign(abs_error=lambda f: (f[column] - f["actual"]).abs())
        return accuracy_summary(scored)["mean_station_accuracy"]

    result = {
        "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "observations": source,
        "refit_at": info["refit_at"],
        "cycles": len(cutoffs),
        "rows": int(len(joined)),
        "identical": bool((joined["k_prediction"] == joined["experiment"]).all()),
        "max_abs_rel_diff": float(rel.max()),
        "median_abs_rel_diff": float(rel.median()),
        "experiment_mean_station_accuracy": accuracy("experiment"),
        "runtime_mean_station_accuracy": accuracy("k_prediction"),
        "train_rows": info["train_rows"],
        "torch": __import__("torch").__version__,
        "platform": sys.platform,
        "seconds": round(time.perf_counter() - started, 1),
    }
    result["accuracy_diff_pp"] = result["runtime_mean_station_accuracy"] - result["experiment_mean_station_accuracy"]
    print(json.dumps(result, indent=2))
    if args.write_artifact:
        (cks.artifact_dir() / "parity_local.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
