from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "model_improvement_candidates.py"
SPEC = importlib.util.spec_from_file_location("model_improvement_candidates", MODULE_PATH)
assert SPEC is not None
model_improvement_candidates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["model_improvement_candidates"] = model_improvement_candidates
SPEC.loader.exec_module(model_improvement_candidates)


def test_official_metrics_average_station_accuracy() -> None:
    frame = pd.DataFrame(
        [
            {"station_id": "03000", "actual": 100.0, "prediction": 90.0, "horizon_minutes": 15, "target_hour": 8, "target_day_of_week": 0},
            {"station_id": "03000", "actual": 100.0, "prediction": 90.0, "horizon_minutes": 30, "target_hour": 8, "target_day_of_week": 0},
            {"station_id": "02300", "actual": 100.0, "prediction": 50.0, "horizon_minutes": 15, "target_hour": 8, "target_day_of_week": 0},
        ]
    )

    metrics = model_improvement_candidates.official_metrics(frame)

    assert metrics["mean_station_accuracy"] == 70.0
    station_metrics = metrics["station_metrics"].set_index("station_id")
    assert station_metrics.loc["03000", "station_accuracy"] == 90.0
    assert station_metrics.loc["02300", "station_accuracy"] == 50.0


def test_make_samples_masks_lags_after_data_cutoff() -> None:
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=120, freq="15min")
    observations = pd.DataFrame(
        {
            "station_id": ["03000"] * len(observed_at),
            "observed_at": observed_at,
            "demand": [float(value) for value in range(len(observed_at))],
        }
    )
    observations = model_improvement_candidates.load_observations_from_frame(observations)
    cutoff = observed_at[100]

    samples = model_improvement_candidates.make_samples(observations, [cutoff])

    horizon_15 = samples[samples["horizon_minutes"] == 15].iloc[0]
    horizon_60 = samples[samples["horizon_minutes"] == 60].iloc[0]
    assert horizon_15["lag_15"] == 100.0
    assert horizon_15["lag_30"] == 99.0
    assert horizon_60["lag_60"] == 100.0
    assert pd.isna(horizon_60["lag_15"])
    assert pd.isna(horizon_60["lag_30"])
    assert pd.isna(horizon_60["lag_45"])
