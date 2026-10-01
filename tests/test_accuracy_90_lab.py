from __future__ import annotations

import pandas as pd

from scripts.accuracy_90_lab import official_accuracy, select_ensembles, target_audit


def tiny_frame() -> pd.DataFrame:
    rows = []
    cutoffs = [
        pd.Timestamp("2026-09-03T00:00:00Z"),
        pd.Timestamp("2026-09-03T01:00:00Z"),
        pd.Timestamp("2026-09-10T18:00:00Z"),
        pd.Timestamp("2026-09-10T19:00:00Z"),
    ]
    for cutoff in cutoffs:
        for station in ("01000", "02000"):
            for horizon in (15, 30):
                target_at = cutoff + pd.Timedelta(minutes=horizon)
                rows.append(
                    {
                        "data_cutoff": cutoff,
                        "target_at": target_at,
                        "station_id": station,
                        "horizon_minutes": horizon,
                        "actual": 100.0,
                        "pred_G": 90.0,
                        "pred_I": 95.0,
                        "pred_abl::LSg:g2|snr4+shape3|snr16": 96.0,
                        "pred_abl::L:g2": 91.0,
                        "is_official": True,
                        "max_source_at": cutoff,
                    }
                )
    return pd.DataFrame(rows)


def test_official_accuracy_is_mean_station_accuracy() -> None:
    frame = tiny_frame()

    assert official_accuracy(frame, "pred_G") == 90.0
    assert official_accuracy(frame, "pred_I") == 95.0


def test_target_audit_requires_competition_grid_and_cutoff_safety() -> None:
    audit = target_audit(tiny_frame())

    assert audit["rows_per_cycle_min"] == 4
    assert audit["rows_per_cycle_max"] == 4
    assert audit["stations_per_cycle_min"] == 2
    assert audit["target_relation_matches_horizon"] is True
    assert audit["max_source_at_lte_data_cutoff"] is True


def test_ensemble_selection_uses_selection_windows() -> None:
    frame = tiny_frame()
    frame["target_hour_local"] = 0
    frame["is_peak_hour"] = False
    frame["demand_level"] = "q1_low"

    table, selected = select_ensembles(frame)

    assert not table.empty
    assert selected
    assert table.iloc[0]["selection_objective"] >= table.iloc[-1]["selection_objective"]
