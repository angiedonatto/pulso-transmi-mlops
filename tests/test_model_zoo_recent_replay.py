from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_replay_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "model_zoo_recent_replay.py"
    spec = importlib.util.spec_from_file_location("model_zoo_recent_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_load_i_uses_default_artifact_uri(monkeypatch):
    replay = load_replay_module()
    calls = []

    def fake_load(uri):
        calls.append(uri)
        return {"ok": True}

    monkeypatch.setattr(replay.ci, "load_candidate_i_artifact", fake_load)

    assert replay.safe_load_i([]) == {"ok": True}
    assert calls == [replay.ci.DEFAULT_ARTIFACT_URI]


def test_accuracy_summary_local_matches_station_mean_and_global_wape():
    replay = load_replay_module()
    frame = pd.DataFrame(
        [
            {"station_id": "A", "actual": 100.0, "abs_error": 10.0},
            {"station_id": "A", "actual": 100.0, "abs_error": 20.0},
            {"station_id": "B", "actual": 50.0, "abs_error": 25.0},
        ]
    )

    summary = replay.accuracy_summary_local(frame)

    assert summary["rows"] == 3
    assert summary["global_wape"] == 55.0 / 250.0
    assert summary["global_wape_accuracy"] == 78.0
    assert summary["mean_station_accuracy"] == 67.5


def test_filter_complete_revealed_requires_targets_actuals_and_g_trace():
    replay = load_replay_module()
    cycle_ids = ["complete", "missing_actual", "missing_trace", "short_targets"]
    cycles = pd.DataFrame(
        {
            "cycle_id": cycle_ids,
            "data_cutoff": pd.date_range("2026-09-18T00:00:00Z", periods=4, freq="h"),
        }
    )
    rows = []
    observations = []
    traces = []
    for cycle_offset, cycle_id in enumerate(cycle_ids):
        limit = replay.EXPECTED_TARGETS if cycle_id != "short_targets" else replay.EXPECTED_TARGETS - 1
        for idx in range(limit):
            station_id = f"{idx:05d}"
            target_at = pd.Timestamp("2026-09-18T01:00:00Z") + pd.Timedelta(days=cycle_offset, minutes=15 * idx)
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "station_id": station_id,
                    "target_at": target_at,
                    "target_order": idx,
                }
            )
            if cycle_id != "missing_actual":
                observations.append({"station_id": station_id, "observed_at": target_at, "demand": 1.0})
            if cycle_id != "missing_trace":
                traces.append({"cycle_id": cycle_id, "row": idx})

    filtered_cycles, filtered_targets, filtered_traces = replay.filter_complete_revealed(
        cycles,
        pd.DataFrame(rows),
        pd.DataFrame(observations),
        pd.DataFrame(traces),
    )

    assert filtered_cycles["cycle_id"].tolist() == ["complete"]
    assert set(filtered_targets["cycle_id"]) == {"complete"}
    assert set(filtered_traces["cycle_id"]) == {"complete"}
