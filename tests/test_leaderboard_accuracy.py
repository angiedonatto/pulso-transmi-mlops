from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from drift_monitor import accuracy_summary, build_drift_report, leaderboard_window, performance_drift
from test_drift_monitor import CONFIG, build_tables


def load_autopilot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "retrain_autopilot.py"
    spec = importlib.util.spec_from_file_location("retrain_autopilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


autopilot = load_autopilot()


def rows(spec: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(spec, columns=["cycle_id", "station_id", "actual", "prediction"])
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
    return frame


def test_leaderboard_accuracy_is_mean_of_station_accuracies_pooled_over_the_window() -> None:
    # station A: big volume, 10% error. station B: small volume, 200% error (accuracy clipped at 0)
    frame = rows([("c1", "A", 1000.0, 900.0), ("c1", "B", 10.0, 30.0)])
    summary = accuracy_summary(frame)
    assert summary["mean_station_accuracy"] == pytest.approx((90.0 + 0.0) / 2)
    assert summary["global_wape"] == pytest.approx(120.0 / 1010.0)
    assert summary["global_wape_accuracy"] == pytest.approx(100 * (1 - 120.0 / 1010.0))
    assert summary["accuracy_at_20"] == pytest.approx(50.0)


def test_a_cycle_with_wape_above_one_is_zero_pooled_but_not_zero_by_station() -> None:
    frame = rows([("c1", "A", 100.0, 100.0), ("c1", "B", 10.0, 400.0)])
    summary = accuracy_summary(frame)
    assert summary["global_wape_accuracy"] == 0.0  # pooled WAPE = 390/110 > 1
    assert summary["mean_station_accuracy"] == pytest.approx(50.0)


def test_window_pools_each_station_instead_of_averaging_cycles() -> None:
    # cycle 1: station A perfect on low volume; cycle 2: A misses 50% on high volume
    frame = rows([("c1", "A", 10.0, 10.0), ("c2", "A", 1000.0, 500.0)])
    pooled_by_station = accuracy_summary(frame)["mean_station_accuracy"]
    mean_of_cycles = (accuracy_summary(frame[frame["cycle_id"] == "c1"])["mean_station_accuracy"] + accuracy_summary(frame[frame["cycle_id"] == "c2"])["mean_station_accuracy"]) / 2
    assert pooled_by_station == pytest.approx(100 * (1 - 500 / 1010))
    assert mean_of_cycles == pytest.approx(75.0)
    assert pooled_by_station < mean_of_cycles  # the legacy average was optimistic


def test_missing_cycles_count_as_zero_predictions() -> None:
    cutoffs = [pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:00:00Z")]
    scored = pd.DataFrame(
        {"station_id": ["A"], "actual": [100.0], "abs_error": [10.0], "data_cutoff": [cutoffs[0]]}
    )
    observations = pd.DataFrame({"station_id": ["A"], "observed_at": [cutoffs[1] + pd.Timedelta(minutes=15)], "demand": [100.0]})
    result = leaderboard_window(scored, observations, expected_cutoffs=cutoffs, stations=["A"], horizons=[15])
    assert result["coverage"] == pytest.approx(0.5)
    assert result["missing_cutoffs"] == ["2026-01-01T01:00:00Z"]
    assert result["global_wape"] == pytest.approx(110.0 / 200.0)
    assert result["mean_station_accuracy"] == pytest.approx(45.0)
    assert result["accuracy_if_missing_excluded"] == pytest.approx(90.0)


def test_performance_drift_reports_both_families_and_uses_the_leaderboard_formula() -> None:
    tables = build_tables(station_recent_errors={"03000": 95.0})
    report = build_drift_report(config=CONFIG, **tables)
    performance = report.performance
    assert performance["accuracy_metric"] == "mean_station_accuracy"
    for key in ("recent6_accuracy", "recent6_global_wape_accuracy", "recent6_mean_cycle_accuracy", "recent_global_wape_accuracy", "leaderboard_rolling_24h"):
        assert key in performance
    # 03000 at 5% accuracy, 02300 perfect -> mean of stations 52.5; pooled WAPE 95/200 -> 52.5 too, so
    # check the per-cycle record exposes both and that the trigger is the station mean
    cycle = performance["cycle_accuracy"][-1]
    assert cycle["mean_station_accuracy"] == pytest.approx(52.5)
    assert cycle["global_wape_accuracy"] == pytest.approx(52.5)
    assert performance["recent6_accuracy"] == pytest.approx(52.5)


def test_collapsed_cycle_is_visible_in_global_wape_accuracy() -> None:
    tables = build_tables(recent_cycle_errors=[0.0, 0.0, 0.0, 0.0, 0.0, -150.0])
    scored_tables = {key: tables[key] for key in ("observations", "predictions", "submission_receipts", "forecast_cycles", "forecast_cycle_targets")}
    from drift_monitor import official_scored_predictions

    scored = official_scored_predictions(**scored_tables)
    performance = performance_drift(scored, CONFIG)
    last = performance["cycle_accuracy"][-1]
    assert last["global_wape_accuracy"] == 0.0  # prediction 2.5x actual everywhere
    assert last["mean_station_accuracy"] == 0.0
    # 5 perfect cycles + 1 cycle with WAPE 1.5: pooling per station -> 1 - 150/600 = 75%,
    # while averaging cycle accuracies (legacy) -> (5*100 + 0)/6 = 83.3%
    assert performance["recent6_accuracy"] == pytest.approx(75.0)
    assert performance["recent6_global_wape_accuracy"] == pytest.approx(75.0)
    assert performance["recent6_mean_cycle_accuracy"] == pytest.approx(500 / 6)


def test_autopilot_trigger_uses_leaderboard_metric_and_exposes_both() -> None:
    performance = {"recent6_accuracy": 70.0, "recent6_global_wape_accuracy": 65.0, "recent6_mean_cycle_accuracy": 74.0, "leaderboard_rolling_24h": {"mean_station_accuracy": 70.0, "coverage": 1.0}}
    assert autopilot._trigger_accuracy(performance) == 70.0
    view = autopilot.accuracy_metrics_view(performance)
    assert view["trigger_metric"] == "mean_station_accuracy"
    assert view["recent6_global_wape_accuracy"] == 65.0
    assert view["recent6_mean_cycle_accuracy_legacy"] == 74.0


def test_autopilot_scores_challengers_on_shared_cycles_only() -> None:
    tables = build_tables(reference_error=40.0, recent_error=0.0)
    challenger_id = "22222222-2222-4222-8222-222222222222"
    preds = tables["predictions"]
    recent_cycles = sorted(preds["cycle_id"].unique())[-2:]
    shadow = preds[preds["cycle_id"].isin(recent_cycles)].copy()
    shadow["model_id"] = challenger_id
    shadow["prediction"] = shadow["prediction"] - 10.0
    tables["predictions"] = pd.concat([preds, shadow], ignore_index=True)
    tables["model_versions"] = pd.concat(
        [tables["model_versions"], pd.DataFrame([{"model_id": challenger_id, "is_active": False, "trained_until": "2026-01-06T00:00:00Z", "model_name": "challenger", "model_type": "shadow"}])],
        ignore_index=True,
    )
    scores = autopilot.score_shadow_candidates(tables, recent_cycles=6)
    assert len(scores) == 1
    score = scores[0]
    assert score.cycles == 2
    # champion is perfect on those 2 cycles (recent_error=0); its older 40%-error cycles must not leak in
    assert score.champion_recent_accuracy == pytest.approx(100.0)
    assert score.recent_accuracy == pytest.approx(90.0)
    assert score.delta_vs_champion_recent == pytest.approx(-10.0)
    assert score.recent_global_wape_accuracy == pytest.approx(90.0)
