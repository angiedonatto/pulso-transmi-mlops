from __future__ import annotations

import pandas as pd

from forecast_cycle_sync import CycleTarget, ForecastCycle
from scripts.candidate_f_live_temporal_replay import leakage_check, live_target_samples
from scripts.model_improvement_candidates import load_observations_from_frame


def observations() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2026-01-01T00:00:00Z", periods=20 * 96, freq="15min")
    for station_id, offset in [("05000", 100), ("07111", 300), ("09122", 500)]:
        for idx, observed_at in enumerate(times):
            rows.append({"station_id": station_id, "observed_at": observed_at, "demand": offset + idx % 96})
    return load_observations_from_frame(pd.DataFrame(rows))


def cycle(cutoff: str = "2026-01-18T00:00:00Z") -> ForecastCycle:
    targets = []
    order = 0
    for station_id in ["05000", "07111", "09122"]:
        for horizon in [15, 30, 45, 60]:
            order += 1
            target_at = pd.Timestamp(cutoff).tz_convert("UTC") + pd.Timedelta(minutes=horizon)
            targets.append(
                CycleTarget(
                    station_id=station_id,
                    target_at=target_at.isoformat().replace("+00:00", "Z"),
                    horizon_minutes=horizon,
                    target_order=order,
                    raw_target={},
                )
            )
    return ForecastCycle("cycle-test", cutoff, "", len(targets), targets, {})


def actual_lookup(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    return {(str(row.station_id), pd.Timestamp(row.observed_at)): float(row.demand) for row in frame.itertuples(index=False)}


def test_live_features_do_not_use_rows_after_cutoff() -> None:
    frame = observations()
    replay_cycle = cycle()
    cutoff = pd.Timestamp(replay_cycle.data_cutoff).tz_convert("UTC")
    before = live_target_samples(replay_cycle, frame[frame["observed_at"] <= cutoff], actual_lookup(frame))
    mutated = frame.copy()
    target_at = cutoff + pd.Timedelta(minutes=60)
    mutated.loc[(mutated["station_id"] == "05000") & (mutated["observed_at"] == target_at), "demand"] = 999999.0
    after = live_target_samples(replay_cycle, mutated[mutated["observed_at"] <= cutoff], actual_lookup(mutated))
    feature_cols = [col for col in before.columns if col not in {"actual"}]

    pd.testing.assert_frame_equal(before[feature_cols], after[feature_cols])
    assert before.loc[(before["station_id"] == "05000") & (before["horizon_minutes"] == 60), "actual"].iloc[0] != after.loc[
        (after["station_id"] == "05000") & (after["horizon_minutes"] == 60), "actual"
    ].iloc[0]


def test_live_samples_preserve_exact_target_set() -> None:
    frame = observations()
    replay_cycle = cycle()
    cutoff = pd.Timestamp(replay_cycle.data_cutoff).tz_convert("UTC")
    samples = live_target_samples(replay_cycle, frame[frame["observed_at"] <= cutoff], actual_lookup(frame))

    assert len(samples) == replay_cycle.expected_predictions
    assert samples["target_order"].tolist() == list(range(1, replay_cycle.expected_predictions + 1))
    assert samples["actual"].notna().all()


def test_leakage_check_rejects_inconsistent_target_cutoff_relation() -> None:
    predictions = pd.DataFrame(
        [
            {
                "cycle_id": "c1",
                "station_id": "05000",
                "data_cutoff": "2026-01-18T00:00:00Z",
                "target_at": "2026-01-18T00:30:00Z",
                "horizon_minutes": 15,
                "prediction": 1.0,
                "actual": 1.0,
            }
            for _ in range(48)
        ]
    )
    cycles = pd.DataFrame([{"cycle_id": "c1", "data_cutoff": "2026-01-18T00:00:00Z"}])

    assert leakage_check(cycles, predictions) == "failed"
