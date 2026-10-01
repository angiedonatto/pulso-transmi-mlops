from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from scripts import candidate_h_guard as ch

BOGOTA_TZ = ch.BOGOTA_TZ


def _scored_row(
    *,
    cycle_id: int = 1,
    station_id: str = "S1",
    accepted_at: pd.Timestamp,
    data_cutoff: pd.Timestamp,
    target_observed_at: pd.Timestamp,
    actual: float,
    prediction: float,
) -> dict:
    return {
        "cycle_id": cycle_id,
        "station_id": station_id,
        "accepted_at": accepted_at,
        "data_cutoff": data_cutoff,
        "target_observed_at": target_observed_at,
        "actual": actual,
        "prediction": prediction,
    }


def _observation(station_id: str, observed_at: pd.Timestamp, demand: float) -> dict:
    return {"station_id": station_id, "observed_at": observed_at, "demand": demand}


# ---------------------------------------------------------------------------
# 1. Profiles must never look past `data_cutoff` (no leakage).
# ---------------------------------------------------------------------------


def test_profile_at_only_uses_observations_at_or_before_cutoff() -> None:
    obs = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1"],
            "quarter_hour": [10, 10, 10],
            "observed_at": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"]
            ),
            "demand": [10.0, 20.0, 1000.0],
        }
    )
    lookup = ch.build_profile_lookup(obs, ["station_id", "quarter_hour"])

    # Cutoff strictly between the 2nd and 3rd observation: the future, extreme
    # value (1000.0) must never be visible.
    value, count = ch.profile_at(lookup, ("S1", 10), pd.Timestamp("2026-01-02T12:00:00Z"))
    assert count == 2
    assert value == pytest.approx((10.0 + 20.0) / 2)
    assert value != pytest.approx((10.0 + 20.0 + 1000.0) / 3)

    # Cutoff exactly at an observation timestamp is inclusive of that row.
    value_inclusive, count_inclusive = ch.profile_at(lookup, ("S1", 10), pd.Timestamp("2026-01-01T00:00:00Z"))
    assert count_inclusive == 1
    assert value_inclusive == pytest.approx(10.0)

    # Cutoff before any observation yields "no data", never a leaked value.
    value_before, count_before = ch.profile_at(lookup, ("S1", 10), pd.Timestamp("2025-12-31T00:00:00Z"))
    assert count_before == 0
    assert math.isnan(value_before)


def test_attach_profiles_general_ignores_observations_after_data_cutoff() -> None:
    target_time = pd.Timestamp("2026-02-02 09:00:00", tz=BOGOTA_TZ)
    data_cutoff = target_time - pd.Timedelta(minutes=15)

    # Every observation below shares `target_time`'s hour:minute so it lands in the
    # same (station, quarter_hour) profile bucket as the target row.
    past_days = [target_time - pd.Timedelta(days=offset) for offset in range(1, 6)]
    future_day = target_time + pd.Timedelta(days=1)
    observations = pd.DataFrame(
        [_observation("S1", day.tz_convert("UTC"), 50.0) for day in past_days]
        + [_observation("S1", future_day.tz_convert("UTC"), 99999.0)]
    )
    scored = pd.DataFrame(
        [
            _scored_row(
                accepted_at=data_cutoff.tz_convert("UTC"),
                data_cutoff=data_cutoff.tz_convert("UTC"),
                target_observed_at=target_time.tz_convert("UTC"),
                actual=50.0,
                prediction=50.0,
            )
        ]
    )

    frame = ch.attach_profiles(scored, observations)

    assert frame.loc[0, "profile_general_count"] == 5
    assert frame.loc[0, "profile_general"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 2. DOW profile backs off to the general profile when DOW history is sparse.
# ---------------------------------------------------------------------------


def _dow_backoff_frame(dow_observation_count: int, dow_demand: float, general_demand: float) -> pd.DataFrame:
    target_time = pd.Timestamp("2026-02-05 08:00:00", tz=BOGOTA_TZ)  # Thursday
    data_cutoff = target_time - pd.Timedelta(minutes=15)

    # All timestamps share `target_time`'s hour:minute so they land in the same
    # quarter_hour bucket; multiples of 7 days also share its day-of-week.
    dow_obs = [
        _observation(
            "S1",
            (target_time - pd.Timedelta(days=7 * (n + 1))).tz_convert("UTC"),
            dow_demand,
        )
        for n in range(dow_observation_count)
    ]
    general_obs = [
        _observation(
            "S1",
            (target_time - pd.Timedelta(days=offset)).tz_convert("UTC"),
            general_demand,
        )
        for offset in range(1, 16)
        if offset % 7 != 0  # keep these off the target day-of-week
    ]
    observations = pd.DataFrame(dow_obs + general_obs)
    scored = pd.DataFrame(
        [
            _scored_row(
                accepted_at=data_cutoff.tz_convert("UTC"),
                data_cutoff=data_cutoff.tz_convert("UTC"),
                target_observed_at=target_time.tz_convert("UTC"),
                actual=general_demand,
                prediction=general_demand,
            )
        ]
    )
    return ch.attach_profiles(scored, observations)


def test_dow_profile_backs_off_to_general_when_history_is_sparse() -> None:
    frame = _dow_backoff_frame(dow_observation_count=ch.MIN_DOW_HISTORY - 2, dow_demand=500.0, general_demand=100.0)

    assert frame.loc[0, "profile_dow_backoff"] == True  # noqa: E712
    assert frame.loc[0, "profile_dow_count"] < ch.MIN_DOW_HISTORY
    assert frame.loc[0, "profile_dow"] == pytest.approx(frame.loc[0, "profile_general"])


def test_dow_profile_uses_its_own_mean_once_history_is_sufficient() -> None:
    frame = _dow_backoff_frame(dow_observation_count=ch.MIN_DOW_HISTORY + 1, dow_demand=500.0, general_demand=100.0)

    assert frame.loc[0, "profile_dow_backoff"] == False  # noqa: E712
    assert frame.loc[0, "profile_dow_count"] >= ch.MIN_DOW_HISTORY
    assert frame.loc[0, "profile_dow"] == pytest.approx(500.0)
    assert frame.loc[0, "profile_dow"] != pytest.approx(frame.loc[0, "profile_general"])


# ---------------------------------------------------------------------------
# 3. Weight selection must only ever look at strictly earlier blocks.
# ---------------------------------------------------------------------------


def _metrics_row(block_index: int, delta: float, *, station_id: str = "05000", variant: str = "general", weight: float = 0.2) -> dict:
    return {
        "block_index": block_index,
        "station_id": station_id,
        "profile_variant": variant,
        "profile_weight": weight,
        "delta_vs_b_pp": delta,
    }


def test_guard_decision_ignores_metrics_from_the_current_or_future_blocks() -> None:
    good_history = [_metrics_row(i, delta) for i, delta in enumerate([0.6, 0.7, 0.8])]
    metrics_without_future = pd.DataFrame(good_history)
    metrics_with_bad_future = pd.DataFrame(
        good_history + [_metrics_row(3, -999.0)]  # same block_index as `current_block` below
    )

    decision_without_future = ch.choose_guard_decision(
        metrics_without_future, station_id="05000", current_block=3, current_targets=100, config=ch.DEFAULT_RULE
    )
    decision_with_bad_future = ch.choose_guard_decision(
        metrics_with_bad_future, station_id="05000", current_block=3, current_targets=100, config=ch.DEFAULT_RULE
    )

    assert decision_without_future == decision_with_bad_future
    assert decision_with_bad_future.guard_enabled is True
    assert decision_with_bad_future.profile_weight == pytest.approx(0.2)


def test_guard_decision_for_a_block_cannot_see_its_own_block_metrics() -> None:
    # Only 2 strictly-prior blocks (0, 1) are available for block_index=2, which is
    # below DEFAULT_RULE.min_prior_windows (3), even though block 2 itself looks great.
    metrics = pd.DataFrame(
        [_metrics_row(0, 0.8), _metrics_row(1, 0.9), _metrics_row(2, 5.0)]
    )

    decision = ch.choose_guard_decision(metrics, station_id="05000", current_block=2, current_targets=100, config=ch.DEFAULT_RULE)

    assert decision.guard_enabled is False
    assert decision.reason == "insufficient prior windows"


# ---------------------------------------------------------------------------
# 4. The guard must stay disabled after any severe regression, even with good
#    recent windows.
# ---------------------------------------------------------------------------


def test_guard_not_enabled_after_a_severe_regression_in_prior_history() -> None:
    deltas = [-2.0, 0.8, 0.9, 1.0, 1.1]  # block 0 breaches severe_regression_pp (-1.00)
    metrics = pd.DataFrame([_metrics_row(i, delta) for i, delta in enumerate(deltas)])

    decision = ch.choose_guard_decision(metrics, station_id="05000", current_block=5, current_targets=100, config=ch.DEFAULT_RULE)

    assert decision.guard_enabled is False
    assert decision.reason == "no candidate passed conservative gates"


def test_guard_enabled_when_no_prior_window_breaches_the_severe_regression_floor() -> None:
    deltas = [-0.5, 0.8, 0.9, 1.0, 1.1]  # worst prior delta stays above -1.00
    metrics = pd.DataFrame([_metrics_row(i, delta) for i, delta in enumerate(deltas)])

    decision = ch.choose_guard_decision(metrics, station_id="05000", current_block=5, current_targets=100, config=ch.DEFAULT_RULE)

    assert decision.guard_enabled is True


# ---------------------------------------------------------------------------
# 5. Guard requires at least `min_prior_windows` of history before the
#    evaluation window -- and requires enough current targets to trust the block.
# ---------------------------------------------------------------------------


def test_guard_disabled_when_training_history_is_shorter_than_the_evaluation_window() -> None:
    config = replace(ch.DEFAULT_RULE, min_prior_windows=4)
    metrics = pd.DataFrame([_metrics_row(i, 2.0) for i in range(3)])  # only 3 prior blocks, need 4

    decision = ch.choose_guard_decision(metrics, station_id="05000", current_block=3, current_targets=100, config=config)

    assert decision.guard_enabled is False
    assert decision.reason == "insufficient prior windows"


def test_guard_disabled_when_current_block_has_too_few_targets() -> None:
    metrics = pd.DataFrame([_metrics_row(i, 2.0) for i in range(5)])

    decision = ch.choose_guard_decision(
        metrics, station_id="05000", current_block=5, current_targets=ch.DEFAULT_RULE.min_targets - 1, config=ch.DEFAULT_RULE
    )

    assert decision.guard_enabled is False
    assert decision.reason == "insufficient current targets"
