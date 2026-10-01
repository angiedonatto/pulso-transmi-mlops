from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import candidate_j_regime as cj


START = pd.Timestamp("2026-08-03T05:00:00Z")  # Monday 00:00 Bogota
STATIONS = ["03000", "05100", "07111", "09000"]


def daily_profile(times: pd.DatetimeIndex) -> np.ndarray:
    local = times.tz_convert(cj.BOGOTA_TZ)
    hour = np.asarray(local.hour + local.minute / 60.0)
    return 100 + 400 * np.exp(-((hour - 7.0) ** 2) / 2.0) + 300 * np.exp(-((hour - 18.0) ** 2) / 2.0)


def make_wide(days: int = 40, noise: float = 0.03, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(START, periods=days * 96, freq="15min")
    data = {s: daily_profile(times) * (1 + noise * rng.standard_normal(len(times))) for s in STATIONS}
    return pd.DataFrame(data, index=times)


THRESHOLDS = cj.RegimeThresholds.uniform(STATIONS, level24_abs_max=0.15, dispersion24_max=0.15, jump_abs_max=0.4, hour_anomaly_abs_max=0.4)


def targets_for(cutoff: pd.Timestamp, base: float = 200.0, factor: float = 1.1) -> pd.DataFrame:
    return pd.DataFrame(
        [{"station_id": s, "target_at": cutoff + pd.Timedelta(minutes=h), "horizon_minutes": h, "base_prediction": base, "g2_factor": factor}
         for s in STATIONS for h in (15, 30, 45, 60)]
    )


def regimes_frame(regime: str, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame({"station_id": STATIONS, "regime": regime, "recent_profile_eligible": False})
    for key, value in overrides.items():
        frame[key] = value
    return frame


USEFUL = pd.DataFrame({"station_id": STATIONS, "g2_useful": True, "g2_effect_pp": 1.0})


def test_signals_are_causal() -> None:
    wide = make_wide()
    anomaly = cj.hourly_anomaly(wide, cj.seasonal_baseline(wide))
    cutoff = START + pd.Timedelta(days=35, hours=10)
    before = cj.signals_at(cutoff, wide, anomaly)
    mutated = wide.copy()
    mutated.loc[mutated.index > cutoff] *= 7.0
    after = cj.signals_at(cutoff, mutated, cj.hourly_anomaly(mutated, cj.seasonal_baseline(mutated)))
    pd.testing.assert_frame_equal(before, after)


def test_calibrated_thresholds_are_the_stable_extremes() -> None:
    wide = make_wide()
    anomaly = cj.hourly_anomaly(wide, cj.seasonal_baseline(wide))
    start, end = START + pd.Timedelta(days=30), START + pd.Timedelta(days=32)
    th = cj.calibrate_thresholds(wide, anomaly, start, end)
    stats = pd.concat([cj.signals_at(c, wide, anomaly) for c in pd.date_range(start, end, freq="h")])
    for station, group in stats.groupby("station_id"):
        assert th.get(station, "jump_abs_max") == pytest.approx(group["jump"].max())
        assert th.get(station, "level24_abs_max") == pytest.approx(group["level24"].abs().max())
    assert th.get("99999", "jump_abs_max") == pytest.approx(stats["jump"].max())  # unknown station -> max
    assert th.samples > 0


def test_shock_needs_most_stations_to_jump() -> None:
    base = pd.DataFrame({"station_id": STATIONS, "jump": [1.0, 1.0, 1.0, 0.1], "level24": 0.0, "level24_prev": 0.0, "dispersion24": 0.0, "dispersion24_prev": 0.0, "day_consistency": 0.9})
    assert (cj.classify_regimes(base, THRESHOLDS)["regime"] == cj.SEVERE_SHOCK).all()  # 3/4 = 75%
    base.loc[2, "jump"] = 0.1
    assert (cj.classify_regimes(base, THRESHOLDS)["regime"] == cj.NORMAL).all()  # 2/4 = 50%
    # a shock declared within the last 3 cutoffs is kept (minimum regime length of the drift analysis)
    assert (cj.classify_regimes(base, THRESHOLDS, previous_shares=[0.5, 0.8])["regime"] == cj.SEVERE_SHOCK).all()
    assert (cj.classify_regimes(base, THRESHOLDS, previous_shares=[0.5, 0.5, 0.9])["regime"] == cj.NORMAL).all()


def test_sustained_drift_and_recent_profile_eligibility() -> None:
    signals = pd.DataFrame({
        "station_id": STATIONS, "jump": 0.0,
        "level24": [0.9, 0.9, 0.9, 0.0], "level24_prev": [0.9, 0.0, 0.9, 0.0],
        "dispersion24": [0.4, 0.4, 0.4, 0.0], "dispersion24_prev": [0.4, 0.0, 0.4, 0.0],
        "day_consistency": [0.9, 0.9, 0.5, 0.9],
    })
    out = cj.classify_regimes(signals, THRESHOLDS).set_index("station_id")
    jumped = cj.classify_regimes(signals.assign(jump=[0.9, 0.0, 0.0, 0.0]), THRESHOLDS).set_index("station_id")
    assert not jumped.loc["03000", "recent_profile_eligible"]  # last hour jumped away from yesterday
    assert out.loc["03000", "recent_profile_eligible"]  # drift >= 24h, repeatable
    assert not out.loc["05100", "recent_profile_eligible"]  # drift younger than 24h
    assert not out.loc["07111", "recent_profile_eligible"]  # profile not repeatable
    assert out.loc["09000", "regime"] == cj.NORMAL


def test_normal_regime_reproduces_candidate_g() -> None:
    wide = make_wide()
    cutoff = START + pd.Timedelta(days=35, hours=10)
    out = cj.compose_predictions(targets_for(cutoff), regimes_frame(cj.NORMAL), USEFUL, THRESHOLDS, wide=wide, cutoff=cutoff)
    assert (out["path"] == cj.PATH_G).all()
    np.testing.assert_allclose(out["j_prediction"], out["g_prediction"])


def test_shock_uses_latest_observation_and_ignores_future_rows() -> None:
    wide = make_wide()
    cutoff = START + pd.Timedelta(days=35, hours=10)
    out = cj.compose_predictions(targets_for(cutoff), regimes_frame(cj.SEVERE_SHOCK), USEFUL, THRESHOLDS, wide=wide, cutoff=cutoff)
    assert (out["path"] == cj.PATH_LATEST).all()
    for station in STATIONS:
        assert (out.loc[out["station_id"] == station, "j_prediction"] == wide.at[cutoff, station]).all()
    future = wide.copy()
    future.loc[future.index > cutoff] = 1e9
    again = cj.compose_predictions(targets_for(cutoff), regimes_frame(cj.SEVERE_SHOCK), USEFUL, THRESHOLDS, wide=future, cutoff=cutoff)
    np.testing.assert_allclose(out["j_prediction"], again["j_prediction"])


def test_sustained_drift_drops_g2_where_it_did_not_help() -> None:
    wide = make_wide()
    cutoff = START + pd.Timedelta(days=35, hours=10)
    useless = USEFUL.assign(g2_useful=[False, True, True, True])
    out = cj.compose_predictions(targets_for(cutoff), regimes_frame(cj.SUSTAINED_DRIFT), useless, THRESHOLDS, wide=wide, cutoff=cutoff)
    first = out[out["station_id"] == "03000"]
    assert (first["path"] == cj.PATH_G_BASE_ONLY).all()
    np.testing.assert_allclose(first["j_prediction"], first["base_prediction"])
    assert (out[out["station_id"] != "03000"]["path"] == cj.PATH_G).all()


def test_recent_profile_uses_yesterday_scaled_by_the_last_hour() -> None:
    wide = make_wide(noise=0.0)
    cutoff = START + pd.Timedelta(days=35, hours=10)
    wide.loc[wide.index > cutoff - pd.Timedelta(hours=30)] *= 2.0  # new level that started 30h ago
    regimes = regimes_frame(cj.SUSTAINED_DRIFT, recent_profile_eligible=True)
    out = cj.compose_predictions(targets_for(cutoff), regimes, USEFUL.assign(rp_useful=True), THRESHOLDS, wide=wide, cutoff=cutoff)
    assert (out["path"] == cj.PATH_RECENT_PROFILE).all()
    row = out.iloc[0]
    yday = wide.at[row["target_at"] - pd.Timedelta(hours=24), row["station_id"]]
    assert row["j_prediction"] == pytest.approx(yday, rel=1e-6)  # both days on the new level -> ratio ~1


def test_guard_blocks_an_extreme_factor_on_a_base_that_already_reacted() -> None:
    wide = make_wide()
    cutoff = START + pd.Timedelta(days=35, hours=10)
    expected = cj.seasonal_expectation(wide, cutoff + pd.Timedelta(minutes=15), "03000")
    reacted = targets_for(cutoff, base=expected * 4.0, factor=cj.G2_CLIP[1])
    out = cj.compose_predictions(reacted, regimes_frame(cj.NORMAL), USEFUL, THRESHOLDS, wide=wide, cutoff=cutoff)
    assert out["guard_fired"].all()
    np.testing.assert_allclose(out["j_prediction"], out["base_prediction"])
    calm = targets_for(cutoff, base=expected, factor=cj.G2_CLIP[1])  # base did not react -> factor kept
    out = cj.compose_predictions(calm, regimes_frame(cj.NORMAL), USEFUL, THRESHOLDS, wide=wide, cutoff=cutoff)
    assert not out["guard_fired"].any()


def test_g2_usefulness_compares_g_with_its_own_base() -> None:
    cutoff = pd.Timestamp("2026-09-10T12:00:00Z")
    times = pd.date_range(cutoff - pd.Timedelta(hours=4), cutoff, freq="15min")
    history = pd.DataFrame({"station_id": "03000", "target_at": times, "actual": 100.0, "base_prediction": 200.0, "g_prediction": 110.0})
    history = pd.concat([history, history.assign(station_id="05100", base_prediction=150.0, g_prediction=300.0)])  # base 50% vs G 0%
    out = cj.g2_usefulness(history, cutoff).set_index("station_id")
    assert out.loc["03000", "g2_useful"] and out.loc["03000", "g2_effect_pp"] == pytest.approx(90.0)
    assert not out.loc["05100", "g2_useful"]
    short = cj.g2_usefulness(history.tail(3), cutoff)
    assert short["g2_useful"].all()  # too few rows -> keep G behaviour


def test_seasonal_expectation_only_looks_back_whole_weeks() -> None:
    wide = make_wide()
    when = START + pd.Timedelta(days=35, hours=3)
    values = [wide.at[when - pd.Timedelta(days=7 * k), "03000"] for k in range(1, 5)]
    assert cj.seasonal_expectation(wide, when, "03000") == pytest.approx(float(np.median(values)))
    assert math.isnan(cj.seasonal_expectation(wide, START + pd.Timedelta(days=10), "03000"))


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({"delta_stable": -1.0, "delta_phase1": 0, "delta_phase2": 5, "delta_shock": 30, "worst_stable_station_delta": 0, "shock_hours": 24, "recovery_observed": True}, "reject"),
        ({"delta_stable": 0.0, "delta_phase1": 0, "delta_phase2": -1, "delta_shock": 1, "worst_stable_station_delta": 0, "shock_hours": 24, "recovery_observed": True}, "reject"),
        ({"delta_stable": 0.0, "delta_phase1": 0, "delta_phase2": 1, "delta_shock": 20, "worst_stable_station_delta": 0, "shock_hours": 5, "recovery_observed": False}, "continue_shadow"),
        ({"delta_stable": 0.0, "delta_phase1": 0, "delta_phase2": 1, "delta_shock": 20, "worst_stable_station_delta": -0.2, "shock_hours": 24, "recovery_observed": True}, "promotion_candidate"),
    ],
)
def test_pre_registered_decision(inputs: dict, expected: str) -> None:
    assert cj.decide(inputs)["decision"] == expected


def test_v1_1_uses_recent_profile_only_where_it_beat_g() -> None:
    wide = make_wide(noise=0.0)
    cutoff = START + pd.Timedelta(days=35, hours=10)
    regimes = regimes_frame(cj.SUSTAINED_DRIFT, recent_profile_eligible=True)
    gate = USEFUL.assign(rp_useful=[True, False, False, False])
    v1_1 = cj.compose_predictions(targets_for(cutoff), regimes, gate, THRESHOLDS, wide=wide, cutoff=cutoff, config=cj.CANDIDATE_J_V1_1)
    assert (v1_1[v1_1["station_id"] == "03000"]["path"] == cj.PATH_RECENT_PROFILE).all()
    assert (v1_1[v1_1["station_id"] != "03000"]["path"] == cj.PATH_G).all()
    v1 = cj.compose_predictions(targets_for(cutoff), regimes, gate, THRESHOLDS, wide=wide, cutoff=cutoff, config=cj.CANDIDATE_J_V1)
    assert (v1["path"] == cj.PATH_RECENT_PROFILE).all()  # v1 ignored G's own performance


def test_recent_profile_usefulness_needs_to_beat_both_g_and_base() -> None:
    cutoff = pd.Timestamp("2026-09-10T12:00:00Z")
    times = pd.date_range(cutoff - pd.Timedelta(hours=4), cutoff, freq="15min")
    history = pd.DataFrame({"station_id": "03000", "target_at": times, "actual": 100.0, "base_prediction": 130.0, "g_prediction": 110.0, "rp_prediction": 105.0})
    history = pd.concat([history, history.assign(station_id="05100", rp_prediction=112.0)])
    out = cj.g2_usefulness(history, cutoff).set_index("station_id")
    assert out.loc["03000", "rp_useful"] and out.loc["03000", "rp_effect_pp"] == pytest.approx(5.0)
    assert not out.loc["05100", "rp_useful"]
    assert not cj.g2_usefulness(history.drop(columns="rp_prediction"), cutoff)["rp_useful"].any()


def test_post_hoc_variant_is_capped_at_continue_shadow() -> None:
    passing = {"delta_stable": 0.0, "delta_phase1": 0, "delta_phase2": 1, "delta_shock": 20, "worst_stable_station_delta": 0, "shock_hours": 24, "recovery_observed": True}
    assert cj.decide(passing, post_hoc=True)["decision"] == "continue_shadow"



def test_local_helpers_match_the_drift_analysis_used_by_the_replay() -> None:
    dr = pytest.importorskip("drift_regime")
    raw = make_wide(days=36, noise=0.05).stack().rename("demand").reset_index().rename(columns={"level_0": "observed_at", "level_1": "station_id"})
    ours = cj.observation_matrix(raw)
    theirs = dr.demand_matrix(dr.prepare_observations(raw))
    pd.testing.assert_frame_equal(ours, theirs, check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(cj.seasonal_baseline(ours), dr.seasonal_baseline(theirs, weeks=4), check_names=False, check_freq=False)
    assert cj.HOURLY_OFFSET == dr.HOURLY_OFFSET and cj.WEEK_STEPS == dr.WEEK_STEPS
