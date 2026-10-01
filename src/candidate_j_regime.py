"""Candidate J: regime-aware wrapper around Candidate G (shadow only).

J never retrains anything new and introduces no tuned parameter. It reuses Candidate G's two
pieces (the Candidate F base and the G2 EWMA factor) and decides, per cycle and station and using
only data <= data_cutoff, which of four paths to use. Every rule below is tied to a quantitative
finding of `reports/drift_regime_analysis/` (see EVIDENCE).

    regime          path                  prediction
    NORMAL          g                     base_F * guarded G2 factor        (= Candidate G)
    SUSTAINED_DRIFT g / g_base_only       base_F * G2 if G2 has helped this station in the last 24h, else base_F
                    recent_profile        y(t - 24h) * r   when the new regime is >= 24h old and repeats day to day
    SEVERE_SHOCK    latest_observation    y(data_cutoff)   (no base_F, no G2, no lag_96/weekly)

Compounding guard (every path that applies G2): a factor at the G2 clip is not applied when the
base has already moved beyond the station's stable range in the same direction.

Thresholds are each station's own extremes in a stable calibration window that ends before every
replayed cycle (no look-ahead), mirroring the per-station null ranges of the drift analysis; they
are not values tuned on the drift.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


# Self-contained (no dependency on the drift-analysis module); values and helpers are identical to
# drift_regime's, which the replay used (parity checked in tests/test_candidate_j_regime.py).
BOGOTA_TZ = "America/Bogota"
SLOT = pd.Timedelta(minutes=15)
DAY_STEPS = 96
WEEK_STEPS = 7 * DAY_STEPS
HOURLY_OFFSET = 20.0  # 4 slots x 5: additive stabiliser of the hourly log-ratios


NORMAL = "NORMAL"
SUSTAINED_DRIFT = "SUSTAINED_DRIFT"
SEVERE_SHOCK = "SEVERE_SHOCK"
PATH_G = "g"
PATH_G_BASE_ONLY = "g_base_only"
PATH_RECENT_PROFILE = "recent_profile"
PATH_LATEST = "latest_observation"
G2_CLIP = (0.35, 2.5)  # candidate_g_adaptive.CLIP_LOW / CLIP_HIGH (unchanged)

# Quantitative findings each rule depends on (reports/drift_regime_analysis, snapshot 2026-09-18 10:00Z).
EVIDENCE: dict[str, str] = {
    "normal_keep_g": "G reaches 83.3% (leaderboard formula) in regime_1 and 90.6-92.2% on the stations still stable in regime_2",
    "g2_useful_sustained": "G2 adds +4.0 pp in regime_2 (corr(factor, needed) = 0.74), up to +38.5 pp at 03000 and +26 pp at 05100",
    "g2_heterogeneous": "G2 effect by station in regime_2 ranges from -2.8 pp (07111) to +38.5 pp (03000)",
    "g2_fails_shock": "G2 effect -9.6 pp in regime_3; corr(factor, needed) ~ 0; 30% of rows at the 2.5 clip vs 0.2% in regime_2",
    "double_adaptation": "0% cycle (cutoff 2026-09-18 01:00): base 1,531 already reacted, x2.5 factor -> 3,827 vs actual 450-1,317 at 07111",
    "lag96_day1_day2": "lag_96 naive accuracy on day 1 of regime_2: 67.6% (0-42% at 02300/03000/05000/05100); day 2: 86.8% (ref 83.6-84.3%)",
    "new_profile_repeats": "day-to-day profile correlation in regime_2: 0.82-0.98 (02300, 03000, 05000, 07111, 09122); 0.72 at 05100",
    "shock_short_lags": "regime_3 naive accuracy: lag_15 76.6%, lag_30 59.9%, lag_60 32.1%, lag_96 29.4%, weekly 23.0% vs G 26.0% and base F 38.5%",
    "shock_is_global": "regime_3 hits 12/12 stations; the analysis calls a change global when >= 75% of stations are affected",
    "min_regime_length": "the drift analysis segments the global series with a 3 h minimum regime length; regime_3 has lasted >= 5 h",
    "per_station_null": "the drift analysis compares every station with its own stable range (per-station null windows)",
    "keep_where_g_works": "the drift analysis recommends 'keep current behavior' where G already absorbs a sustained shift (6 stations at 86.8-92.2% in regime_2) "
                          "and a recent-profile refit only where it does not",
    "v1_replay_diagnosis": "post-hoc: in the J v1 replay the recent-profile path cost -4.9 pp at 07107 and -5.2 pp at 06000 in phase 1 (stations where G worked) "
                           "while gaining +4.3 pp overall on its rows in phase 2",
}


THRESHOLD_KEYS = ("level24_abs_max", "dispersion24_max", "jump_abs_max", "hour_anomaly_abs_max")


@dataclass(frozen=True)
class RegimeThresholds:
    """Per-station extremes of each causal statistic over a stable calibration window (hourly steps).
    EVIDENCE["per_station_null"]. Unknown stations fall back to the max over stations."""

    per_station: dict[str, dict[str, float]]
    calibration_start: str
    calibration_end: str
    samples: int

    def get(self, station_id: str, key: str) -> float:
        values = self.per_station.get(str(station_id))
        if values is None:
            return max(v[key] for v in self.per_station.values())
        return float(values[key])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def uniform(cls, stations: Iterable[str], **values: float) -> "RegimeThresholds":
        return cls(per_station={str(s): {k: float(values[k]) for k in THRESHOLD_KEYS} for s in stations}, calibration_start="", calibration_end="", samples=0)


@dataclass(frozen=True)
class CandidateJConfig:
    global_shock_share: float = 0.75  # EVIDENCE["shock_is_global"]
    min_day_consistency: float = 0.8  # EVIDENCE["new_profile_repeats"] (same threshold as the drift analysis)
    min_regime_hours_for_recent_profile: int = 24  # EVIDENCE["lag96_day1_day2"]
    g2_usefulness_window_hours: int = 24  # one daily cycle, as the drift-analysis windows
    min_g2_usefulness_rows: int = 8  # = candidate_g_adaptive.FEEDBACK_WINDOW; fewer rows -> keep G behaviour
    shock_memory_hours: int = 3  # EVIDENCE["min_regime_length"]: a declared shock lasts at least 3 h
    # v1.1: the recent-profile path must have beaten G (and base_F) on the station's resolved targets of the
    # last 24h; without that evidence G is kept (EVIDENCE["keep_where_g_works"], ["v1_replay_diagnosis"])
    require_realized_recent_profile_gain: bool = True
    version: str = "candidate_j_v1_1"
    hourly_offset: float = HOURLY_OFFSET  # same stabiliser as the drift analysis log-ratios
    evidence: dict[str, str] = field(default_factory=lambda: dict(EVIDENCE))


# --------------------------------------------------------------------------------------
# Causal signals
# --------------------------------------------------------------------------------------


def observation_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    """Regular 15-min UTC grid x station of demand (missing slots stay NaN)."""
    frame = observations[["station_id", "observed_at", "demand"]].copy()
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["observed_at", "demand"]).drop_duplicates(["station_id", "observed_at"], keep="last")
    wide = frame.pivot(index="observed_at", columns="station_id", values="demand").sort_index()
    return wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="15min"))


def seasonal_baseline(wide: pd.DataFrame, *, weeks: int = 4) -> pd.DataFrame:
    """Causal median of the same slot 1..`weeks` weeks earlier; NaN unless all weeks exist."""
    stack = np.stack([wide.shift(WEEK_STEPS * w).to_numpy(dtype=float) for w in range(1, weeks + 1)])
    return pd.DataFrame(np.median(stack, axis=0), index=wide.index, columns=wide.columns)


def hourly_anomaly(wide: pd.DataFrame, baseline: pd.DataFrame, *, offset: float = HOURLY_OFFSET) -> pd.DataFrame:
    """A(t) = log((sum of the 4 slots ending at t + c) / (same for the causal baseline + c)).

    Uses y(t-45m..t) and baseline values that themselves only use t-7d..t-28d, so A(t) is known at t."""
    valid = wide.notna() & baseline.notna()
    y = wide.where(valid).rolling(4, min_periods=4).sum()
    b = baseline.where(valid).rolling(4, min_periods=4).sum()
    return np.log((y + offset) / (b + offset))


def signals_at(cutoff: pd.Timestamp, wide: pd.DataFrame, anomaly: pd.DataFrame) -> pd.DataFrame:
    """Per-station statistics at `cutoff` using only rows <= cutoff."""
    cutoff = pd.Timestamp(cutoff)
    hours = [cutoff - pd.Timedelta(hours=k) for k in range(24)]
    prev_hours = [cutoff - pd.Timedelta(hours=24 + k) for k in range(24)]
    a_last24 = anomaly.reindex(hours)
    a_prev24 = anomaly.reindex(prev_hours)
    today = wide.loc[(wide.index > cutoff - pd.Timedelta(hours=24)) & (wide.index <= cutoff)]
    yesterday = wide.loc[(wide.index > cutoff - pd.Timedelta(hours=48)) & (wide.index <= cutoff - pd.Timedelta(hours=24))]
    rows = []
    for station in wide.columns:
        a_now = _value(anomaly, cutoff, station)
        a_yday = _value(anomaly, cutoff - pd.Timedelta(hours=24), station)
        rows.append(
            {
                "station_id": str(station),
                "a_now": a_now,
                "a_yday": a_yday,
                "jump": abs(a_now - a_yday) if math.isfinite(a_now) and math.isfinite(a_yday) else math.nan,
                "level24": _mean(a_last24[station]),
                "level24_prev": _mean(a_prev24[station]),
                "dispersion24": _std(a_last24[station]),
                "dispersion24_prev": _std(a_prev24[station]),
                "day_consistency": _corr(today[station].to_numpy(dtype=float), yesterday[station].to_numpy(dtype=float)) if len(today) == len(yesterday) == DAY_STEPS else math.nan,
            }
        )
    return pd.DataFrame(rows)


def calibrate_thresholds(wide: pd.DataFrame, anomaly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> RegimeThresholds:
    """Stable range of every statistic per station: max over the hourly cutoffs in [start, end]."""
    cutoffs = pd.date_range(pd.Timestamp(start).ceil("h"), pd.Timestamp(end), freq="h")
    stats = pd.concat([signals_at(c, wide, anomaly) for c in cutoffs], ignore_index=True)
    per_station = {}
    for station_id, group in stats.groupby("station_id"):
        per_station[str(station_id)] = {
            "level24_abs_max": float(group["level24"].abs().max()),
            "dispersion24_max": float(group["dispersion24"].max()),
            "jump_abs_max": float(group["jump"].max()),
            "hour_anomaly_abs_max": float(group["a_now"].abs().max()),
        }
    return RegimeThresholds(per_station=per_station, calibration_start=_iso(cutoffs[0]), calibration_end=_iso(cutoffs[-1]), samples=int(stats["jump"].notna().sum()))


def shock_share(signals: pd.DataFrame, thresholds: RegimeThresholds) -> float:
    """Share of stations whose last hour departs from the same hour yesterday beyond their stable range."""
    limits = signals["station_id"].map(lambda s: thresholds.get(s, "jump_abs_max"))
    return float((signals["jump"] > limits).mean()) if len(signals) else 0.0


def classify_regimes(
    signals: pd.DataFrame,
    thresholds: RegimeThresholds,
    config: CandidateJConfig | None = None,
    *,
    previous_shares: Iterable[float] = (),
) -> pd.DataFrame:
    """Regime per station. SEVERE_SHOCK is global and is kept while any of the last
    `shock_memory_hours` cutoffs (this one included; `previous_shares` = earlier cutoffs' shares)
    had >= `global_shock_share` of stations jumping. SUSTAINED_DRIFT/NORMAL are per station."""
    config = config or CandidateJConfig()
    frame = signals.copy()
    limit = lambda key: frame["station_id"].map(lambda s: thresholds.get(s, key))  # noqa: E731
    frame["jumped"] = frame["jump"] > limit("jump_abs_max")
    share = shock_share(frame, thresholds)
    recent = [share] + list(previous_shares)[: max(0, config.shock_memory_hours - 1)]
    level_shift = frame["level24"].abs() > limit("level24_abs_max")
    profile_shift = frame["dispersion24"] > limit("dispersion24_max")
    prev_level = frame["level24_prev"].abs() > limit("level24_abs_max")
    prev_profile = frame["dispersion24_prev"] > limit("dispersion24_max")
    frame["level_shift"] = level_shift
    frame["profile_shift"] = profile_shift
    frame["drift_older_than_24h"] = (level_shift | profile_shift) & (prev_level | prev_profile)
    frame["shock_share"] = share
    frame["regime"] = np.where(level_shift | profile_shift, SUSTAINED_DRIFT, NORMAL)
    if max(recent) >= config.global_shock_share:
        frame["regime"] = SEVERE_SHOCK
    # yesterday's profile is only usable when the regime is >= 24h old and the last hour did not jump
    # away from yesterday (EVIDENCE["lag96_day1_day2"]); the ratio last-hour/yesterday would otherwise
    # adapt twice to the same jump
    frame["recent_profile_eligible"] = (
        (frame["regime"] == SUSTAINED_DRIFT)
        & frame["profile_shift"]
        & frame["drift_older_than_24h"]
        & ~frame["jumped"]
        & (frame["day_consistency"] >= config.min_day_consistency)
    )
    return frame


# --------------------------------------------------------------------------------------
# G2 usefulness and composition
# --------------------------------------------------------------------------------------


CANDIDATE_J_V1 = CandidateJConfig(require_realized_recent_profile_gain=False, version="candidate_j_v1")
CANDIDATE_J_V1_1 = CandidateJConfig()


def g2_usefulness(history: pd.DataFrame, cutoff: pd.Timestamp, config: CandidateJConfig | None = None) -> pd.DataFrame:
    """Realised usefulness of each adaptation per station over resolved targets in (cutoff - 24h, cutoff].

    `history` rows: station_id, target_at, actual, base_prediction, g_prediction (what G applied) and,
    optionally, rp_prediction (the recent-profile prediction that would have been made).
    - g2_useful: G (base_F x G2) at least as accurate as base_F; default True without evidence.
    - rp_useful: recent profile strictly more accurate than both G and base_F; default False without evidence.
    """
    config = config or CandidateJConfig()
    cutoff = pd.Timestamp(cutoff)
    window = history[(history["target_at"] > cutoff - pd.Timedelta(hours=config.g2_usefulness_window_hours)) & (history["target_at"] <= cutoff)]
    columns = ["station_id", "g2_rows", "g2_effect_pp", "g2_useful", "rp_effect_pp", "rp_useful"]
    rows = []
    for station_id, group in window.groupby("station_id"):
        actual = float(group["actual"].sum())
        if len(group) < config.min_g2_usefulness_rows or actual <= 0:
            rows.append({"station_id": station_id, "g2_rows": len(group), "g2_effect_pp": math.nan, "g2_useful": True, "rp_effect_pp": math.nan, "rp_useful": False})
            continue
        accuracy = lambda column: 100 * max(0.0, 1 - float((group[column] - group["actual"]).abs().sum()) / actual)  # noqa: E731
        base_acc, g_acc = accuracy("base_prediction"), accuracy("g_prediction")
        rp_effect, rp_useful = math.nan, False
        if "rp_prediction" in group and group["rp_prediction"].notna().all():
            rp_acc = accuracy("rp_prediction")
            rp_effect = rp_acc - max(g_acc, base_acc)
            rp_useful = bool(rp_effect > 0)
        rows.append({"station_id": station_id, "g2_rows": len(group), "g2_effect_pp": g_acc - base_acc, "g2_useful": bool(g_acc >= base_acc),
                     "rp_effect_pp": rp_effect, "rp_useful": rp_useful})
    return pd.DataFrame(rows, columns=columns)


def recent_profile_predictions(targets: pd.DataFrame, wide: pd.DataFrame, cutoff: pd.Timestamp, *, offset: float = HOURLY_OFFSET) -> np.ndarray:
    """y(t - 24h) x (last hour <= cutoff / same hour yesterday), per target (EVIDENCE["lag96_day1_day2"]).
    Uses only rows <= cutoff (t - 24h <= cutoff because horizons are <= 60 min)."""
    cutoff = pd.Timestamp(cutoff)
    visible = wide.loc[wide.index <= cutoff]
    last_hour = visible.loc[visible.index > cutoff - pd.Timedelta(hours=1)].sum(min_count=4)
    same_hour_yday = visible.loc[(visible.index > cutoff - pd.Timedelta(hours=25)) & (visible.index <= cutoff - pd.Timedelta(hours=24))].sum(min_count=4)
    ratio = (last_hour + offset) / (same_hour_yday + offset)
    values = []
    for station_id, target_at in zip(targets["station_id"], targets["target_at"]):
        yday = _value(visible, pd.Timestamp(target_at) - pd.Timedelta(hours=24), station_id)
        r = float(ratio.get(station_id, math.nan))
        values.append(max(0.0, yday * r) if math.isfinite(yday) and math.isfinite(r) else math.nan)
    return np.asarray(values, dtype=float)


def compose_predictions(
    targets: pd.DataFrame,
    regimes: pd.DataFrame,
    usefulness: pd.DataFrame,
    thresholds: RegimeThresholds,
    *,
    wide: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: CandidateJConfig | None = None,
) -> pd.DataFrame:
    """Candidate J predictions for one cycle.

    `targets` rows: station_id, target_at, horizon_minutes, base_prediction, g2_factor (G's factor).
    Only observations <= cutoff are read from `wide` (the seasonal expectation of a target slot uses
    t-7d..t-28d, all <= cutoff).
    """
    config = config or CandidateJConfig()
    cutoff = pd.Timestamp(cutoff)
    if (wide.index > cutoff).any():
        wide = wide.loc[wide.index <= cutoff]
    use_cols = [c for c in ["station_id", "g2_useful", "g2_effect_pp", "rp_useful", "rp_effect_pp"] if c in usefulness.columns]
    frame = targets.merge(regimes, on="station_id", how="left").merge(usefulness[use_cols], on="station_id", how="left")
    frame["g2_useful"] = frame["g2_useful"].fillna(True).astype(bool)
    frame["rp_useful"] = frame["rp_useful"].fillna(False).astype(bool) if "rp_useful" in frame else False
    frame["g_prediction"] = np.maximum(0.0, frame["base_prediction"] * frame["g2_factor"])

    # compounding guard: the cycle's mean base vs the causal baseline over the same target slots
    expected = np.array([seasonal_expectation(wide, t, s) for s, t in zip(frame["station_id"], frame["target_at"])], dtype=float)
    frame["baseline_at_target"] = expected
    reaction = frame.groupby("station_id").apply(
        lambda g: math.log((g["base_prediction"].sum() + config.hourly_offset) / (g["baseline_at_target"].sum() + config.hourly_offset))
        if g["baseline_at_target"].notna().all() else math.nan,
        include_groups=False,
    ).rename("base_reaction")
    frame = frame.merge(reaction.reset_index(), on="station_id", how="left")
    log_factor = np.log(frame["g2_factor"].clip(lower=1e-9))
    at_clip = (frame["g2_factor"] >= G2_CLIP[1] - 1e-9) | (frame["g2_factor"] <= G2_CLIP[0] + 1e-9)
    same_direction = np.sign(log_factor) == np.sign(frame["base_reaction"])
    stable_range = frame["station_id"].map(lambda s: thresholds.get(s, "hour_anomaly_abs_max"))
    frame["guard_fired"] = at_clip & same_direction & (frame["base_reaction"].abs() > stable_range)
    guarded = np.where(frame["guard_fired"], 1.0, frame["g2_factor"])

    latest = {str(s): float(wide[s].dropna().iloc[-1]) if wide[s].notna().any() else math.nan for s in wide.columns}
    frame["rp_prediction"] = recent_profile_predictions(frame, wide, cutoff, offset=config.hourly_offset)

    paths, values = [], []
    for row, factor in zip(frame.itertuples(index=False), guarded):
        if row.regime == SEVERE_SHOCK and math.isfinite(latest.get(row.station_id, math.nan)):
            paths.append(PATH_LATEST)
            values.append(latest[row.station_id])
            continue
        rp_allowed = bool(row.rp_useful) or not config.require_realized_recent_profile_gain
        if row.regime == SUSTAINED_DRIFT and bool(row.recent_profile_eligible) and rp_allowed and math.isfinite(row.rp_prediction):
            paths.append(PATH_RECENT_PROFILE)
            values.append(float(row.rp_prediction))
            continue
        if row.regime == SUSTAINED_DRIFT and not row.g2_useful:
            paths.append(PATH_G_BASE_ONLY)
            values.append(max(0.0, float(row.base_prediction)))
            continue
        paths.append(PATH_G)
        values.append(max(0.0, float(row.base_prediction) * float(factor)))
    frame["path"] = paths
    frame["j_prediction"] = values
    return frame


# --------------------------------------------------------------------------------------
# Pre-registered decision rule (written before running the replay)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionCriteria:
    """Fixed before the replay. Accuracy = leaderboard formula (mean_station_accuracy)."""

    max_normal_loss_pp: float = 0.5  # A: preserve G in the stable period
    max_stable_station_loss_pp: float = 1.0  # E: no stable station harmed by more than 1 pp
    min_sustained_gain_pp: float = 0.0  # B: not worse than G in phase 2
    min_shock_gain_pp: float = 5.0  # C: material gain in the shock
    min_shock_hours_for_promotion: float = 12.0  # C: shock evidence must cover at least half a day
    require_post_shock_recovery_data: bool = True  # D: recovery must be observable to promote


def decide(results: dict[str, Any], criteria: DecisionCriteria | None = None, *, post_hoc: bool = False) -> dict[str, Any]:
    """`results` keys: delta_stable, delta_phase1, delta_phase2, delta_shock (pp, J - G),
    worst_stable_station_delta, shock_hours, recovery_observed.
    `post_hoc=True` (a variant designed after seeing these data) caps the outcome at continue_shadow."""
    criteria = criteria or DecisionCriteria()
    checks = {
        "A_preserves_normal": results["delta_stable"] >= -criteria.max_normal_loss_pp,
        "B_not_worse_phase2": results["delta_phase2"] >= criteria.min_sustained_gain_pp,
        "C_shock_gain": results["delta_shock"] >= criteria.min_shock_gain_pp,
        "C_shock_evidence_long_enough": results["shock_hours"] >= criteria.min_shock_hours_for_promotion,
        "D_recovery_observed": bool(results["recovery_observed"]) or not criteria.require_post_shock_recovery_data,
        "E_stable_stations_not_harmed": results["worst_stable_station_delta"] >= -criteria.max_stable_station_loss_pp,
    }
    if not checks["A_preserves_normal"] or not checks["E_stable_stations_not_harmed"]:
        decision = "reject"
        reason = "harms the regime/stations where Candidate G already works"
    elif not checks["B_not_worse_phase2"] and not checks["C_shock_gain"]:
        decision = "reject"
        reason = "no gain in the sustained drift nor in the shock"
    elif all(checks.values()):
        decision = "promotion_candidate"
        reason = "all pre-registered checks pass (still requires manual review; nothing is promoted automatically)"
    else:
        decision = "continue_shadow"
        failed = [name for name, ok in checks.items() if not ok]
        reason = "pending checks: " + ", ".join(failed)
    if post_hoc and decision == "promotion_candidate":
        decision = "continue_shadow"
        reason = "all checks pass but the variant was designed on these data; needs forward shadow evidence"
    return {"decision": decision, "reason": reason, "checks": checks, "criteria": asdict(criteria), "post_hoc": post_hoc}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def seasonal_expectation(wide: pd.DataFrame, when: pd.Timestamp, station: Any, *, weeks: int = 4) -> float:
    """Median of the same slot 1..`weeks` weeks earlier (same definition as drift_regime.seasonal_baseline)."""
    values = [_value(wide, pd.Timestamp(when) - pd.Timedelta(days=7 * k), station) for k in range(1, weeks + 1)]
    values = [v for v in values if math.isfinite(v)]
    return float(np.median(values)) if len(values) == weeks else math.nan


def _value(frame: pd.DataFrame, when: pd.Timestamp, column: Any) -> float:
    try:
        value = frame.at[pd.Timestamp(when), column]
    except KeyError:
        return math.nan
    return float(value) if value is not None and not pd.isna(value) else math.nan


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()) if arr.size and np.isfinite(arr).all() else math.nan


def _std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.std(ddof=0)) if arr.size >= 3 and np.isfinite(arr).all() else math.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _iso(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


__all__ = [
    "CANDIDATE_J_V1", "CANDIDATE_J_V1_1", "CandidateJConfig", "DecisionCriteria", "EVIDENCE", "recent_profile_predictions", "NORMAL", "PATH_G", "PATH_G_BASE_ONLY", "PATH_LATEST", "PATH_RECENT_PROFILE",
    "RegimeThresholds", "SEVERE_SHOCK", "SUSTAINED_DRIFT", "SLOT", "WEEK_STEPS", "calibrate_thresholds", "classify_regimes",
    "compose_predictions", "decide", "g2_usefulness", "hourly_anomaly", "observation_matrix", "seasonal_baseline", "seasonal_expectation", "shock_share", "signals_at",
]
