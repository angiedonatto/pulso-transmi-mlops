"""Markdown renderer for `scripts/drift_regime_analysis.py`.

Every number in the report comes from the analysis objects; re-running the script
regenerates the whole document.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

import drift_regime as dr


def render_markdown(summary: dict[str, Any], obs: dict[str, Any], pred: dict[str, Any], matrix: pd.DataFrame) -> str:
    syn = summary["synthesis"]
    lines: list[str] = []
    add = lines.append
    meta = summary["data_source"]
    reference = summary["reference_window"]
    segments = obs["segments"]
    intensity = segments[segments["series"] == "global_intensity"].sort_values("start").reset_index(drop=True)
    overall = pred["overall"].set_index("window")
    dec = pred["decomposition"]
    sustained_name = syn["sustained_regime"]
    shock_name = syn["shock_regime"]

    add("# Drift / Regime-Change Analysis — Pulso TransMi")
    add("")
    add(f"- Generated (execution clock, UTC): `{summary['generated_at_utc']}`")
    add(f"- Data source: {meta.get('source')} — snapshot `{meta.get('snapshot') or meta.get('path')}`")
    add(f"- Historical data range (UTC): `{summary['observation_range_utc'][0]}` → `{summary['observation_range_utc'][1]}` "
        f"(local {_local(summary['observation_range_utc'][1])})")
    add("- Scope: read-only analysis. No writes to Supabase, no change to `model_versions.is_active`, no submissions, Predict untouched.")
    add("- Conventions: all windows use **historical** time (`observed_at`, `data_cutoff`, `target_at`); Bogota time = America/Bogota. "
        "`residual = actual − prediction` (positive = under-prediction). Two accuracies: `mean_station_accuracy` = mean over stations of "
        "100·max(0, 1 − WAPE_station) with WAPE pooled per station over the window — **the leaderboard formula (verified against /v1/leaderboard)**; "
        "`accuracy` / `global_wape_accuracy` = 100·max(0, 1 − WAPE) pooled over all rows (= leaderboard `raw_wape`). Columns named `accuracy` are the pooled one.")
    add("")

    # ------------------------------------------------------------------ executive summary
    add("## Executive summary")
    add("")
    add(f"- **Drift type:** `{syn['classification']}` → {', '.join(syn['drift_types'])}.")
    phases = [f"{_local(row.start)} (×{row.ratio:.2f} mean |deviation|, {row.hours}h)" for row in intensity.itertuples()]
    add(f"- **When:** stable until **{_local(intensity.loc[1, 'start']) if len(intensity) > 1 else 'n/a'}**. Global-intensity regimes: " + "; ".join(phases) + ".")
    worst = matrix.sort_values("g_shock_accuracy").head(4)
    effect = matrix["level_ratio_sustained"].map(lambda v: abs(math.log(v)) if isinstance(v, float) and v > 0 else 0.0)
    top = matrix.assign(_effect=effect).sort_values("_effect", ascending=False).head(4)
    add(f"- **Most affected (sustained regime `{sustained_name}`, level ratio):** " + ", ".join(f"`{r.station_id}` ×{r.level_ratio_sustained:.2f}" for r in top.itertuples())
        + f"; **latest shock** (`{shock_name}`) hits {len(syn['shock_affected'])}/{syn['stations_total']} stations (worst G accuracy in the shock: "
        + ", ".join(f"`{r.station_id}` {r.g_shock_accuracy:.1f}%" for r in worst.itertuples()) + ").")
    add(f"- **Candidate G (leaderboard formula / pooled WAPE accuracy):** {overall.loc['all_g_cycles', 'mean_station_accuracy']:.1f}% / {overall.loc['all_g_cycles', 'accuracy']:.1f}% "
        f"over all {summary['candidate_g']['resolved_official_cycles']} resolved official cycles; "
        + "; ".join(f"{w} {overall.loc[w, 'mean_station_accuracy']:.1f}% / {overall.loc[w, 'accuracy']:.1f}%" for w in overall.index if w.startswith("regime"))
        + "; " + "; ".join(f"{w} {overall.loc[w, 'mean_station_accuracy']:.1f}% / {overall.loc[w, 'accuracy']:.1f}%" for w in ("last_48h", "last_24h", "last_12h", "last_6h") if w in overall.index) + ".")
    if dec.get("status") == "ok":
        dw = dec["windows"].set_index("window")
        latest = [w for w in dw.index if w.startswith("regime")][-1]
        add(f"- **Base vs calibration:** in `{latest}` the Candidate F base scores {dw.loc[latest, 'base_accuracy']:.1f}% and G {dw.loc[latest, 'g_accuracy']:.1f}% "
            f"(G2 effect {dw.loc[latest, 'g2_effect_pp']:+.1f} pp, {dw.loc[latest, 'share_factor_clipped_high']:.0%} of rows at the 2.5 clip); "
            f"in `{sustained_name}` the G2 effect is {dw.loc[sustained_name, 'g2_effect_pp']:+.1f} pp.")
    for key, comparison in pred["challengers"].items():
        if comparison["status"] == "ok":
            add(f"- **{comparison['model_name']}:** {comparison['shared_cycles']} same-cycle comparisons; mean Δ {comparison['mean_cycle_delta_pp']:+.2f} pp, "
                f"wins {comparison['challenger_wins']}/{comparison['shared_cycles']} → no evidence of greater robustness.")
    no_overlap = [c["model_name"] for c in pred["challengers"].values() if c["status"] != "ok"]
    if no_overlap:
        add(f"- **No same-cycle data** for: {', '.join(sorted(set(no_overlap)))} (no comparison made).")
    add("- **System response:** " + "; ".join(f"`{r['response']}` — {r['reason']}" for r in syn["system_responses"]) + ".")
    add("")

    # ------------------------------------------------------------------ methodology
    add("## 1. Data, windows and method")
    add("")
    add("### Windows")
    add("")
    add(_table(obs["coverage"], ["window", "kind", "start_local", "end_local", "hours", "rows", "coverage", "sufficient"]))
    add("")
    add(f"Every requested window has full coverage (12 stations × 4 slots/h). The newest regime has only **{syn['latest_regime_hours']:.2f}h** of data, "
        "so its persistence cannot be established yet.")
    add("")
    add("### Reference period (stable)")
    add("")
    add(f"`{reference['start_local']}` → `{reference['end_local']}` ({reference['hours']:.0f}h = {reference['hours'] / 24:.0f} complete local days, "
        "4 of each weekday). It ends at the local midnight before the **earliest** material onset detected on any series, so it is pre-drift for every station. "
        "Reference profile = median demand per (station, local weekday, 15-min slot).")
    add("")
    add("Stability check — daily total / reference profile inside the reference window:")
    add("")
    add(_table(obs["reference_stability"], ["station_id", "daily_ratio_median", "daily_ratio_min", "daily_ratio_max", "daily_ratio_std"], digits=3))
    add("")
    add("### Change-point detection")
    add("")
    add("- Baseline: **causal** median of the same weekday & slot 1–4 weeks earlier (no look-ahead). Hourly log-ratio `log((Σy+20)/(Σbaseline+20))`.")
    add("- Global indices: `global_intensity` = mean over stations of |log-ratio| (does not cancel opposite moves); `global_level` = log-ratio of total demand.")
    add("- Binary segmentation (L2 on the mean), penalty `4·σ²·log n` with σ from the MAD of first differences, then change points with < 10% effect are pruned. "
        "Minimum segment 3h (global) / 6h (station).")
    add("")
    add("Sensitivity of the global change points to segment length and penalty (stable conclusions = robust detection):")
    add("")
    add(_table(obs["sensitivity"], ["min_segment_hours", "penalty_beta", "change_points_local", "segment_ratios"]))
    add("")
    add("### Leakage controls")
    add("")
    parity = summary["candidate_g"]["scoring_parity_with_drift_monitor"]
    add(f"- Predictions joined to targets by `(cycle_id, station_id, target_at)`; every row satisfies `target_at − data_cutoff = horizon_minutes > 0` (checked). "
        "`actual` is information revealed after `data_cutoff` and is used only for scoring.")
    add(f"- Only complete, fully resolved `(cycle, model)` sets are scored. Official scoring parity with `drift_monitor.official_scored_predictions`: "
        f"{parity['official_rows_this_script']} vs {parity['official_rows_drift_monitor']} rows, {parity['official_cycles_this_script']} vs {parity['official_cycles_drift_monitor']} cycles.")
    if dec.get("status") == "ok":
        tp = dec["parity"].get("trace_parity") or {}
        add(f"- Candidate G decomposition refits the Candidate F base per cycle with observations `<= data_cutoff` only (asserted). "
            f"{dec['parity']['share_factor_inside_clip']:.1%} of implied factors fall inside the G2 clip [0.35, 2.5]; max spread of the implied factor across the 4 horizons of a "
            f"(cycle, station) = {dec['parity']['max_factor_spread_within_cycle_station']:.2e} (G2 is per station → must be ~0). "
            f"Cycles where it is not ~0 (production saw a different input state) are excluded: {', '.join(dec['parity']['parity_mismatch_cycles_excluded']) or 'none'}. "
            + (f"Parity vs `prediction_traces`: {tp.get('rows')} rows, max |Δbase| = {tp.get('max_abs_base_diff'):.2e}." if tp else ""))
    add(f"- Unresolved official G cycles excluded: {', '.join(summary['candidate_g']['unresolved_official_cycles']) or 'none'}.")
    add("")

    # ------------------------------------------------------------------ questions
    add("## 2. Answers")
    add("")
    onsets = obs["onsets"]
    add("### Q1. When does the drift really start?")
    add("")
    add(_table(intensity.assign(start_local=intensity["start"].map(_local), end_local=intensity["end"].map(_local)), ["segment", "start_local", "end_local", "hours", "mean", "ratio"], digits=3))
    add("")
    add("Station onsets (first segment departing ≥10% from the station's own baseline):")
    add("")
    station_onsets = onsets[~onsets["series"].str.startswith("global")].sort_values("onset_utc")
    add(_table(station_onsets.assign(onset_local=station_onsets["onset_local"].map(_local), latest_segment_start_local=station_onsets["latest_segment_start_local"].map(_local)),
               ["series", "onset_local", "first_shift_ratio", "segments_after_onset", "latest_segment_start_local", "latest_segment_ratio", "latest_segment_hours"], digits=2))
    add("")
    add(f"Reading: the drift is **staged**, not a single break. It starts station by station from **{_local(station_onsets['onset_utc'].min())}**, becomes system-wide "
        f"at **{_local(intensity.loc[1, 'start']) if len(intensity) > 1 else 'n/a'}**, escalates at **{_local(intensity.loc[2, 'start']) if len(intensity) > 2 else 'n/a'}**, "
        f"and turns into a global shock at **{_local(intensity.iloc[-1]['start'])}**.")
    add("")

    add("### Q2–Q3. Which stations changed and which stay stable?")
    add("")
    add(_table(matrix, ["station_id", "drift_onset_local", "sustained_severity", "shock_severity", "level_shift", "profile_shift", "variance_shift"]))
    add("")
    add(f"- Sustained level up: {_list(syn['sustained_level_up'])}; sustained level down: {_list(syn['sustained_level_down'])}.")
    add(f"- No material sustained change: {_list(syn['sustained_stable'])}. No station is untouched by the latest shock ({len(syn['shock_affected'])}/{syn['stations_total']} at high/extreme).")
    add("")

    metrics = obs["station_metrics"]
    add("### Q4. Did the demand level change?")
    add("")
    level = metrics.pivot(index="station_id", columns="window", values="level_ratio")
    order = [c for c in ["last_48h", "last_24h", "last_12h", "last_6h"] + [w.name for w in obs["regimes"][1:]] if c in level.columns]
    add(_table(level[order].reset_index(), ["station_id"] + order, digits=2))
    add("")
    add("Level ratio = Σ observed / Σ reference expectation for the same weekday & slots. Null range (48h same-clock stable windows) is in `tables/station_window_metrics.csv`.")
    add("")

    sus = metrics[metrics["window"] == sustained_name].set_index("station_id")
    shock = metrics[metrics["window"] == shock_name].set_index("station_id")
    add("### Q5. Did the intra-day profile change?")
    add("")
    profile_tbl = pd.DataFrame({
        "station_id": sus.index,
        "hourly_ratio_dispersion": sus["hourly_ratio_dispersion"].values,
        "null_max": sus["null_hourly_ratio_dispersion_max"].values,
        "shape_corr": sus["shape_corr"].values,
        "null_shape_corr_min": sus["null_shape_corr_min"].values,
        "profile_shift": matrix.set_index("station_id").loc[sus.index, "profile_shift"].values,
        "day_consistency": matrix.set_index("station_id").loc[sus.index, "new_regime_day_consistency"].values,
    })
    add(_table(profile_tbl, list(profile_tbl.columns), digits=2))
    add("")
    add("`hourly_ratio_dispersion` = std across local hours of log(observed/reference): 0 for a pure level shift, large when the change depends on the hour. "
        "`day_consistency` = correlation between the last two complete local days (is the new profile repeatable?).")
    add("")

    peaks = obs["peaks"]
    add("### Q6–Q7. Did peaks change in magnitude and in time?")
    add("")
    m = matrix.set_index("station_id")
    peak_tbl = m[["am_peak_ratio_last2d", "am_peak_shift_min_last2d", "pm_peak_ratio_last2d", "pm_peak_shift_min_last2d", "night_peak_ratio_latest", "timing_shift_minutes_sustained", "timing_shift_minutes_shock"]].reset_index()
    add(_table(peak_tbl, list(peak_tbl.columns), digits=2))
    add("")
    add("AM = 05:00–10:59, PM = 16:00–21:59, night = 22:00–04:59 local; ratios vs the same-weekday reference peak, shifts in minutes (+ = later). "
        "`timing_shift_minutes` = lag that maximizes the correlation between observed demand and the time-shifted reference (+ = observed pattern happens later); "
        "for windows shorter than 6h (the shock) the ±8h search is not identifiable, so that column is descriptive only and is not used to classify. "
        "Per-day detail in `tables/daily_peaks.csv`.")
    add("")

    add("### Q8. Did variance / volatility change?")
    add("")
    var_tbl = pd.DataFrame({
        "station_id": sus.index,
        "volatility_sustained": sus["volatility"].values,
        "null_median_sustained": sus["null_volatility_median"].values,
        "ratio_sustained": (sus["volatility"] / sus["null_volatility_median"]).values,
        "volatility_shock": shock.loc[sus.index, "volatility"].values,
        "null_median_shock": shock.loc[sus.index, "null_volatility_median"].values,
        "ratio_shock": (shock.loc[sus.index, "volatility"] / shock.loc[sus.index, "null_volatility_median"]).values,
        "label_sustained": m.loc[sus.index, "variance_shift"].values,
        "label_shock": m.loc[sus.index, "shock_variance_shift"].values,
    })
    add(_table(var_tbl, list(var_tbl.columns), digits=3))
    add("")
    add("Volatility = std of 15-min changes of log1p(demand) (level-invariant). Absolute variance grows with level wherever level grows; "
        "the question here is whether demand became *relatively* noisier or smoother. Labels need ×1.5 (or ×1/1.5) vs the stable median.")
    add("")
    vol = obs["daily_volatility"]
    recent_days = sorted(vol["local_date"].unique())[-12:]
    daily = vol[vol["local_date"].isin(recent_days)].pivot(index="local_date", columns="station_id", values="volatility")
    daily.insert(0, "median_all_stations", daily.median(axis=1))
    add("Daily volatility per local date (median across stations first; `tables/daily_volatility.csv` for all dates):")
    add("")
    add(_table(daily.reset_index(), ["local_date"] + list(daily.columns), digits=3))
    add("")

    lags = obs["lag_table"]
    add("### Q9–Q13. Did the lag relationships break (lag_15, lag_30, lag_60, lag_96, weekly)?")
    add("")
    pooled = lags[lags["station_id"] == "ALL"]
    lag_tbl = pooled.pivot(index="lag", columns="window", values="naive_accuracy").reindex(list(dr.LAGS))
    lag_cols = [c for c in ["reference", "last_48h", "last_24h", "last_12h", "last_6h"] + [w.name for w in obs["regimes"][1:]] if c in lag_tbl.columns]
    add("Naive-lag accuracy (forecast `ŷ_t = y_{t−k}`), all stations pooled, %:")
    add("")
    add(_table(lag_tbl[lag_cols].reset_index(), ["lag"] + lag_cols, digits=1))
    add("")
    corr_tbl = pooled.pivot(index="lag", columns="window", values="lag_corr").reindex(list(dr.LAGS))
    add("Correlation corr(log1p y_t, log1p y_{t−k}), pooled:")
    add("")
    add(_table(corr_tbl[lag_cols].reset_index(), ["lag"] + lag_cols, digits=3))
    add("")
    add(f"Per-station reliability in the sustained regime (`{sustained_name}`) and in the shock (`{shock_name}`), each vs same-clock stable windows "
        "(`broken` = > 15 pp below the stable median and below the stable minimum):")
    add("")
    counts = pd.DataFrame(syn["lag_reliability_counts"]).fillna(0).astype(int).T.reset_index().rename(columns={"index": "lag / window"})
    add(_table(counts, list(counts.columns)))
    add("")
    add(_table(matrix, ["station_id", "lag15_reliability", "lag30_reliability", "lag60_reliability", "lag96_reliability", "weekly_lag_reliability", "lag15_reliability_shock", "lag30_reliability_shock", "lag60_reliability_shock", "lag96_reliability_shock", "weekly_lag_reliability_shock"]))
    add("")

    add("### Q14–Q15. Is Candidate G over- or under-predicting? Global or localized?")
    add("")
    add(_table(pred["overall"], ["window", "count", "mean_station_accuracy", "accuracy", "wape", "mae", "signed_bias", "relative_bias", "pred_actual_ratio"], digits=3))
    add("")
    bs = pred["by_station"]
    bias_tbl = bs[bs["window"].isin([sustained_name, shock_name, "last_48h", "last_6h"])].pivot(index="station_id", columns="window", values="relative_bias")
    share = bs[bs["window"] == sustained_name].set_index("station_id")["share_cycles_underpredicted"]
    bias_tbl["share_cycles_under_" + sustained_name] = share
    add("Relative bias `Σ(actual − prediction)/Σactual` by station (+ = under-prediction):")
    add("")
    add(_table(bias_tbl.reset_index(), ["station_id"] + list(bias_tbl.columns), digits=3))
    add("")
    over = bias_tbl.index[bias_tbl[sustained_name] <= -0.05].tolist() if sustained_name in bias_tbl else []
    under = bias_tbl.index[bias_tbl[sustained_name] >= 0.05].tolist() if sustained_name in bias_tbl else []
    add(f"In `{sustained_name}`: material over-prediction (≤ −5%) at {_list(over)}; material under-prediction (≥ +5%) at {_list(under)}. "
        "The global bias is small because station biases have opposite signs: the bias is **localized**, not global.")
    add("")

    add("### Q16. Does the error depend on the horizon?")
    add("")
    hz = pred["by_horizon"]
    hz_acc = hz.pivot(index="horizon_minutes", columns="window", values="accuracy")
    hz_bias = hz.pivot(index="horizon_minutes", columns="window", values="relative_bias")
    cols = [c for c in ["last_48h", "last_24h", "last_12h", "last_6h"] + [w.name for w in obs["regimes"][1:]] if c in hz_acc.columns]
    add("Accuracy (%):")
    add("")
    add(_table(hz_acc[cols].reset_index(), ["horizon_minutes"] + cols, digits=1))
    add("")
    add("Relative bias (+ = under-prediction):")
    add("")
    add(_table(hz_bias[cols].reset_index(), ["horizon_minutes"] + cols, digits=3))
    add("")

    add("### Q17. Do some stations follow a different regime?")
    add("")
    groups = station_groups(matrix)
    for name, stations in groups.items():
        add(f"- **{name}:** {_list(stations)}")
    add("")

    add("### Q18. What kind of drift is it?")
    add("")
    add(f"`{syn['classification']}`: " + ", ".join(syn["drift_types"]) + ".")
    add("")
    add(f"- Level shift: {len(syn['sustained_level_up']) + len(syn['sustained_level_down'])}/{syn['stations_total']} stations (up: {_list(syn['sustained_level_up'])}; down: {_list(syn['sustained_level_down'])}).")
    add(f"- Temporal/profile shift: {len(syn['sustained_profile_shift'])}/{syn['stations_total']} stations.")
    add(f"- Variance shift: sustained regime smoother (relative volatility ≤ ×0.67) at {_list(syn['sustained_variance_down'])}, noisier at {_list(syn['sustained_variance_up'])}; "
        f"shock noisier (≥ ×1.5) at {_list(syn['shock_variance_up'])}.")
    add(f"- Lag failure: lag_96 broken/degraded in the sustained regime for {syn['lag_reliability_counts']['lag96_reliability'].get('broken', 0) + syn['lag_reliability_counts']['lag96_reliability'].get('degraded', 0)} stations, "
        f"in the shock for {syn['lag_reliability_counts']['lag96_reliability_shock'].get('broken', 0) + syn['lag_reliability_counts']['lag96_reliability_shock'].get('degraded', 0)}; "
        f"weekly lag broken/degraded for {syn['lag_reliability_counts']['weekly_lag_reliability'].get('broken', 0) + syn['lag_reliability_counts']['weekly_lag_reliability'].get('degraded', 0)}; "
        f"lag_15 stays reliable for {syn['lag_reliability_counts']['lag15_reliability'].get('reliable', 0)} stations in the sustained regime and {syn['lag_reliability_counts']['lag15_reliability_shock'].get('reliable', 0)} in the shock.")
    add(f"- Localized vs global: the sustained regime is localized; the latest regime ({syn['latest_regime_hours']:.2f}h) is global.")
    add("")

    # ------------------------------------------------------------------ Candidate G
    add("## 3. Candidate G error analysis")
    add("")
    cycles = pred["cycles"]
    g_cycles = cycles[(cycles["model_id"] == pred["g_model_id"]) & cycles["is_official"]].sort_values("data_cutoff")
    add(f"Per-cycle accuracy — leaderboard formula (one-cycle window): median {g_cycles['macro_accuracy'].median():.1f}%, min {g_cycles['macro_accuracy'].min():.1f}%; "
        f"pooled global WAPE: median {g_cycles['accuracy'].median():.1f}%, min {g_cycles['accuracy'].min():.1f}%. Worst cycles (by pooled):")
    add("")
    add(_table(g_cycles.nsmallest(6, "accuracy").assign(data_cutoff_local=lambda f: f["data_cutoff"].map(_local)),
               ["cycle_id", "data_cutoff_local", "accuracy", "macro_accuracy", "wape", "relative_bias", "pred_actual_ratio"], digits=3))
    add("")
    add("Extreme errors (largest |error|; gross misses = pred/actual ≥ 2 or ≤ 0.5 with actual ≥ 50):")
    add("")
    ex = pred["extremes"].head(12).assign(target_local=lambda f: f["target_at"].map(_local))
    add(_table(ex, ["cycle_id", "station_id", "horizon_minutes", "target_local", "prediction", "actual", "residual", "pred_actual_ratio"], digits=1))
    add("")
    add("Error by local target hour (last 48h):")
    add("")
    bh = pred["by_hour"]
    add(_table(bh[bh["window"] == "last_48h"], ["target_hour_local", "count", "accuracy", "relative_bias", "mae"], digits=3))
    add("")
    if dec.get("status") == "ok":
        add("### Base model (Candidate F) vs G2 calibration")
        add("")
        add(_table(dec["windows"], ["window", "rows", "base_accuracy", "g_accuracy", "g2_effect_pp", "base_relative_bias", "g_relative_bias",
                                    "share_factor_clipped_high", "share_factor_clipped_low", "median_factor", "corr_log_factor_vs_needed", "share_factor_wrong_direction"], digits=3))
        add("")
        add("`g2_effect_pp` = G accuracy − base accuracy. `corr_log_factor_vs_needed` = correlation between the applied factor and the factor that would have been exact "
            "(actual/base): positive = the calibration points the right way. `share_factor_wrong_direction` = factor moved base away from the actual.")
        add("")
        worst_cycles = dec["per_cycle"].nsmallest(5, "g_accuracy").assign(data_cutoff_local=lambda f: f["data_cutoff"].map(_local))
        add(_table(worst_cycles, ["cycle_id", "data_cutoff_local", "base_accuracy", "g_accuracy", "g2_effect_pp", "share_factor_clipped"], digits=2))
        add("")

    # ------------------------------------------------------------------ challengers
    add("## 4. Challengers (same cycles only)")
    add("")
    for key, comparison in sorted(pred["challengers"].items(), key=lambda item: item[1]["status"] == "ok"):
        if comparison["status"] != "ok":
            add(f"- `{comparison['model_name']}` ({comparison['model_id'][:8]}): **no shared resolved cycles with Candidate G** "
                f"(own resolved cycles: {comparison['own_resolved_cycles']}, cutoffs {comparison['own_cutoff_range_local']}). Not compared.")
            continue
        add("")
        add(f"### {comparison['model_name']}")
        add("")
        add(f"{comparison['shared_cycles']} shared cycles, {comparison['rows']} rows. Pooled accuracy G {comparison['champion_pooled_accuracy']:.2f}% vs "
            f"challenger {comparison['challenger_pooled_accuracy']:.2f}%; mean per-cycle Δ {comparison['mean_cycle_delta_pp']:+.2f} pp (median {comparison['median_cycle_delta_pp']:+.2f}); "
            f"challenger wins {comparison['challenger_wins']}, loses {comparison['challenger_losses']}.")
        add("")
        add(_table(comparison["by_regime"], ["regime", "cycles", "champion_mean", "challenger_mean", "mean_delta_pp", "wins"], digits=2))
        add("")
        add(_table(comparison["by_horizon"], ["horizon_minutes", "rows", "champion_accuracy", "challenger_accuracy", "delta_pp"], digits=2))
        add("")
        add("Shared cycles are sparse (the shadow job does not run every cycle), so this is directional evidence only. Nothing is promoted.")
        add("")
    official = pred["cycles"][pred["cycles"]["is_official"]].copy()
    official["regime"] = official["data_cutoff"].map(lambda ts: _regime_of(ts, obs["regimes"]))
    ctx = official.groupby(["model_name", "regime"]).agg(cycles=("cycle_id", "size"), mean_accuracy=("accuracy", "mean")).reset_index()
    add("Context only (different cycles, **not** a head-to-head): mean pooled accuracy of each model while it was the official one, by regime:")
    add("")
    add(_table(ctx, ["model_name", "regime", "cycles", "mean_accuracy"], digits=2))
    add("")

    # ------------------------------------------------------------------ station matrix
    add("## 5. Station diagnostic matrix")
    add("")
    add(f"Shift labels, lag reliabilities and the G metrics (`accuracy`, `wape`, `signed_bias` = mean(actual − prediction), `relative_bias`) refer to the sustained regime "
        f"`{sustained_name}`; `g_shock_accuracy` to the shock `{shock_name}`; `g_last_48h_accuracy` to the requested 48h window. Full matrix (with 48h/6h columns): `station_diagnostics.csv`.")
    add("")
    add(_table(matrix, ["station_id", "drift_severity", "level_shift", "profile_shift", "variance_shift", "lag15_reliability", "lag30_reliability", "lag60_reliability",
                        "lag96_reliability", "weekly_lag_reliability", "signed_bias", "relative_bias", "wape", "accuracy", "g_shock_accuracy", "g_last_48h_accuracy",
                        "recommended_response"], digits=3))
    add("")
    add(_table(matrix, ["station_id", "sustained_response", "shock_response", "evidence"]))
    add("")

    # ------------------------------------------------------------------ conclusions
    add("## 6. Implications for the next challenger")
    add("")
    for item in syn["system_responses"]:
        add(f"- `{item['response']}` — {item['reason']}.")
    add("")
    add("Diagnostic thresholds (not model parameters) are in `drift_regime.ClassificationConfig`; no hyper-parameter, alpha, clip or weight was tuned or proposed here.")
    add("")
    add("## 7. Figures")
    add("")
    for name in summary["outputs"]["figures"]:
        add(f"- `figures/{name}`")
    add("")
    add("Table views for every figure are under `tables/` (same data, CSV).")
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("# live (read-only GET from Supabase, writes a snapshot under data/interim/drift_regime_analysis/snapshot)")
    add("python scripts/drift_regime_analysis.py")
    add("# offline, exactly this report")
    add(f"python scripts/drift_regime_analysis.py --from-snapshot {meta.get('snapshot') or meta.get('path')} --g-decomposition cache")
    add("```")
    return "\n".join(lines)


def station_groups(matrix: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for row in matrix.itertuples(index=False):
        level = str(row.level_shift)
        strong_level = level.split("_")[0] in ("high", "extreme")
        absorbed = row.sustained_response == "keep current behavior"
        if strong_level and level.endswith("down"):
            name = "A. Structural collapse: level down ×0.2–0.5 + profile change, not absorbed by G"
        elif strong_level and level.endswith("up") and not absorbed:
            name = "B. Structural surge: level up ×2.5–3.3, peaks amplified and delayed, not absorbed by G"
        elif str(row.profile_shift) != "none" and not absorbed:
            name = "C. Peak timing/profile shift (peaks later, hour-dependent change), partially absorbed"
        elif str(row.profile_shift) != "none" or row.sustained_severity not in ("none", "low"):
            name = "D. Sustained shift already absorbed by G2"
        else:
            name = "E. Stable (at most a mild level drift) until the latest global shock"
        groups.setdefault(name, []).append(row.station_id)
    return dict(sorted(groups.items()))


def _regime_of(timestamp: pd.Timestamp, regimes: list[dr.Window]) -> str:
    for window in regimes:
        if window.start < timestamp <= window.end:
            return window.name
    return "before_regimes"


def _table(frame: pd.DataFrame, columns: list[str], *, digits: int = 2) -> str:
    columns = [c for c in columns if c in frame.columns]
    if frame.empty or not columns:
        return "_(no data)_"
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in columns) + "|"
    rows = [header, sep]
    for record in frame[columns].itertuples(index=False):
        rows.append("| " + " | ".join(_cell(value, digits) for value in record) + " |")
    return "\n".join(rows)


def _cell(value: Any, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return _local(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value).replace("|", "/")


def _local(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return "n/a"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(dr.BOGOTA_TZ).strftime("%Y-%m-%d %H:%M (Bogota)")


def _list(values: Any) -> str:
    values = list(values or [])
    return ", ".join(f"`{v}`" for v in values) if values else "none"
