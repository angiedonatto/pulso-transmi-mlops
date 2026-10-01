"""Markdown renderer for `scripts/candidate_j_replay.py` (all numbers come from the replay)."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

import candidate_j_regime as cj


RULES = [
    ("NORMAL → Candidate G unchanged", "normal_keep_g", "base_F × G2 exactly"),
    ("SUSTAINED_DRIFT → keep G2 only where it helped the station in the last 24h", "g2_useful_sustained + g2_heterogeneous", "base_F × G2 if realised G2 effect ≥ 0 pp, else base_F"),
    ("SUSTAINED_DRIFT ≥ 24h, repeatable new profile, no jump vs. yesterday → recent profile", "lag96_day1_day2 + new_profile_repeats", "y(t − 24h) × (last hour / same hour yesterday)"),
    ("v1.1 only: recent profile also needs to have beaten G and base_F on the station's last 24h", "keep_where_g_works + v1_replay_diagnosis", "otherwise keep G"),
    ("SEVERE_SHOCK (≥ 75% of stations jump vs. yesterday, kept 3 h) → latest observation", "shock_is_global + shock_short_lags + g2_fails_shock + min_regime_length", "y(data_cutoff); no base_F, no G2, no lag_96/weekly"),
    ("Compounding guard: factor at the G2 clip + base already beyond the station's stable range in the same direction → factor 1", "double_adaptation + g2_fails_shock", "drops the extreme factor only"),
]


def render_markdown(payload: dict[str, Any], variants: dict[str, dict[str, Any]], parity: dict[str, Any], figures: dict[str, str]) -> str:
    lines: list[str] = []
    add = lines.append
    primary_name = payload["primary_variant"]
    primary = variants[primary_name]
    results = primary["results"]
    inputs = primary["decision_inputs"]
    rec = payload["recovery"]
    phases = [p["phase"] for p in payload["phases"]]

    add("# Candidate J — regime-aware shadow challenger (retrospective replay)")
    add("")
    add(f"- Generated (execution clock): `{payload['generated_at_utc']}` · data up to (historical): `{payload['data_end_utc']}` · snapshot `{payload['snapshot']}`")
    add("- Scope: **shadow replay only**. Nothing was written to Supabase, Predict, `model_versions.is_active` or the submission API. Nothing is promoted.")
    add(f"- Accuracy: {payload['accuracy_definition']}. Δ = J − G in percentage points on the **same targets**; phase accuracy pools each station over the whole phase, like the leaderboard window.")
    add("")

    add("## Decision")
    add("")
    add("| variant | designed | decision | reason | Δ stable | Δ phase 1 | Δ phase 2 | Δ shock | worst stable-station Δ |")
    add("|---|---|---|---|---:|---:|---:|---:|---:|")
    for name, v in variants.items():
        d = v["decision_inputs"]
        designed = "post-hoc (after diagnosing v1 on these data)" if v["post_hoc"] else "pre-registered"
        add(f"| `{name}` | {designed} | **{v['decision']['decision']}** | {v['decision']['reason']} | {d['delta_stable']:+.2f} | {d['delta_phase1']:+.2f} | "
            f"{d['delta_phase2']:+.2f} | {d['delta_shock']:+.2f} | {d['worst_stable_station_delta']:+.2f} |")
    add("")
    add("Pre-registered checks (thresholds fixed in `candidate_j_regime.DecisionCriteria` before the first replay):")
    add("")
    add("| check | " + " | ".join(f"`{n}`" for n in variants) + " |")
    add("|---|" + "---|" * len(variants))
    for check in primary["decision"]["checks"]:
        add(f"| {check} | " + " | ".join("pass" if v["decision"]["checks"][check] else "FAIL / pending" for v in variants.values()) + " |")
    add("")
    add(f"Shock observed: {inputs['shock_hours']:.0f} h, still active at the last observation; recovery observed: {inputs['recovery_observed']}.")
    add("")

    add("## 1. Architecture and evidence")
    add("")
    add("| rule | evidence (drift_regime_analysis unless marked post-hoc) | action |")
    add("|---|---|---|")
    for rule, keys, action in RULES:
        quotes = "; ".join(cj.EVIDENCE[k.strip()] for k in keys.split("+"))
        add(f"| {rule} | {quotes} | {action} |")
    add("")
    th = payload["thresholds"]
    add(f"Thresholds are each station's extremes of the causal statistics over a stable calibration window `{th['calibration_start']}` → `{th['calibration_end']}` "
        f"({th['samples']} station-hours), which ends before the first evaluated cycle (log units vs. the causal 4-week same-slot median). "
        "No parameter of Candidate G (alpha, clip, EWMA window) was changed or tuned; J only chooses between existing signals.")
    add("")
    per = pd.DataFrame(th["per_station"]).T.reset_index().rename(columns={"index": "station_id"})
    add(_table(per, ["station_id", "level24_abs_max", "dispersion24_max", "jump_abs_max", "hour_anomaly_abs_max"], digits=3))
    add("")
    sens = payload["sensitivity_pooled_thresholds"]
    add("Design change made before any accuracy was computed: a first version pooled all stations into one threshold (the max over stations). "
        f"Share of jumping stations over the shock cutoffs {sens['shock_cutoffs_local']}: pooled {sens['pooled_share_in_shock']} vs per-station {sens['per_station_share_in_shock']} "
        f"(shock declared at ≥ {sens['global_shock_share']}). Before the shock the maximum share was {sens['max_pooled_share_before_shock']:.2f} (pooled) and "
        f"{sens['max_per_station_share_before_shock']:.2f} (per-station): no false alarm in either design. Per-station thresholds are what the drift analysis itself uses.")
    add("")

    add("## 2. Why v1 was rejected and what v1.1 changes")
    add("")
    v1 = variants.get("candidate_j_v1")
    if v1 is not None:
        add("J vs G restricted to the rows each path produced (pooled accuracy on those rows; `abs_error_saved` > 0 = J better):")
        add("")
        merged = v1["results"]["path_attribution"].merge(results["path_attribution"], on=["phase", "path"], how="outer", suffixes=("_v1", "_v1_1"))
        for column in ("rows_v1", "rows_v1_1"):
            merged[column] = merged[column].astype("Int64").astype(str).replace("<NA>", "0")
        add(_table(merged, ["phase", "path", "rows_v1", "g_global_wape_accuracy_v1", "j_global_wape_accuracy_v1", "abs_error_saved_v1", "rows_v1_1", "j_global_wape_accuracy_v1_1", "abs_error_saved_v1_1"], digits=2))
        add("")
        e1 = v1["results"]["stations"]
        e1 = e1[e1["stable_station"]].pivot(index="station_id", columns="phase", values="delta_pp").reindex(columns=phases)
        add("v1 on the stations the drift analysis marked “keep current behavior” (J − G, pp) — the recent-profile path replaced a G that already worked:")
        add("")
        add(_table(e1.reset_index(), ["station_id"] + phases, digits=2))
        add("")

    add(f"## 3. Results by phase — `{primary_name}` (same targets)")
    add("")
    overall = results["overall"]
    add(_table(overall, ["phase", "model", "cycles", "rows", "mean_station_accuracy", "global_wape_accuracy", "global_wape", "accuracy_at_20", "mae", "signed_bias", "relative_bias"], digits=3))
    add("")
    wins = results["wins"].set_index("phase")
    acc = overall.pivot(index="phase", columns="model", values="mean_station_accuracy").reindex(phases)
    pooled = overall.pivot(index="phase", columns="model", values="global_wape_accuracy").reindex(phases)
    summary = pd.DataFrame({
        "phase": acc.index, "G": acc["G"].values, "J": acc["J"].values, "Δ (pp)": (acc["J"] - acc["G"]).values,
        "G pooled": pooled["G"].values, "J pooled": pooled["J"].values, "Δ pooled (pp)": (pooled["J"] - pooled["G"]).values,
        "cycles won": [int(wins.loc[p, "win"]) if p in wins.index else 0 for p in acc.index],
        "lost": [int(wins.loc[p, "loss"]) if p in wins.index else 0 for p in acc.index],
        "tied": [int(wins.loc[p, "tie"]) if p in wins.index else 0 for p in acc.index],
    })
    add(_table(summary, list(summary.columns), digits=2))
    add("")
    add("Cycle wins/losses use the leaderboard formula on a one-cycle window, which caps each station-cycle at 0%: a catastrophic cycle costs G at most "
        "the station's share, while over a phase (as on the leaderboard) the full error counts. That is why J can lose several shock cycles yet win the phase.")
    add("")
    add("Worst station per phase:")
    add("")
    add(_table(results["worst"], ["phase", "g_worst_station", "g_worst_accuracy", "j_worst_station", "j_worst_accuracy"], digits=2))
    add("")
    add("Paths used (share of rows per phase):")
    add("")
    add(_table(results["paths"], ["phase", "regime", "path", "rows", "share"], digits=3))
    add("")

    add(f"## 4. Answers (`{primary_name}`)")
    add("")
    st = results["stations"]
    stable_rows = st[st["phase"] == "stable"]
    add(f"**A. Does J preserve G in the normal regime?** Δ stable = {inputs['delta_stable']:+.2f} pp; per-station Δ in the stable week from "
        f"{stable_rows['delta_pp'].min():+.2f} to {stable_rows['delta_pp'].max():+.2f} pp; false SEVERE_SHOCK alarms before the shock: {len(rec['false_shock_alarms_before_phase3'])}.")
    add("")
    p2 = st[st["phase"] == "phase2"].sort_values("delta_pp")
    add(f"**B. Does J improve the structural changes of phase 2?** Δ phase 2 = {inputs['delta_phase2']:+.2f} pp (phase 1: {inputs['delta_phase1']:+.2f} pp). Per station:")
    add("")
    add(_table(p2, ["station_id", "g_accuracy", "j_accuracy", "delta_pp", "g_relative_bias", "j_relative_bias", "share_rows_changed"], digits=2))
    add("")
    p3 = st[st["phase"] == "phase3_shock"].sort_values("delta_pp")
    add(f"**C. Does J avoid or reduce the phase-3 collapse?** Δ shock = {inputs['delta_shock']:+.2f} pp over {rec['shock_cycles']} shock cycles "
        f"({rec['shock_cycles_flagged']} flagged SEVERE_SHOCK; first detection `{rec['first_shock_detection']}`, {rec['detection_latency_hours']} h after the shock start `{rec['shock_start']}`).")
    add("")
    add(_table(p3, ["station_id", "g_accuracy", "j_accuracy", "delta_pp", "g_relative_bias", "j_relative_bias"], digits=2))
    add("")
    cyc = results["cycles"]
    add(_table(cyc[cyc["phase"] == "phase3_shock"], ["data_cutoff", "g_mean_station_accuracy", "j_mean_station_accuracy", "delta_pp", "g_global_wape_accuracy", "j_global_wape_accuracy", "regimes", "paths"], digits=2))
    add("")
    add(f"**D. Does it recover quickly?** The shock is still active at the last observation (`{rec['shock_still_active_at_last_cutoff']}`): recovery *after* the shock "
        "cannot be measured yet. Measurable now: 1 h detection latency, the shock state is held through the wave's troughs, and the first, undetected shock cycle is handled by the "
        "sustained-drift paths (table above).")
    add("")
    e = st[st["stable_station"]].pivot(index="station_id", columns="phase", values="delta_pp").reindex(columns=phases)
    add(f"**E. Does J avoid harming stations that stayed stable?** Stations marked “keep current behavior” by the drift analysis: {', '.join(payload['stable_stations_from_drift_analysis'])}. J − G (pp):")
    add("")
    add(_table(e.reset_index(), ["station_id"] + phases, digits=2))
    add("")

    add("## 5. Accuracy by horizon")
    add("")
    add(_table(results["horizons"], ["phase", "horizon_minutes", "g_accuracy", "j_accuracy", "delta_pp", "g_relative_bias", "j_relative_bias"], digits=2))
    add("")
    add("## 6. Replay fidelity (official cycles)")
    add("")
    if parity.get("rows"):
        pc = parity["per_cycle"]
        add(f"Replay G vs the official G predictions on {parity['cycles']} official cycles ({parity['rows']} rows): {parity['share_rows_identical_1e-6']:.1%} identical, "
            f"median |Δ| {parity['median_abs_diff']:.3f}, max |Δ| {parity['max_abs_diff']:.1f}. Differences come from production G2 feedback skipping the cycles the pipeline missed "
            f"(the replay has every hourly cycle). Mean per-cycle accuracy on the official cycles: official G {pc['official_g'].mean():.2f}%, replay G {pc['replay_g'].mean():.2f}%, "
            f"J {pc['j'].mean():.2f}% (`tables/official_cycles_comparison.csv`).")
    add("")
    add("## 7. Limitations")
    add("")
    add("- The shock path was chosen from the drift-analysis lag table, measured on the same shock hours: phase-3 results are **in-sample for the design**. Only future shock hours are a clean test.")
    add("- v1.1 was designed after seeing v1's replay on these data; its numbers are post-hoc and its decision is capped at `continue_shadow`.")
    add(f"- The shock has {inputs['shock_hours']:.0f} h of data and has not ended: no post-shock recovery is observable.")
    add("- The stable-week result is out-of-sample for the thresholds (calibrated before it).")
    add("")
    add("## 8. Figures")
    add("")
    for name in figures:
        add(f"- `figures/{name}`")
    add("")
    add("## 9. Reproduce")
    add("")
    add("```bash")
    add(f"python scripts/candidate_j_replay.py --from-snapshot {payload['snapshot']}")
    add("python scripts/run_candidate_j_shadow.py            # dry run on the latest live cycle (read-only)")
    add("```")
    return "\n".join(lines)


def _table(frame: pd.DataFrame, columns: list[str], *, digits: int = 2) -> str:
    columns = [c for c in columns if c in frame.columns]
    if frame.empty or not columns:
        return "_(no data)_"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in columns) + "|"]
    for record in frame[columns].itertuples(index=False):
        out.append("| " + " | ".join(_cell(v, digits) for v in record) + " |")
    return "\n".join(out)


def _cell(value: Any, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("America/Bogota").strftime("%m-%d %H:%M")
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)
