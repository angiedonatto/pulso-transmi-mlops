"""Candidate I -- final tables + reports/candidate_i/summary.md (run after `candidate_i_lab.py final`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidate_i_core as core  # noqa: E402
import candidate_i_eval as ev  # noqa: E402
from candidate_i_report import REPORT_DIR, markdown_table  # noqa: E402

MODELS = {"B (walk-forward)": "pred_B", "B (served)": "pred_B_served", "F": "pred_F", "G": "pred_G", "I": "pred_I"}


def final_table(fr: pd.DataFrame, runtime: dict[str, str]) -> pd.DataFrame:
    masks = ev.window_masks(fr)
    g_cyc = core.cycle_scores(fr, "pred_G").set_index("data_cutoff")["cycle_accuracy"]
    rows = []
    for name, col in MODELS.items():
        cyc = core.cycle_scores(fr, col).set_index("data_cutoff")["cycle_accuracy"]
        hist = cyc[cyc.index >= ev.STABLE[0]]

        def win(w: str) -> float:
            cut = fr.loc[masks[w], "data_cutoff"].unique()
            vals = cyc.reindex(cut).dropna()
            return float(vals.mean()) if len(vals) else np.nan

        st = ev.station_table(fr, [col], masks["all_scored"])[col].dropna()
        d = (hist - g_cyc.reindex(hist.index)).dropna()
        rows.append(
            {
                "model": name,
                "Historical": float(hist.mean()),
                "Recent12": win("recent12"),
                "Recent6": win("recent6"),
                "Stable regime": win("stable"),
                "Drift regime": win("drift_all"),
                "Drift test (sealed)": win("drift_test"),
                "Worst station": f"{st.idxmin()} {st.min():.2f}" if len(st) else "",
                "Median cycle": float(hist.median()),
                "Worst cycle": float(hist.min()),
                "Wins vs G": "" if name == "G" else int((d > 1e-9).sum()),
                "Losses vs G": "" if name == "G" else int((d < -1e-9).sum()),
                "Runtime/cycle": runtime.get(name, ""),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def station_deltas(fr: pd.DataFrame) -> pd.DataFrame:
    masks = ev.window_masks(fr)
    out = {}
    for w in ("stable", "drift_dev", "drift_test", "all_scored"):
        t = ev.station_table(fr, ["pred_G", "pred_I"], masks[w])
        out[f"{w}_I-G"] = t["pred_I"] - t["pred_G"]
    return pd.DataFrame(out)


def horizon_deltas(fr: pd.DataFrame) -> pd.DataFrame:
    masks = ev.window_masks(fr)
    out = {}
    for w in ("stable", "drift_test", "all_scored"):
        t = ev.horizon_table(fr, ["pred_F", "pred_G", "pred_I"], masks[w])
        out[f"{w}_G"] = t["pred_G"]
        out[f"{w}_I"] = t["pred_I"]
        out[f"{w}_I-G"] = t["pred_I"] - t["pred_G"]
    return pd.DataFrame(out)


def glab_verification(fr: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the windows of the Candidate G lab report (its 'recent12'/'recent6')."""
    off = fr[fr["is_official"].astype(bool) & (fr["data_cutoff"] >= ev.GLAB12[0]) & (fr["data_cutoff"] <= ev.GLAB12[1])]
    cut = sorted(off["data_cutoff"].unique())
    rows = {}
    for label, cuts in (("glab_recent12", cut), ("glab_recent6", cut[-6:])):
        sub = fr[fr["data_cutoff"].isin(cuts)]
        rows[label] = {name: core.mean_cycle_accuracy(sub, col) for name, col in MODELS.items()}
    return pd.DataFrame(rows).T


def _delta_rows(summary: dict) -> pd.DataFrame:
    rows = {}
    for w in ("stable", "drift_dev", "drift_test", "drift_all", "official", "recent12", "recent6", "glab12", "all_scored"):
        d = summary["deltas_vs_G"][w]
        b = summary["bootstrap_vs_G"].get(w, {})
        rows[w] = {
            "cycles": d["cycles"],
            "mean_delta_pp": d["mean_delta"],
            "median_delta_pp": d["median_delta"],
            "worst_cycle_delta_pp": d["worst_delta"],
            "wins": d["wins"],
            "losses": d["losses"],
            "P(I>G) block bootstrap": f"{b['p_better']:.3f}" if b.get("p_better") is not None else "",
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "window"
    return out


def render_markdown(summary: dict, table: pd.DataFrame, stations: pd.DataFrame, horizons: pd.DataFrame, glab: pd.DataFrame, stress: pd.DataFrame | None) -> str:
    table = table.copy()
    table.index.name = "model"
    stations.index.name = "station_id"
    horizons.index.name = "horizon_minutes"
    glab.index.name = "window"
    abl = pd.DataFrame(summary["ablations"]).T[["stable", "drift_dev", "drift_test", "recent12", "recent6", "all_scored"]]
    abl.index.name = "ablation (delta vs I, pp)"
    parity = pd.DataFrame(summary["parity_production_vs_lab"]).set_index("cutoff")[["rows", "max_abs_diff_prediction", "max_abs_diff_factor", "runtime_candidate_i_s", "runtime_one_f_refit_s"]]
    leak = summary["leakage_audit"]
    sel = summary["selection_record"]
    lines = [
        "# Candidate I -- SNR-gated resolved-error level correction (`candidate_i_f4_oos_g2_snr_gate`)",
        "",
        "## Status",
        "",
        "- Classification: **READY FOR SHADOW**. Not promoted; `is_active`, the leaderboard model, the submission payload, scoring and Candidate G are untouched.",
        "- Research only until shadow evidence exists: no Supabase writes, no commits, no workflow changes.",
        "",
        "## Model",
        "",
        "```",
        "final = F_base * clip(exp(gate * log(level_ratio)), 0.35, 2.5)",
        "gate  = z^2 / (z^2 + 4),   z = |mean(log r, last 8 slots)| / (sd(log r, last 32 slots) / sqrt(8))",
        "```",
        "",
        "- `F_base`: production Candidate F (profile_ratio HGB, refit every cycle) -- the same base Candidate G corrects.",
        "- `r = actual / base` on an **out-of-sample history** rebuilt from observations only: the resolved slots of the last 24 hourly cycles, each predicted by F refit once per UTC day on `target_at <= floor(history_cutoff, 1 day)`. No dependency on served predictions, no cold start after restarts.",
        "- `level_ratio`: the G2 estimator (geomspace(0.2, 1) weighted mean of the last 8 ratios).",
        "- The gate is the whole difference with G: a persistent residual shift (high SNR) is corrected like G does; residual noise is not (factor -> 1), which removes G's no-drift cost.",
        "- Fail-safe: any adaptive-layer failure or missing signal -> factor 1.0 (serves Candidate F).",
        "- Code: `src/candidate_i_snr_gate.py` (self-contained, frozen parameters validated from `artifacts/shadow/candidate_i_v1/metadata.json`).",
        "",
        "## Protocol",
        "",
        "- Every cycle replayed hourly with the exact production code paths (F parity with `build_candidate_f_base_frame`: max diff 0.0). G replayed with production `g2_factor` fed every hour (steady-state feedback -- favourable to G).",
        "- Windows fixed before any result: stable 09-03 00:00 -> 09-10 17:00 (186 cycles), drift_dev 09-10 18:00 -> 09-13 11:00 (66), drift_test 09-13 12:00 -> 09-15 03:00 (40, **sealed until selection**).",
        f"- Selection rule: {sel['rule']}.",
        f"- Selection recorded at {sel['recorded_at_utc']} before the test reveal: top-3 {sel['top3_by_J']}; {sel['pairwise_bootstrap']}; eliminated {sel['eliminated']}; selected `{sel['selected']}` ({sel['selected_reason']}).",
        "",
        "## Final comparison",
        "",
        markdown_table(table, 3),
        "",
        "Historical = all cycles from 09-03 (292). Recent12/Recent6 = last 12/6 official cycles. Drift regime = drift_dev + drift_test. Worst station = lowest per-station accuracy (historical). Runtime measured on this laptop under load (median of 6 cycles).",
        "",
        "## Deltas vs G (per cycle, pp)",
        "",
        markdown_table(_delta_rows(summary), 3),
        "",
        "## Station deltas (I - G, pp)",
        "",
        markdown_table(stations, 3),
        "",
        "## Horizons (pooled station WAPE accuracy)",
        "",
        markdown_table(horizons, 3),
        "",
        "## Ablations (delta vs I, pp; negative = the removed part helps)",
        "",
        markdown_table(abl, 3),
        "",
        "- `I + shape` (the co-top candidate LSg) was tied with I at selection (P=0.56) and scores higher on the test; switching after the reveal would use the test for selection, so it is the next shadow experiment, not the challenger.",
        "",
    ]
    if stress is not None:
        s = stress.rename(columns={"model": "policy"}).copy()
        s["policy"] = s["policy"].str.replace("pol::", "", regex=False)
        s = s.set_index(["scenario", "policy"])[["post_all", "post_affected", "post_unaffected", "first12h_affected"]]
        s.index = [f"{a} / {b}" for a, b in s.index]
        s.index.name = "scenario / model"
        lines += [
            "## Synthetic stress (drift injected into observations, bases re-replayed; 4 affected stations from 09-04 12:00)",
            "",
            "All policies correct the same base (F refit every 6 h, `pred_F6h`); signals from the daily-refit out-of-sample history. `L:g2|snr4` is Candidate I.",
            "",
            markdown_table(s, 3),
            "",
        ]
    lines += [
        "## Leakage audit",
        "",
        f"- Predictions audited: {leak['predictions_audited']}; latest timestamp used > data_cutoff: **{leak['violations_max_timestamp_gt_cutoff']}**.",
        f"- Signals with max_source_at > cutoff: {leak['signal_max_source_at_gt_cutoff']}; history models fit after their history cycle: {leak['history_model_trained_after_history_cycle']}; history slots that were not forecasts: {leak['history_slot_not_a_forecast']}.",
        "- Production module refuses observations after data_cutoff; unit test asserts slots after the cutoff never change the factor.",
        "",
        "## Production parity (src/candidate_i_snr_gate.py vs lab path)",
        "",
        markdown_table(parity, 6),
        "",
        "## Verification of the Candidate G lab windows",
        "",
        markdown_table(glab, 3),
        "",
        "- B served and F reproduce the reported 74.14/74.56 (recent12) and 73.40/73.93 (recent6); G reproduces 83.39 on recent6 and scores 83.47 (vs reported 82.86) on recent12 here because this replay gives G steady-state feedback from the start of the window.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/candidate_i_lab.py snapshot && python3 scripts/candidate_i_lab.py parity",
        "python3 scripts/candidate_i_lab.py bases && python3 scripts/candidate_i_lab.py variant F24h --cadence 24",
        "python3 scripts/candidate_i_lab.py explore --analysis --meta",
        "python3 scripts/candidate_i_lab.py stress --policies 'META:huber|win7.0' 'LSg:g2|snr4+shape3|snr16' 'L:g2|snr4' 'L:ros4|snr4'",
        "python3 scripts/candidate_i_lab.py parity-i",
        "python3 scripts/candidate_i_lab.py final --challenger 'L:g2|snr4' && python3 scripts/candidate_i_summary.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    fr = pd.read_pickle(core.INTERIM_DIR / "final_frame.pkl")
    parity = json.loads((core.INTERIM_DIR / "parity_candidate_i.json").read_text())
    t_i = np.median([p["runtime_candidate_i_s"] for p in parity])
    t_f = np.median([p["runtime_one_f_refit_s"] for p in parity])
    runtime = {
        "F": f"{t_f:.0f}s",
        "B (walk-forward)": f"~{t_f:.0f}s",
        "B (served)": "static artifact",
        "G": f"{t_f:.0f}s x (1 + served cycles in feedback, <=41)",
        "I": f"{t_i:.0f}s",
    }
    table = final_table(fr, runtime)
    stations = station_deltas(fr)
    horizons = horizon_deltas(fr)
    glab = glab_verification(fr)
    stress = pd.read_csv(core.INTERIM_DIR / "stress_board.csv") if (core.INTERIM_DIR / "stress_board.csv").exists() else None
    summary = json.loads((REPORT_DIR / "summary.json").read_text())
    summary["final_table"] = json.loads(table.to_json(orient="index"))
    summary["station_deltas_I_minus_G"] = json.loads(stations.to_json(orient="index"))
    summary["horizon_results"] = json.loads(horizons.to_json(orient="index"))
    summary["glab_window_verification"] = json.loads(glab.to_json(orient="index"))
    summary["parity_production_vs_lab"] = parity
    summary["selection_record"] = json.loads((core.INTERIM_DIR / "selection_record.json").read_text())
    if stress is not None:
        summary["synthetic_stress"] = json.loads(stress.to_json(orient="records"))
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(summary, table, stations, horizons, glab, stress))
    print(table.round(3).to_string())
    print(stations.round(3).to_string())
    print(horizons.round(3).to_string())
    print(glab.round(3).to_string())


if __name__ == "__main__":
    main()
