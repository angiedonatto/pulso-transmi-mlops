"""Candidate I -- final comparison report (B, F, G, challenger) -> reports/candidate_i/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import candidate_i_eval as ev
from candidate_i_core import ROOT, cycle_scores

REPORT_DIR = ROOT / "reports" / "candidate_i"
WINDOW_ORDER = ["stable", "drift_dev", "drift_test", "drift_all", "official", "recent12", "recent6", "glab12", "all_scored"]


def comparison_payload(frame: pd.DataFrame, models: dict[str, str], challenger: str, reference: str = "G") -> dict[str, Any]:
    cols = list(models.values())
    scores = ev.window_scores(frame, cols, WINDOW_ORDER)
    scores.index = [next(k for k, v in models.items() if v == idx) for idx in scores.index]
    ch, ref = models[challenger], models[reference]
    deltas = {w: ev.delta_summary(frame, ch, ref, w) for w in WINDOW_ORDER}
    boot = {w: ev.block_bootstrap_p_better(frame, ch, ref, w) for w in ("drift_test", "official", "drift_all", "stable")}
    masks = ev.window_masks(frame)
    stations = {w: ev.station_table(frame, cols, masks[w]) for w in ("stable", "drift_test", "drift_all")}
    horizons = {w: ev.horizon_table(frame, cols, masks[w]) for w in ("stable", "drift_test", "drift_all")}
    for tables in (stations, horizons):
        for w, t in tables.items():
            t.columns = [next(k for k, v in models.items() if v == c) for c in t.columns]
    return {"scores": scores, "deltas": deltas, "bootstrap": boot, "stations": stations, "horizons": horizons}


def write_report(frame: pd.DataFrame, models: dict[str, str], challenger: str, payload: dict[str, Any], extra: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_frames = []
    for name, col in models.items():
        cs = cycle_scores(frame, col).rename(columns={"cycle_accuracy": name}).set_index(["cycle_id", "data_cutoff"])
        cycle_frames.append(cs)
    cycles = pd.concat(cycle_frames, axis=1).reset_index().sort_values("data_cutoff")
    masks = ev.window_masks(frame)
    for w in ("stable", "drift_dev", "drift_test", "official", "recent12", "recent6"):
        cut = set(frame.loc[masks[w], "data_cutoff"].unique())
        cycles[f"in_{w}"] = cycles["data_cutoff"].isin(cut)
    cycles[f"delta_{challenger}_vs_G"] = cycles[challenger] - cycles["G"]
    cycles.to_csv(REPORT_DIR / "cycle_metrics.csv", index=False)

    st = []
    for w, t in payload["stations"].items():
        st.append(t.assign(window=w).reset_index())
    pd.concat(st, ignore_index=True).to_csv(REPORT_DIR / "station_metrics.csv", index=False)
    hz = []
    for w, t in payload["horizons"].items():
        hz.append(t.assign(window=w).reset_index())
    pd.concat(hz, ignore_index=True).to_csv(REPORT_DIR / "horizon_metrics.csv", index=False)

    keep = ["cycle_id", "data_cutoff", "station_id", "target_at", "horizon_minutes", "actual"] + list(models.values())
    out = frame[keep].rename(columns={v: f"pred_{k}" for k, v in models.items()})
    out.to_csv(REPORT_DIR / "predictions.csv", index=False)

    summary = {
        "challenger": challenger,
        "scores": json.loads(payload["scores"].to_json(orient="index")),
        "deltas_vs_G": payload["deltas"],
        "bootstrap_vs_G": payload["bootstrap"],
        **extra,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records"))
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}" if np.isfinite(v) else ""
        return str(v).replace("|", "\\|")

    cols = [cell(c) for c in frame.columns]
    lines = ["| " + " | ".join([cell(frame.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, *vals in frame.itertuples(name=None):  # per-column dtypes (ints stay ints)
        lines.append("| " + " | ".join([cell(idx)] + [cell(v) for v in vals]) + " |")
    return "\n".join(lines)
