"""Render `reports/candidate_k/latest.md` from the experiment tables (read-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "candidate_k"
PHASE_NAMES = {"stable": "stable", "phase1": "phase 1", "phase2": "phase 2", "phase3_shock": "shock (before J freeze)", "oos_shock": "shock (after J freeze)"}


def _table(frame: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def render(report_dir: Path) -> str:
    summary = json.loads((report_dir / "summary.json").read_text())
    tables = report_dir / "tables"
    overall = pd.read_csv(tables / "overall.csv")
    wins = pd.read_csv(tables / "wins.csv")
    horizons = pd.read_csv(tables / "horizons.csv")
    stations = pd.read_csv(tables / "stations.csv", dtype={"station_id": str})
    primary = summary["primary_variant"]
    decision = summary["decisions"][primary]
    known = ["G", "J", *summary["variants"]]
    models = [*known, *sorted(m for m in set(overall["model"]) if m not in known)]
    phases = [p for p in PHASE_NAMES if p in set(overall["phase"])]

    acc = overall.pivot(index="model", columns="phase", values="mean_station_accuracy").loc[models, phases]
    acc.columns = [PHASE_NAMES[p] for p in phases]
    acc = acc.reset_index()
    gwape = overall.pivot(index="model", columns="phase", values="global_wape_accuracy").loc[models, phases]
    gwape.columns = [PHASE_NAMES[p] for p in phases]
    gwape = gwape.reset_index()
    bias = overall.pivot(index="model", columns="phase", values="relative_bias").loc[models, phases].mul(100)
    bias.columns = [PHASE_NAMES[p] for p in phases]
    bias = bias.reset_index()
    cycles = overall.drop_duplicates("phase").set_index("phase")["cycles"]

    w = wins[wins["model"] == primary].copy()
    w["phase"] = w["phase"].map(PHASE_NAMES)
    hz = horizons.assign(phase=horizons["phase"].map(PHASE_NAMES))[["phase", "horizon_minutes", "G", "J", primary]]
    hz[f"{primary} - J"] = hz[primary] - hz["J"]
    st = stations.assign(delta=stations[primary] - stations["J"])
    worst_normal = st[st["phase"].isin(["stable", "phase1", "phase2"])].nsmallest(5, "delta")[["phase", "station_id", "J", primary, "delta"]]
    shock_st = st[st["phase"] == "oos_shock"][["station_id", "G", "J", primary, "delta"]].sort_values("station_id")

    pooled = pd.DataFrame(summary["pooled"])
    pooled_table = pooled.pivot(index="model", columns="window", values="mean_station_accuracy").loc[models, ["all_cycles", "last_7d", "last_24h", "normal_phases"]].reset_index()
    pooled_by = pooled.set_index(["window", "model"])["mean_station_accuracy"]
    deltas = decision["delta_pp_by_phase"]
    lines = [
        "# Candidate K: regime-robust deep forecaster",
        "",
        f"Generated {summary['generated_at']}. Research only: nothing here writes to Supabase, submits, or touches Predict or `model_versions`.",
        "",
        "## Verdict",
        "",
        f"- **Primary variant:** `{primary}` (K v2.1, pure deep learning). K's design freeze is {summary['protocol']['k_design_freeze']} (virtual time). Only cycles after it are a design-blind test.",
        f"- **Pre-registered verdict vs J (current champion): `{decision['verdict']}`.** Checks: " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in decision["checks"].items()) + ".",
        "- **Delta vs J by phase (pp of `mean_station_accuracy`):** " + ", ".join(f"{PHASE_NAMES[p]} {v:+.2f}" for p, v in deltas.items()) + ".",
        f"  - Worst per-station delta in normal phases: {decision['worst_station_delta_pp_normal']:+.2f} pp.",
        f"- **Leaderboard metric over pooled windows:**",
        f"  - all {int(pooled.loc[pooled['window'] == 'all_cycles', 'cycles'].iloc[0])} cycles: K {pooled_by[('all_cycles', primary)]:.2f} vs J {pooled_by[('all_cycles', 'J')]:.2f};",
        f"  - last 7 days: K {pooled_by[('last_7d', primary)]:.2f} vs J {pooled_by[('last_7d', 'J')]:.2f};",
        f"  - last 24 h: K {pooled_by[('last_24h', primary)]:.2f} vs J {pooled_by[('last_24h', 'J')]:.2f};",
        f"  - normal phases only: K {pooled_by[('normal_phases', primary)]:.2f} vs J {pooled_by[('normal_phases', 'J')]:.2f}.",
        "- **Recommendation (judgment, separate from the pre-registered rule):** do not promote. If approved, run K in shadow and evaluate it on cycles after its freeze.",
        "  - K roughly doubles J's accuracy in the shock and wins on every pooled window that contains it.",
        "  - It is still 0.5-1 pp below J in stable and phase-1 regimes, which the pre-registered tolerance does not allow.",
        "",
        "## Iterations (all reported, none hidden)",
        "",
        *[f"- {line}" for line in summary["protocol"]["k_iterations"]],
        "- Each iteration was informed by results on the same evaluation window. That is why K is frozen at v2.1 and the next evidence must come from new cycles.",
        "",
        "## What K is",
        "",
        "- **K v2.1 (`k_staged_re`) has three networks:**",
        "  1. A *seasonal expert*: a TiDE-style residual MLP over the 12 h history, the 7-day profile, the 4-week baseline, yesterday's value, the calendar and a station embedding. It is trained on real data at every refit.",
        "  2. A *local-dynamics expert*: RevIN plus a residual MLP over the last 8 h.",
        "     - It is pre-trained **only on synthetic series** (sinusoids with random periods between 45 min and 32 h, trends, level shifts and regime switches) and then frozen.",
        "     - It never sees an observation, so there is nothing to leak.",
        "  3. A per-horizon *gate* that mixes the two. It sees regime evidence: recent-vs-profile ratios, volatility, backtests of the local expert over the last 1-4 h, and the realized error of the seasonal expert's own forecasts issued 1 h and 2 h earlier.",
        "     - The gate value is K's regime trace.",
        "- **Scale and loss:** every demand channel is divided by the station's trailing 7-day mean, and the loss is L1 in that space, which approximates the per-station WAPE behind the leaderboard.",
        "- **Gate training:** real samples, plus 35% of samples whose recent tail and targets are replaced by a synthetic regime that starts 4-12 h before the cutoff.",
        "  - The seasonal references stay unchanged, as they would at the start of a real regime change.",
        "  - The v1 variants used a simpler augmentation (level shifts, 1-16 h oscillations, warps, ramps) inside a single end-to-end network.",
        f"- **Ensemble:** {summary['protocol']['seeds']} seeds, median.",
        f"- **Walk-forward:** {summary['protocol']['refit']}. Training uses one origin every {summary['protocol']['train_stride_slots'] * 15} min. Every input is <= the cycle `data_cutoff`, and K never trains on a target it is scored on.",
        "- **Comparison rows:** G and J are scored on exactly the same rows.",
        "  - Before the J design freeze, the rows come from the J replay.",
        "  - After the freeze, they are the official production predictions.",
        "",
        "## Pooled windows (`mean_station_accuracy`)",
        "",
        _table(pooled_table),
        "",
        "## Accuracy per phase (`mean_station_accuracy`, leaderboard metric)",
        "",
        "Cycles: " + ", ".join(f"{PHASE_NAMES[p]} {int(cycles[p])}" for p in phases) + ".",
        "",
        _table(acc),
        "",
        "Global WAPE accuracy:",
        "",
        _table(gwape),
        "",
        "Relative bias, % (positive means under-forecast):",
        "",
        _table(bias),
        "",
        "## Wins and losses per cycle",
        "",
        _table(w[["vs", "phase", "win", "loss", "tie"]]),
        "",
        "## Per horizon",
        "",
        _table(hz),
        "",
        "## Stations",
        "",
        "Largest per-station losses vs J in the normal phases:",
        "",
        _table(worst_normal),
        "",
        "Shock after the J freeze, per station:",
        "",
        _table(shock_st),
        "",
        "## Caveats (read before acting)",
        "",
        "1. **Not design-blind for K.** The architecture and the augmentation were chosen after the shock was inspected up to the latest observation, which shows a near-periodic oscillation of about 4 h.",
        "   - The post-freeze shock cycles are walk-forward for K, with no training leakage. They are not a design-blind test.",
        "   - The only clean test is cycles after K's own design freeze.",
        f"2. **Few shock cycles:** {int(cycles.get('phase3_shock', 0))} before the J freeze and {int(cycles.get('oos_shock', 0))} after it. There is no recovery yet, so the 12-hour and recovery requirements of the J protocol are not met for K either.",
        "3. **No production path yet.** K needs PyTorch, which is not a dependency of Predict or the shadow workflows. Deploying it in shadow needs either a CPU torch install in a separate workflow, or a numpy forward pass over exported weights. Both need explicit approval.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/candidate_k_experiment.py --variants " + ",".join(summary["variants"]) + f" --seeds {summary['protocol']['seeds']}",
        "python scripts/candidate_k_figures.py && python scripts/candidate_k_report.py",
        "```",
        "",
        "Snapshot: `data/interim/candidate_k/snapshot/manifest.json` (Supabase read-only GET, sha256 per table).",
    ]
    figures = report_dir / "figures"
    if figures.exists():
        lines += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted(figures.glob("*.png"))]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    (args.report_dir / "latest.md").write_text(render(args.report_dir))
    print(args.report_dir / "latest.md")


if __name__ == "__main__":
    main()
