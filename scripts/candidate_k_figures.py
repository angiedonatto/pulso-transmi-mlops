"""Figures for the Candidate K report, from `reports/candidate_k/tables` (read-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "candidate_k"
# Okabe-Ito (colorblind-safe); each model keeps its color in every figure
# charts show at most four series (G, J, primary K, K gate over J); other variants live in tables
COLORS = {"G": "#999999", "J": "#0072B2", "k_staged_re": "#D55E00", "k_staged_re_j": "#009E73", "k_staged": "#D55E00", "k_staged_j": "#009E73", "k_gated_aug": "#CC79A7"}
LABELS = {"G": "G (previous champion)", "J": "J v1.1 (champion)", "k_staged_re": "K (pure DL)", "k_staged_re_j": "K gate over J (hybrid)",
          "k_staged": "K staged v2 (pure DL)", "k_staged_j": "K v2 gate over J",
          "k_gated_aug": "K gated + aug (v1)", "k_tide_aug": "K TiDE + aug (v1)", "k_gated_noaug": "K gated, no aug (v1)"}
PHASE_LABELS = {"stable": "Stable", "phase1": "Phase 1", "phase2": "Phase 2", "phase3_shock": "Shock\n(pre-freeze J)", "oos_shock": "Shock\n(post-freeze J)"}
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def phase_bars(overall: pd.DataFrame, models: list[str], path: Path) -> None:
    phases = [p for p in PHASE_LABELS if p in set(overall["phase"])]
    table = overall.pivot(index="phase", columns="model", values="mean_station_accuracy").loc[phases, models]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        xs = [i + (k - (len(models) - 1) / 2) * width for i in range(len(phases))]
        bars = ax.bar(xs, table[model], width=width - 0.03, color=COLORS[model], label=LABELS[model])
        if model in ("J", models[-1], models[-2]):
            for x, value in zip(xs, table[model]):
                ax.text(x, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        del bars
    ax.set_xticks(range(len(phases)), [PHASE_LABELS[p] for p in phases], color=INK)
    ax.set_ylabel("mean_station_accuracy (leaderboard)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy per phase, same targets for every model", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shock_timeline(cycles: pd.DataFrame, models: list[str], path: Path, freeze: pd.Timestamp) -> None:
    part = cycles[cycles["phase"].isin(["phase2", "phase3_shock", "oos_shock"])].copy()
    part["data_cutoff"] = pd.to_datetime(part["data_cutoff"], utc=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for model in models:
        ax.plot(part["data_cutoff"], part[model], color=COLORS[model], linewidth=2, marker="o", markersize=3.5, label=LABELS[model])
    shock_start = part.loc[part["phase"] == "phase3_shock", "data_cutoff"].min()
    ax.axvspan(shock_start, part["data_cutoff"].max(), color="#f6f8fa", zorder=0)
    ax.axvline(freeze, color=MUTED, linewidth=1, linestyle="--")
    ax.text(freeze, 101, " J design freeze", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(0, 100)
    ax.set_ylabel("cycle accuracy", color=MUTED, fontsize=9)
    ax.set_title("Per-cycle accuracy: Phase 2 into the severe shock (shaded)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trajectories(predictions: pd.DataFrame, models: dict[str, str], stations: list[str], path: Path) -> None:
    part = predictions[(predictions["horizon_minutes"] == 60) & (predictions["data_cutoff"] >= "2026-09-17T18:00:00Z")]
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.6 * len(stations)), sharex=True)
    for ax, station in zip(axes, stations):
        rows = part[part["station_id"] == station].sort_values("target_at")
        ax.plot(rows["target_at"], rows["actual"], color=INK, linewidth=2.2, label="actual")
        for model, column in models.items():
            ax.plot(rows["target_at"], rows[column], color=COLORS[model], linewidth=1.6, label=LABELS[model])
        ax.set_title(f"Station {station} -- 60-min-ahead forecasts", loc="left", color=INK, fontsize=10)
        ax.set_ylabel("demand", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=len(models) + 1, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gate_trace(predictions: pd.DataFrame, column: str, path: Path) -> None:
    part = predictions[predictions["data_cutoff"] >= "2026-09-15T00:00:00Z"]
    gate = part.groupby("data_cutoff")[column].agg(["mean", "min", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(gate["data_cutoff"], gate["min"], gate["max"], color="#fbe3d6", linewidth=0, label="station range")
    ax.plot(gate["data_cutoff"], gate["mean"], color="#D55E00", linewidth=2, label="mean over stations")
    ax.set_ylim(0, 1)
    ax.set_ylabel("weight on seasonal expert", color=MUTED, fontsize=9)
    ax.set_title("K's learned regime gate (1 = seasonal profile, 0 = local dynamics)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--stations", default="02300,07105")
    args = parser.parse_args()
    tables = args.report_dir / "tables"
    figures = args.report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.report_dir / "summary.json").read_text())
    overall = pd.read_csv(tables / "overall.csv")
    cycles = pd.read_csv(tables / "cycles.csv")
    predictions = pd.read_csv(tables / "predictions.csv", dtype={"station_id": str}, parse_dates=["data_cutoff", "target_at"])
    variants = list(summary["variants"])
    primary = summary.get("primary_variant", variants[0])
    charted = [m for m in ["G", "J", primary, f"{primary}_j"] if m in set(overall["model"])]
    phase_bars(overall, charted, figures / "accuracy_by_phase.png")
    shock_timeline(cycles, charted, figures / "shock_timeline.png", pd.Timestamp(summary["protocol"]["j_design_freeze"]))
    trajectories(predictions, {m: ("g_prediction" if m == "G" else "j_prediction" if m == "J" else f"{m}_prediction") for m in charted}, args.stations.split(","), figures / "shock_trajectories.png")
    gate_column = f"{primary}_seasonal_weight"
    if gate_column in predictions and predictions[gate_column].notna().any():
        gate_trace(predictions, gate_column, figures / "regime_gate.png")
    print(sorted(p.name for p in figures.iterdir()))


if __name__ == "__main__":
    main()
