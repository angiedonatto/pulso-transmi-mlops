"""Static figures for `scripts/drift_regime_analysis.py` (matplotlib, Agg backend).

Every figure is backed by a CSV under `reports/drift_regime_analysis/tables/` (its table
view). Colors follow a validated categorical order (blue, orange, aqua, yellow) with marker
shapes as secondary encoding; change maps use a blue <-> gray <-> red diverging scale where
red means "more than reference / under-predicted" and blue the opposite.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import drift_regime as dr  # noqa: E402


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a19f", "#e34948", "#9e2423"]
)
WINDOW_ORDER = ["last_48h", "last_24h", "last_12h", "last_6h"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
    }
)


def render_all(out: Path, obs: dict[str, Any], pred: dict[str, Any], matrix: pd.DataFrame) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "01_demand_before_after_by_station.png": lambda p: demand_before_after(p, obs),
        "02_hourly_profile_reference_vs_recent.png": lambda p: profile_reference_vs_recent(p, obs),
        "03_heatmap_station_hour_change.png": lambda p: heatmap_station_hour(p, obs),
        "03b_heatmap_station_timeline.png": lambda p: heatmap_station_timeline(p, obs),
        "04_g_residual_station_time.png": lambda p: g_residual_heatmap(p, pred),
        "05_g_accuracy_by_station.png": lambda p: station_dotplot(p, pred["by_station"], "accuracy", "Candidate G accuracy (%)", "Candidate G accuracy by station and trailing window"),
        "06_g_bias_by_station.png": lambda p: station_dotplot(
            p, pred["by_station"].assign(relative_bias=pred["by_station"]["relative_bias"] * 100), "relative_bias",
            "Relative bias = sum(actual - prediction) / sum(actual), %   (+ under-predicts, - over-predicts)",
            "Candidate G signed bias by station", zero_line=True,
        ),
        "07_g_accuracy_by_horizon.png": lambda p: accuracy_by_horizon(p, pred),
        "08_level_ratio_by_station.png": lambda p: level_ratio(p, obs),
        "09_lag_relationships.png": lambda p: lag_relationships(p, obs),
        "10_drift_timeline.png": lambda p: drift_timeline(p, obs, pred),
    }
    if pred["decomposition"].get("status") == "ok":
        figures["11_g_base_vs_calibration.png"] = lambda p: g_decomposition(p, pred)
    for key, comparison in pred["challengers"].items():
        if comparison.get("status") == "ok":
            slug = key.split("|")[0]
            figures[f"12_{slug}_vs_candidate_g.png"] = lambda p, c=comparison: challenger(p, c)
    written = {}
    for name, draw in figures.items():
        path = out / name
        draw(path)
        written[name] = str(path)
    return written


# --------------------------------------------------------------------------------------


def demand_before_after(path: Path, obs: dict[str, Any]) -> None:
    prepared = obs["prepared"]
    anchor = obs["anchor"]
    start = anchor - pd.Timedelta(days=10)
    rows = dr.attach_reference(prepared[prepared["observed_at"] > start], obs["profile"])
    hourly = rows.set_index("observed_at").groupby("station_id")[["demand", "ref"]].resample("1h").sum(min_count=4).reset_index()
    hourly["local"] = hourly["observed_at"].dt.tz_convert(dr.BOGOTA_TZ).dt.tz_localize(None)
    change_points = _global_change_points(obs)
    stations = sorted(hourly["station_id"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for ax, station in zip(axes.flat, stations):
        data = hourly[hourly["station_id"] == station]
        ax.plot(data["local"], data["ref"], color=MUTED, linewidth=1.1, label="Reference expectation (same weekday & slot)")
        ax.plot(data["local"], data["demand"], color=SERIES[0], linewidth=1.2, label="Observed")
        for cp in change_points:
            ax.axvline(cp, color=INK_2, linewidth=0.7)
        ax.set_title(station, loc="left")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    _legend_once(fig, axes.flat[0])
    fig.suptitle("Hourly demand per station, last 10 days vs reference expectation (vertical lines = detected global change points, Bogota time)", color=INK, fontsize=11, x=0.01, ha="left")
    _save(fig, path)


def profile_reference_vs_recent(path: Path, obs: dict[str, Any]) -> None:
    prepared = obs["prepared"]
    profile = obs["profile"]
    weekday_ref = profile[profile["dow"] < 5].groupby(["station_id", "slot"])["ref"].median().reset_index()
    dates = sorted(prepared["local_date"].unique())[-3:]
    stations = sorted(prepared["station_id"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for ax, station in zip(axes.flat, stations):
        ref = weekday_ref[weekday_ref["station_id"] == station]
        ax.plot(ref["slot"] / 4, ref["ref"], color=MUTED, linewidth=1.4, label="Reference weekday median")
        for i, date in enumerate(dates):
            day = prepared[(prepared["station_id"] == station) & (prepared["local_date"] == date)].sort_values("slot")
            complete = len(day) == 96
            ax.plot(day["slot"] / 4, day["demand"], color=SERIES[i], linewidth=1.2, label=f"{date}{'' if complete else ' (partial)'}")
        ax.set_title(station, loc="left")
        ax.set_xticks(range(0, 25, 4))
    for ax in axes[-1]:
        ax.set_xlabel("Local hour (America/Bogota)")
    _legend_once(fig, axes.flat[0])
    fig.suptitle("Intra-day profile: reference weekday vs the most recent local days", color=INK, fontsize=11, x=0.01, ha="left")
    _save(fig, path)


def heatmap_station_hour(path: Path, obs: dict[str, Any]) -> None:
    change = obs["hour_change"]
    windows = [w for w in ("last_48h", "last_6h") if w in set(change["window"])]
    fig, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 5.2), gridspec_kw={"width_ratios": [24, 9][: len(windows)]} if len(windows) == 2 else None)
    axes = np.atleast_1d(axes)
    image = None
    for ax, window in zip(axes, windows):
        data = change[change["window"] == window].pivot(index="station_id", columns="local_hour", values="log2_ratio")
        first = next(w for w in obs["windows"] if w.name == window).start + dr.SLOT
        first_hour = first.tz_convert(dr.BOGOTA_TZ).hour
        hours = sorted(data.columns, key=lambda h: (h - first_hour) % 24) if window != "last_48h" else sorted(data.columns)
        data = data[hours]
        image = ax.imshow(data.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=3.0), aspect="auto")
        ax.set_yticks(range(len(data.index)), data.index)
        ax.set_xticks(range(len(hours)), hours)
        ax.set_xlabel("Local hour")
        ax.set_title(f"{window}: log2(observed / reference)", loc="left")
        ax.grid(False)
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("log2 ratio  (+1 = 2x reference, -1 = half)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle("Relative change by station x hour of day", color=INK, fontsize=11, x=0.01, ha="left")
    _save(fig, path, tight=False)


def heatmap_station_timeline(path: Path, obs: dict[str, Any]) -> None:
    lr = obs["station_lr"]
    lr = lr[lr.index > obs["anchor"] - pd.Timedelta(days=4)]
    data = (lr / math.log(2)).T
    fig, ax = plt.subplots(figsize=(14, 4.8))
    image = ax.imshow(data.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=3.0), aspect="auto", interpolation="nearest")
    local = lr.index.tz_convert(dr.BOGOTA_TZ)
    ticks = [i for i, ts in enumerate(local) if ts.hour % 6 == 0]
    ax.set_xticks(ticks, [local[i].strftime("%d %Hh") for i in ticks], rotation=0)
    ax.set_yticks(range(len(data.index)), data.index)
    ax.grid(False)
    for cp in _global_change_points(obs, naive=False):
        if cp >= lr.index.min():
            ax.axvline(lr.index.get_indexer([cp], method="nearest")[0] - 0.5, color=INK, linewidth=0.9)
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("log2(observed / causal 4-week baseline)", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Regime evolution, last 4 days (hourly, Bogota time; black lines = global change points)", loc="left")
    _save(fig, path)


def g_residual_heatmap(path: Path, pred: dict[str, Any]) -> None:
    g = pred["g"]
    per = g.groupby(["station_id", "data_cutoff"]).agg(residual=("residual", "sum"), actual=("actual", "sum")).reset_index()
    per["relative_residual"] = per["residual"] / per["actual"]
    data = per.pivot(index="station_id", columns="data_cutoff", values="relative_residual")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    image = ax.imshow(data.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5), aspect="auto", interpolation="nearest")
    local = pd.DatetimeIndex(data.columns).tz_convert(dr.BOGOTA_TZ)
    ticks = [i for i, ts in enumerate(local) if ts.hour % 6 == 0]
    ax.set_xticks(ticks, [local[i].strftime("%d %Hh") for i in ticks])
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set_xlabel("Cycle data_cutoff (Bogota time; columns are consecutive resolved cycles, gaps not drawn)")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("sum(actual - prediction) / sum(actual)\nred = under-prediction, blue = over-prediction", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Candidate G residual by station and cycle", loc="left")
    _save(fig, path)


def station_dotplot(path: Path, frame: pd.DataFrame, value: str, xlabel: str, title: str, *, zero_line: bool = False) -> None:
    windows = [w for w in WINDOW_ORDER if w in set(frame["window"])]
    stations = sorted(frame["station_id"].unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    offsets = np.linspace(-0.27, 0.27, len(windows))
    for i, window in enumerate(windows):
        data = frame[frame["window"] == window].set_index("station_id").reindex(stations)
        ax.scatter(data[value], np.arange(len(stations)) + offsets[i], color=SERIES[i], marker=MARKERS[i], s=34, label=window, zorder=3, edgecolors=SURFACE, linewidths=1.0)
    if zero_line:
        ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(range(len(stations)), stations)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(windows))
    _save(fig, path)


def accuracy_by_horizon(path: Path, pred: dict[str, Any]) -> None:
    frame = pred["by_horizon"]
    windows = [w for w in WINDOW_ORDER if w in set(frame["window"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, window in enumerate(windows):
        data = frame[frame["window"] == window].sort_values("horizon_minutes")
        ax.plot(data["horizon_minutes"], data["accuracy"], color=SERIES[i], marker=MARKERS[i], markersize=6, label=window, markeredgecolor=SURFACE)
    ax.set_xticks([15, 30, 45, 60])
    ax.set_xlabel("Horizon (minutes after data_cutoff)")
    ax.set_ylabel("Pooled accuracy (%)")
    ax.set_title("Candidate G accuracy by horizon", loc="left", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(windows))
    _save(fig, path)


def level_ratio(path: Path, obs: dict[str, Any]) -> None:
    metrics = obs["station_metrics"]
    trailing = metrics[metrics["window"].isin(WINDOW_ORDER)]
    stations = sorted(trailing["station_id"].unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    null = trailing[trailing["window"] == "last_48h"].set_index("station_id").reindex(stations)
    if "null_log_level_ratio_min" in null:
        for y, (lo, hi) in enumerate(zip(np.exp(null["null_log_level_ratio_min"]), np.exp(null["null_log_level_ratio_max"]))):
            ax.plot([lo, hi], [y, y], color=GRID, linewidth=7, solid_capstyle="round", zorder=1, label="Stable-period range (48h same-clock windows)" if y == 0 else None)
    windows = [w for w in WINDOW_ORDER if w in set(trailing["window"])]
    offsets = np.linspace(-0.27, 0.27, len(windows))
    for i, window in enumerate(windows):
        data = trailing[trailing["window"] == window].set_index("station_id").reindex(stations)
        ax.scatter(data["level_ratio"], np.arange(len(stations)) + offsets[i], color=SERIES[i], marker=MARKERS[i], s=34, label=window, zorder=3, edgecolors=SURFACE, linewidths=1.0)
    ax.axvline(1.0, color=INK_2, linewidth=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_yticks(range(len(stations)), stations)
    ax.invert_yaxis()
    ax.set_xlabel("Observed / reference demand (same weekday & slots), log scale")
    ax.set_title("Level ratio recent / reference by station", loc="left", pad=44)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    _save(fig, path)


def lag_relationships(path: Path, obs: dict[str, Any]) -> None:
    lags = obs["lag_table"]
    order = list(dr.LAGS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    pooled = lags[lags["station_id"] == "ALL"]
    reference = pooled[pooled["window"] == "reference"].set_index("lag").reindex(order)
    ax1.plot(order, reference["lag_corr"], color=MUTED, marker="o", markersize=6, label="reference (stable period)")
    for i, window in enumerate(["last_48h", "last_6h"]):
        data = pooled[pooled["window"] == window].set_index("lag").reindex(order)
        ax1.plot(order, data["lag_corr"], color=SERIES[i], marker=MARKERS[i], markersize=6, label=window, markeredgecolor=SURFACE)
    ax1.set_ylabel("corr(log1p y_t, log1p y_t-k), all stations pooled")
    ax1.set_title("Lag correlation before vs after", loc="left")
    ax1.legend(loc="lower left")
    data = lags[(lags["window"] == "last_48h") & (lags["station_id"] != "ALL")].pivot(index="station_id", columns="lag", values="naive_accuracy_delta_pp")[order]
    image = ax2.imshow(data.to_numpy(), cmap=DIVERGING.reversed(), norm=TwoSlopeNorm(vcenter=0.0, vmin=-40, vmax=40), aspect="auto")
    ax2.set_xticks(range(len(order)), order)
    ax2.set_yticks(range(len(data.index)), data.index)
    ax2.grid(False)
    ax2.set_title("last_48h: naive-lag accuracy minus stable-period median (pp)", loc="left")
    cbar = fig.colorbar(image, ax=ax2, pad=0.02)
    cbar.set_label("pp  (red = lag explains demand worse than in the stable period)", color=INK_2)
    cbar.outline.set_visible(False)
    _save(fig, path)


def drift_timeline(path: Path, obs: dict[str, Any], pred: dict[str, Any]) -> None:
    indices = obs["indices"]
    start = obs["reference"].start
    indices = indices[indices.index > start]
    local = indices.index.tz_convert(dr.BOGOTA_TZ).tz_localize(None)
    segments = obs["segments"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1.1]})
    for ax, name, label in ((axes[0], "global_intensity", "mean |log ratio| across stations"), (axes[1], "global_level", "log ratio of total demand")):
        ax.plot(local, indices[name], color=SERIES[0], linewidth=1.0, label="hourly index")
        seg = segments[segments["series"] == name]
        for row in seg.itertuples():
            ax.plot([row.start.tz_convert(dr.BOGOTA_TZ).tz_localize(None), row.end.tz_convert(dr.BOGOTA_TZ).tz_localize(None)], [row.mean, row.mean], color=SERIES[1], linewidth=2.2, label="segment mean" if row.segment == 0 else None)
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
    for ax in axes:
        for cp in _global_change_points(obs):
            ax.axvline(cp, color=INK_2, linewidth=0.7)
    axes[0].set_title("Drift timeline: causal drift indices with detected change points, and official accuracy per cycle", loc="left")
    cycles = pred["cycles"]
    official = cycles[cycles["is_official"]].copy()
    official["local"] = official["data_cutoff"].dt.tz_convert(dr.BOGOTA_TZ).dt.tz_localize(None)
    for i, (name, data) in enumerate(official.groupby("model_name", sort=False)):
        axes[2].plot(data["local"], data["accuracy"], color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)], markersize=4, linewidth=1.0, label=name, markeredgecolor=SURFACE)
    worst = official.loc[official["accuracy"].idxmin()] if not official.empty else None
    if worst is not None:
        axes[2].annotate(f"{worst['accuracy']:.1f}% (cutoff {worst['local']:%d %H:%M})", (worst["local"], worst["accuracy"]), xytext=(-150, 10), textcoords="offset points", color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[2].set_ylabel("Pooled accuracy per cycle (%)")
    axes[2].set_ylim(bottom=-2)
    axes[2].legend(loc="lower left")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    _save(fig, path)


def g_decomposition(path: Path, pred: dict[str, Any]) -> None:
    per = pred["decomposition"]["per_cycle"].copy()
    per["local"] = per["data_cutoff"].dt.tz_convert(dr.BOGOTA_TZ).dt.tz_localize(None)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax1.plot(per["local"], per["base_accuracy"], color=SERIES[0], marker="o", markersize=4, linewidth=1.0, label="Candidate F base (no G2 factor)", markeredgecolor=SURFACE)
    ax1.plot(per["local"], per["g_accuracy"], color=SERIES[1], marker="s", markersize=4, linewidth=1.0, label="Candidate G = base x G2 factor (official)", markeredgecolor=SURFACE)
    ax1.set_ylabel("Pooled accuracy per cycle (%)")
    ax1.set_ylim(bottom=-2)
    ax1.legend(loc="lower left")
    ax1.set_title("Where does Candidate G's error come from? Base model vs G2 calibration (base refit per cycle with data <= data_cutoff)", loc="left")
    ax2.bar(per["local"], per["share_factor_clipped"] * 100, width=0.03, color=SERIES[0])
    ax2.set_ylabel("% rows with G2 factor at clip (0.35 or 2.5)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    _save(fig, path)


def challenger(path: Path, comparison: dict[str, Any]) -> None:
    per = comparison["per_cycle"].copy()
    per["local"] = per["data_cutoff"].dt.tz_convert(dr.BOGOTA_TZ).dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(per["local"], per["champion_accuracy"], color=SERIES[0], marker="o", markersize=6, linewidth=1.0, label="Candidate G (official)", markeredgecolor=SURFACE)
    ax.plot(per["local"], per["challenger_accuracy"], color=SERIES[1], marker="s", markersize=6, linewidth=1.0, label=f"{comparison['model_name']} (shadow)", markeredgecolor=SURFACE)
    ax.set_ylabel("Pooled accuracy per cycle (%)")
    ax.set_ylim(bottom=-2)
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    ax.set_title(f"Same-cycle comparison only ({comparison['shared_cycles']} shared resolved cycles)", loc="left")
    _save(fig, path)


# --------------------------------------------------------------------------------------


def _global_change_points(obs: dict[str, Any], *, naive: bool = True) -> list[pd.Timestamp]:
    segments = obs["segments"]
    points = segments[(segments["series"] == "global_intensity") & (segments["segment"] > 0)]["start"].tolist()
    if naive:
        return [ts.tz_convert(dr.BOGOTA_TZ).tz_localize(None) for ts in points]
    return points


def _legend_once(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=len(labels))


def _save(fig: plt.Figure, path: Path, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.95) if fig._suptitle is not None else None)
    fig.savefig(path, dpi=130)
    plt.close(fig)
