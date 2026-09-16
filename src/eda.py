from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pulso-transmi-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pulso-transmi-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_quality import dataset_metadata, read_raw_data, validate_data, write_quality_report
from src.features import BOGOTA_TZ, add_time_features


REPORT_DIR = Path("reports/eda")
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
PROCESSED_DIR = Path("data/processed")


def accuracy_by_station(frame: pd.DataFrame) -> pd.Series:
    grouped = frame.assign(abs_error=(frame["demand"] - frame["prediction"]).abs())
    wape = grouped.groupby("station_id")["abs_error"].sum() / grouped.groupby("station_id")["demand"].sum()
    return 100 * (1 - wape).clip(lower=0)


def prepare_dataset(
    stations: pd.DataFrame,
    observations: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    observations_features = add_time_features(observations)
    context_features = add_time_features(context)
    frame = observations_features.merge(stations, on="station_id", how="left")
    frame = frame.merge(
        context_features[
            [
                "observed_at",
                "rain_mm",
                "rain_forecast",
                "temperature_c",
                "temperature_forecast",
                "event_intensity",
            ]
        ],
        on="observed_at",
        how="left",
    )
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def evaluate_baselines(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = frame["observed_at"].max() - pd.DateOffset(days=7)
    train = frame.loc[frame["observed_at"] <= cutoff].copy()
    validation = frame.loc[frame["observed_at"] > cutoff].copy()

    lagged = frame[["station_id", "observed_at", "demand"]].copy()
    lagged["observed_at"] = lagged["observed_at"] + pd.DateOffset(days=1)
    lagged = lagged.rename(columns={"demand": "prediction"})
    baseline_lag = validation.merge(lagged, on=["station_id", "observed_at"], how="left")
    baseline_lag["baseline"] = "rezago_24h"

    profile = (
        train.groupby(["station_id", "day_of_week", "quarter_hour_block"], as_index=False)["demand"]
        .mean()
        .rename(columns={"demand": "prediction"})
    )
    baseline_profile = validation.merge(profile, on=["station_id", "day_of_week", "quarter_hour_block"], how="left")
    fallback = train.groupby("station_id")["demand"].mean()
    baseline_profile["prediction"] = baseline_profile["prediction"].fillna(baseline_profile["station_id"].map(fallback))
    baseline_profile["baseline"] = "promedio_estacion_dia_bloque"

    scored_frames = []
    station_scores = []
    for name, scored in [
        ("rezago_24h", baseline_lag.dropna(subset=["prediction"])),
        ("promedio_estacion_dia_bloque", baseline_profile.dropna(subset=["prediction"])),
    ]:
        scores = accuracy_by_station(scored).rename("accuracy").reset_index()
        scores["baseline"] = name
        scores["wape"] = 1 - (scores["accuracy"] / 100)
        station_scores.append(scores)
        scored_frames.append(scored)

    station_scores_frame = pd.concat(station_scores, ignore_index=True)
    summary = (
        station_scores_frame.groupby("baseline", as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), median_accuracy=("accuracy", "median"), mean_wape=("wape", "mean"))
        .sort_values("mean_accuracy", ascending=False)
    )
    return station_scores_frame, summary, cutoff


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_tables(frame: pd.DataFrame, context: pd.DataFrame, station_scores: pd.DataFrame, baseline_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    station_summary = (
        frame.groupby(["station_id", "station_name", "corridor"], as_index=False)
        .agg(
            demand_total=("demand", "sum"),
            demand_mean=("demand", "mean"),
            demand_std=("demand", "std"),
            demand_p95=("demand", lambda values: values.quantile(0.95)),
            demand_max=("demand", "max"),
        )
        .sort_values("demand_total", ascending=False)
    )
    daily_demand = frame.groupby("date", as_index=False)["demand"].sum()
    hourly_profile = frame.groupby(["hour"], as_index=False)["demand"].mean()
    weekly_profile = frame.groupby(["day_of_week", "day_name"], as_index=False)["demand"].mean()
    context_summary = context.describe().T.reset_index().rename(columns={"index": "variable"})
    return {
        "station_summary": station_summary,
        "daily_demand": daily_demand,
        "hourly_profile": hourly_profile,
        "weekly_profile": weekly_profile,
        "context_summary": context_summary,
        "baseline_station_scores": station_scores.sort_values(["baseline", "accuracy"]),
        "baseline_summary": baseline_summary,
    }


def generate_figures(frame: pd.DataFrame, context: pd.DataFrame, station_scores: pd.DataFrame, baseline_summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    daily = frame.groupby("date", as_index=False)["demand"].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["demand"], marker="o", linewidth=1.5, color="#2F6B8F")
    plt.xticks(rotation=45, ha="right")
    plt.title("Demanda diaria agregada")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda")
    save_plot(FIGURE_DIR / "01_demanda_diaria.png")

    hourly = frame.groupby("hour", as_index=False)["demand"].mean()
    plt.figure(figsize=(9, 5))
    plt.bar(hourly["hour"], hourly["demand"], color="#2F6B8F")
    plt.title("Perfil promedio por hora")
    plt.xlabel("Hora local")
    plt.ylabel("Demanda promedio")
    save_plot(FIGURE_DIR / "02_perfil_horario.png")

    weekly = frame.groupby(["day_of_week", "day_name"], as_index=False)["demand"].mean().sort_values("day_of_week")
    plt.figure(figsize=(9, 5))
    plt.bar(weekly["day_name"], weekly["demand"], color="#3C8D5B")
    plt.title("Demanda promedio por dia de semana")
    plt.xlabel("Dia")
    plt.ylabel("Demanda promedio")
    save_plot(FIGURE_DIR / "03_perfil_semanal.png")

    station_total = (
        frame.groupby(["station_id", "station_name"], as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
    )
    plt.figure(figsize=(11, 6))
    plt.barh(station_total["station_name"], station_total["demand"], color="#A45A52")
    plt.gca().invert_yaxis()
    plt.title("Demanda total por estacion")
    plt.xlabel("Demanda total")
    plt.ylabel("Estacion")
    save_plot(FIGURE_DIR / "04_ranking_estaciones.png")

    station_hour = frame.groupby(["station_name", "hour"], as_index=False)["demand"].mean()
    pivot = station_hour.pivot(index="station_name", columns="hour", values="demand")
    plt.figure(figsize=(13, 7))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Demanda promedio")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Perfil horario por estacion")
    plt.xlabel("Hora local")
    plt.ylabel("Estacion")
    save_plot(FIGURE_DIR / "05_heatmap_estacion_hora.png")

    plt.figure(figsize=(11, 6))
    station_order = station_total["station_name"].tolist()
    values = [frame.loc[frame["station_name"] == station, "demand"].to_numpy() for station in station_order]
    plt.boxplot(values, orientation="horizontal", tick_labels=station_order, showfliers=False, patch_artist=True)
    plt.title("Distribucion de demanda por estacion")
    plt.xlabel("Demanda")
    plt.ylabel("Estacion")
    save_plot(FIGURE_DIR / "06_distribucion_estaciones.png")

    plt.figure(figsize=(10, 5))
    plt.hist(context["rain_mm"], bins=30, color="#4C78A8")
    plt.title("Distribucion de lluvia observada")
    plt.xlabel("Lluvia mm")
    plt.ylabel("Frecuencia")
    save_plot(FIGURE_DIR / "07_distribucion_lluvia.png")

    plt.figure(figsize=(10, 5))
    plt.hist(context["temperature_c"], bins=30, color="#C46B41")
    plt.title("Distribucion de temperatura")
    plt.xlabel("Temperatura C")
    plt.ylabel("Frecuencia")
    save_plot(FIGURE_DIR / "08_distribucion_temperatura.png")

    context_join = frame.groupby("observed_at", as_index=False).agg(
        demand=("demand", "sum"), rain_mm=("rain_mm", "mean"), temperature_c=("temperature_c", "mean"), event_intensity=("event_intensity", "mean")
    )
    context_join["rain_bucket"] = pd.cut(context_join["rain_mm"], bins=[-0.01, 0, 1, 3, np.inf], labels=["0", "0-1", "1-3", "3+"])
    plt.figure(figsize=(8, 5))
    rain_order = ["0", "0-1", "1-3", "3+"]
    rain_values = [context_join.loc[context_join["rain_bucket"] == bucket, "demand"].to_numpy() for bucket in rain_order]
    plt.boxplot(rain_values, tick_labels=rain_order, patch_artist=True, showfliers=False)
    plt.title("Demanda agregada segun lluvia")
    plt.xlabel("Lluvia mm")
    plt.ylabel("Demanda agregada")
    save_plot(FIGURE_DIR / "09_demanda_vs_lluvia.png")

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        context_join["temperature_c"],
        context_join["demand"],
        c=context_join["event_intensity"],
        cmap="magma",
        s=18,
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Intensidad evento")
    plt.title("Demanda agregada vs temperatura y eventos")
    plt.xlabel("Temperatura C")
    plt.ylabel("Demanda agregada")
    save_plot(FIGURE_DIR / "10_demanda_contexto.png")

    plt.figure(figsize=(9, 5))
    plt.bar(baseline_summary["baseline"], baseline_summary["mean_accuracy"], color="#5B6C9E")
    plt.title("Accuracy promedio por baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Accuracy promedio")
    plt.ylim(0, max(100, baseline_summary["mean_accuracy"].max() + 5))
    save_plot(FIGURE_DIR / "11_accuracy_baselines.png")

    plt.figure(figsize=(11, 6))
    pivot_scores = station_scores.pivot(index="station_id", columns="baseline", values="accuracy").sort_index()
    x = np.arange(len(pivot_scores.index))
    width = 0.38
    for offset, baseline in enumerate(pivot_scores.columns):
        plt.barh(x + (offset - 0.5) * width, pivot_scores[baseline], height=width, label=baseline)
    plt.yticks(x, pivot_scores.index)
    plt.legend()
    plt.title("Accuracy por estacion y baseline")
    plt.xlabel("Accuracy")
    plt.ylabel("Estacion")
    save_plot(FIGURE_DIR / "12_accuracy_estacion_baseline.png")


def write_markdown_summary(
    frame: pd.DataFrame,
    metadata: dict,
    cutoff: pd.Timestamp,
    tables: dict[str, pd.DataFrame],
    baseline_summary: pd.DataFrame,
) -> None:
    dataset = dataset_metadata(metadata)
    local_min = frame["observed_at"].min().tz_convert(BOGOTA_TZ)
    local_max = frame["observed_at"].max().tz_convert(BOGOTA_TZ)
    top_station = tables["station_summary"].iloc[0]
    best_baseline = baseline_summary.iloc[0]

    lines = [
        "# Resumen EDA Pulso TransMi",
        "",
        "## Datos",
        "",
        f"- Dataset: `{dataset['dataset']}`.",
        f"- Rango local: {local_min} a {local_max}.",
        f"- Estaciones: {dataset['station_count']}.",
        f"- Observaciones: {dataset['observation_rows']:,}.",
        f"- Contexto: {dataset['context_rows']:,} registros.",
        "",
        "## Hallazgos iniciales",
        "",
        f"- Estacion con mayor demanda total: {top_station['station_name']} (`{top_station['station_id']}`).",
        f"- Cutoff de validacion temporal: {cutoff.tz_convert(BOGOTA_TZ)}; se reservan los ultimos 7 dias.",
        f"- Mejor baseline inicial: `{best_baseline['baseline']}` con accuracy promedio {best_baseline['mean_accuracy']:.2f}.",
        "- La validacion es temporal; no se usa split aleatorio porque mezclaria pasado y futuro.",
        "",
        "## Baselines",
        "",
        baseline_summary.to_markdown(index=False),
        "",
        "## Figuras generadas",
        "",
    ]
    lines.extend(f"- `figures/{path.name}`" for path in sorted(FIGURE_DIR.glob("*.png")))
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stations, observations, context, metadata = read_raw_data()
    quality_reports, quality_errors = validate_data(stations, observations, context, metadata)
    write_quality_report(quality_reports, quality_errors)
    if quality_errors:
        for error in quality_errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    frame = prepare_dataset(stations, observations, context)
    frame.to_csv(PROCESSED_DIR / "observations_features.csv", index=False)

    station_scores, baseline_summary, cutoff = evaluate_baselines(frame)
    tables = generate_tables(frame, context, station_scores, baseline_summary)
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)

    generate_figures(frame, context, station_scores, baseline_summary)
    write_markdown_summary(frame, metadata, cutoff, tables, baseline_summary)
    print("EDA generado en reports/eda/")


if __name__ == "__main__":
    main()
