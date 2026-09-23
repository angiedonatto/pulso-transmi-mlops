from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BOGOTA_TZ = "America/Bogota"
HORIZONS = (15, 30, 45, 60)
SHORT_LAGS = (15, 30, 45, 60)
DAILY_LAG_MINUTES = 24 * 60
WEEKLY_LAG_MINUTES = 7 * 24 * 60
VALIDATION_DAYS = 7
TRAIN_LOOKBACK_DAYS = 21
MIN_HISTORY_DAYS = 14

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "model_improvement"
TABLE_DIR = REPORT_DIR / "tables"


@dataclass(frozen=True)
class ModelResult:
    name: str
    predictions: pd.DataFrame
    summary: dict[str, object]
    train_seconds: float
    complexity: str


@dataclass(frozen=True)
class ProfileSeries:
    times: np.ndarray
    means: np.ndarray
    medians: np.ndarray


def load_observations(path: Path = RAW_DIR / "observations.csv") -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station_id": "string"})
    return load_observations_from_frame(frame)


def load_observations_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    local = frame["observed_at"].dt.tz_convert(BOGOTA_TZ)
    frame["hour"] = local.dt.hour
    frame["minute"] = local.dt.minute
    frame["day_of_week"] = local.dt.dayofweek
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["quarter_hour_block"] = frame["hour"] * 4 + frame["minute"] // 15
    return frame.sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def validation_cutoffs(observations: pd.DataFrame) -> list[pd.Timestamp]:
    times = sorted(observations["observed_at"].drop_duplicates())
    max_time = times[-1]
    start = max_time - pd.Timedelta(days=VALIDATION_DAYS)
    latest_cutoff = max_time - pd.Timedelta(minutes=max(HORIZONS))
    return [ts for ts in times if start <= ts <= latest_cutoff]


def fold_starts(cutoffs: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    days = sorted({ts.floor("D") for ts in cutoffs})
    return days


def build_lookup(observations: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    return {
        (str(row.station_id), row.observed_at): float(row.demand)
        for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)
    }


def build_profile_lookup(observations: pd.DataFrame) -> dict[tuple[str, int], ProfileSeries]:
    profiles: dict[tuple[str, int], ProfileSeries] = {}
    for (station_id, quarter_hour_block), group in observations.groupby(["station_id", "quarter_hour_block"], sort=True):
        ordered = group.sort_values("observed_at")
        values = ordered["demand"].to_numpy(dtype=float)
        profiles[(str(station_id), int(quarter_hour_block))] = ProfileSeries(
            times=ordered["observed_at"].to_numpy(dtype="datetime64[ns]"),
            means=pd.Series(values).expanding().mean().to_numpy(dtype=float),
            medians=pd.Series(values).expanding().median().to_numpy(dtype=float),
        )
    return profiles


def station_code_map(observations: pd.DataFrame) -> dict[str, int]:
    return {station_id: idx for idx, station_id in enumerate(sorted(observations["station_id"].unique()))}


def _target_time_features(target_at: pd.Timestamp) -> dict[str, int]:
    local = target_at.tz_convert(BOGOTA_TZ)
    return {
        "target_hour": int(local.hour),
        "target_minute": int(local.minute),
        "target_day_of_week": int(local.dayofweek),
        "target_is_weekend": int(local.dayofweek in (5, 6)),
        "target_quarter_hour_block": int(local.hour * 4 + local.minute // 15),
    }


def _value_at(
    lookup: dict[tuple[str, pd.Timestamp], float],
    station_id: str,
    observed_at: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> float:
    if observed_at > cutoff:
        return np.nan
    return lookup.get((station_id, observed_at), np.nan)


def _station_profile_value(
    profiles: dict[tuple[str, int], ProfileSeries],
    station_id: str,
    quarter_hour_block: int,
    cutoff_np: np.datetime64,
    column: str,
) -> float:
    profile = profiles.get((station_id, quarter_hour_block))
    if profile is None:
        return np.nan
    index = int(np.searchsorted(profile.times, cutoff_np, side="right") - 1)
    if index < 0:
        return np.nan
    if column == "median":
        return float(profile.medians[index])
    return float(profile.means[index])


def make_samples(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    lookup: dict[tuple[str, pd.Timestamp], float] | None = None,
    profiles: dict[tuple[str, int], ProfileSeries] | None = None,
) -> pd.DataFrame:
    if not cutoffs:
        return pd.DataFrame()

    base = observations.sort_values(["station_id", "observed_at"]).copy()
    base = base.rename(columns={"observed_at": "target_at", "demand": "actual"})
    by_station = base.groupby("station_id", sort=False)
    for lag in SHORT_LAGS:
        base[f"lag_{lag}"] = by_station["actual"].shift(lag // 15)
    base["lag_96"] = by_station["actual"].shift(DAILY_LAG_MINUTES // 15)
    base["lag_weekly"] = by_station["actual"].shift(WEEKLY_LAG_MINUTES // 15)

    by_slot = base.groupby(["station_id", "quarter_hour_block"], sort=False)["actual"]
    base["profile_mean"] = by_slot.transform(lambda values: values.shift(1).expanding().mean())
    base["profile_median"] = by_slot.transform(lambda values: values.shift(1).expanding().median())
    base = base.rename(
        columns={
            "hour": "target_hour",
            "minute": "target_minute",
            "day_of_week": "target_day_of_week",
            "is_weekend": "target_is_weekend",
            "quarter_hour_block": "target_quarter_hour_block",
        }
    )

    cutoff_index = pd.DatetimeIndex(cutoffs)
    samples = []
    for horizon in HORIZONS:
        part = base.copy()
        part["horizon_minutes"] = horizon
        part["data_cutoff"] = part["target_at"] - pd.Timedelta(minutes=horizon)
        part = part[part["data_cutoff"].isin(cutoff_index)].copy()
        for lag in SHORT_LAGS:
            if lag < horizon:
                part[f"lag_{lag}"] = np.nan
        lag_columns = [f"lag_{lag}" for lag in SHORT_LAGS]
        part["recent_mean"] = part[lag_columns].mean(axis=1, skipna=True)
        part["recent_median"] = part[lag_columns].median(axis=1, skipna=True)
        samples.append(part)
    result = pd.concat(samples, ignore_index=True)
    columns = [
        "station_id",
        "data_cutoff",
        "target_at",
        "horizon_minutes",
        "actual",
        "target_hour",
        "target_minute",
        "target_day_of_week",
        "target_is_weekend",
        "target_quarter_hour_block",
        "lag_15",
        "lag_30",
        "lag_45",
        "lag_60",
        "lag_96",
        "lag_weekly",
        "recent_mean",
        "recent_median",
        "profile_mean",
        "profile_median",
    ]
    return result[columns].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)


def cached_samples(
    cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    lookup: dict[tuple[str, pd.Timestamp], float],
    profiles: dict[tuple[str, int], ProfileSeries],
) -> pd.DataFrame:
    if not cutoffs:
        return pd.DataFrame()
    key = (cutoffs[0], cutoffs[-1], len(cutoffs))
    if key not in cache:
        cache[key] = make_samples(observations, cutoffs, lookup=lookup, profiles=profiles)
    return cache[key].copy()


def official_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    scored = predictions.copy()
    scored["abs_error"] = (scored["actual"] - scored["prediction"]).abs()

    station = _metric_by(scored, ["station_id"]).rename(
        columns={"wape": "station_wape", "accuracy": "station_accuracy"}
    )
    horizon = _metric_by(scored, ["horizon_minutes"]).rename(
        columns={"wape": "horizon_wape", "accuracy": "horizon_accuracy"}
    )
    station_horizon = _metric_by(scored, ["station_id", "horizon_minutes"]).rename(
        columns={"wape": "station_horizon_wape", "accuracy": "station_horizon_accuracy"}
    )
    hour = _metric_by(scored, ["target_hour"]).rename(columns={"wape": "hour_wape", "accuracy": "hour_accuracy"})
    weekday = _metric_by(scored, ["target_day_of_week"]).rename(
        columns={"wape": "weekday_wape", "accuracy": "weekday_accuracy"}
    )

    mean_station_accuracy = float(station["station_accuracy"].mean())
    worst_station = station.sort_values("station_accuracy", ascending=True).iloc[0].to_dict()
    worst_horizon = horizon.sort_values("horizon_accuracy", ascending=True).iloc[0].to_dict()
    worst_station_horizon = station_horizon.sort_values("station_horizon_accuracy", ascending=True).iloc[0].to_dict()

    return {
        "mean_station_accuracy": mean_station_accuracy,
        "mean_station_wape": float(station["station_wape"].mean()),
        "station_metrics": station.sort_values("station_accuracy").reset_index(drop=True),
        "horizon_metrics": horizon.sort_values("horizon_minutes").reset_index(drop=True),
        "station_horizon_metrics": station_horizon.sort_values("station_horizon_accuracy").reset_index(drop=True),
        "hour_metrics": hour.sort_values("hour_wape", ascending=False).reset_index(drop=True),
        "weekday_metrics": weekday.sort_values("weekday_wape", ascending=False).reset_index(drop=True),
        "worst_station": worst_station,
        "worst_horizon": worst_horizon,
        "worst_station_horizon": worst_station_horizon,
        "rows": int(len(scored)),
    }


def _metric_by(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = scored.groupby(keys, as_index=False).agg(
        abs_error=("abs_error", "sum"),
        actual_sum=("actual", "sum"),
        rows=("actual", "size"),
    )
    grouped["wape"] = np.where(grouped["actual_sum"] == 0, 0.0, grouped["abs_error"] / grouped["actual_sum"])
    grouped["accuracy"] = 100.0 * (1.0 - grouped["wape"]).clip(lower=0.0)
    return grouped.drop(columns=["abs_error"])


def champion_predictions(samples: pd.DataFrame) -> ModelResult:
    start = time.perf_counter()
    result = samples.copy()
    result["prediction"] = result["lag_96"].fillna(result["profile_mean"])
    result["prediction"] = result["prediction"].clip(lower=0)
    metrics = official_metrics(result)
    return ModelResult(
        name="champion actual",
        predictions=result,
        summary=metrics,
        train_seconds=time.perf_counter() - start,
        complexity="Muy baja: lag diario + fallback estacion x quarter-hour.",
    )


def candidate_a_predictions(
    observations: pd.DataFrame,
    all_cutoffs: list[pd.Timestamp],
    *,
    lookup: dict[tuple[str, pd.Timestamp], float],
    profiles: dict[tuple[str, int], ProfileSeries],
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> ModelResult:
    start = time.perf_counter()
    validation_start = min(all_cutoffs)
    predictions: list[pd.DataFrame] = []
    selected_weights: dict[str, dict[str, float]] = {}
    for fold_start in fold_starts(all_cutoffs):
        fold_cutoffs = [ts for ts in all_cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_start = max(observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS))
        train_cutoffs = [
            ts
            for ts in sorted(observations["observed_at"].drop_duplicates())
            if train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))
        ]
        train_samples = cached_samples(sample_cache, observations, train_cutoffs, lookup=lookup, profiles=profiles)
        fold_samples = cached_samples(sample_cache, observations, fold_cutoffs, lookup=lookup, profiles=profiles)
        for horizon in HORIZONS:
            weights = _choose_candidate_a_weights(train_samples[train_samples["horizon_minutes"] == horizon])
            selected_weights[f"{fold_start.date()}_h{horizon}"] = weights
            mask = fold_samples["horizon_minutes"] == horizon
            fold_samples.loc[mask, "prediction"] = _blend_candidate_a(fold_samples.loc[mask], weights)
        predictions.append(fold_samples)
    result = pd.concat(predictions, ignore_index=True)
    result["prediction"] = result["prediction"].fillna(result["lag_96"]).fillna(result["profile_mean"]).clip(lower=0)
    metrics = official_metrics(result)
    metrics["selected_weights"] = selected_weights
    metrics["validation_start"] = validation_start.isoformat().replace("+00:00", "Z")
    return ModelResult(
        name="candidate A",
        predictions=result,
        summary=metrics,
        train_seconds=time.perf_counter() - start,
        complexity="Baja: blend por horizonte de lags recientes, lag diario y perfil historico.",
    )


def _choose_candidate_a_weights(train_samples: pd.DataFrame) -> dict[str, float]:
    if train_samples.empty:
        return {"recent": 0.25, "daily": 0.5, "profile": 0.25}
    candidates: list[dict[str, float]] = []
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    for recent in grid:
        for daily in grid:
            profile = round(1.0 - recent - daily, 2)
            if profile < 0:
                continue
            candidates.append({"recent": recent, "daily": daily, "profile": profile})
    best = candidates[0]
    best_score = -math.inf
    for weights in candidates:
        scored = train_samples.copy()
        scored["prediction"] = _blend_candidate_a(scored, weights)
        scored["prediction"] = scored["prediction"].fillna(scored["lag_96"]).fillna(scored["profile_mean"]).clip(lower=0)
        score = float(official_metrics(scored)["mean_station_accuracy"])
        if score > best_score:
            best_score = score
            best = weights
    return best


def _blend_candidate_a(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    recent = frame["recent_median"].fillna(frame["recent_mean"])
    daily = frame["lag_96"]
    profile = frame["profile_median"].fillna(frame["profile_mean"])
    return weights["recent"] * recent + weights["daily"] * daily + weights["profile"] * profile


def candidate_b_predictions(
    observations: pd.DataFrame,
    all_cutoffs: list[pd.Timestamp],
    *,
    lookup: dict[tuple[str, pd.Timestamp], float],
    profiles: dict[tuple[str, int], ProfileSeries],
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> ModelResult:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for candidate B") from exc

    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions: list[pd.DataFrame] = []
    for fold_start in fold_starts(all_cutoffs):
        fold_cutoffs = [ts for ts in all_cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_start = max(observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS))
        train_cutoffs = [
            ts
            for ts in sorted(observations["observed_at"].drop_duplicates())
            if train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))
        ]
        train_samples = cached_samples(sample_cache, observations, train_cutoffs, lookup=lookup, profiles=profiles)
        fold_samples = cached_samples(sample_cache, observations, fold_cutoffs, lookup=lookup, profiles=profiles)
        train_x = _candidate_b_matrix(train_samples, codes)
        fold_x = _candidate_b_matrix(fold_samples, codes)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=17,
        )
        model.fit(train_x, train_samples["actual"].to_numpy())
        fold_samples["prediction"] = np.maximum(0.0, model.predict(fold_x))
        predictions.append(fold_samples)
    result = pd.concat(predictions, ignore_index=True)
    metrics = official_metrics(result)
    return ModelResult(
        name="candidate B",
        predictions=result,
        summary=metrics,
        train_seconds=time.perf_counter() - start,
        complexity="Media: HistGradientBoostingRegressor global, reentrenado por fold diario.",
    )


def _candidate_b_matrix(samples: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(codes).astype(float)
    columns = [
        "station_code",
        "horizon_minutes",
        "target_hour",
        "target_minute",
        "target_day_of_week",
        "target_is_weekend",
        "target_quarter_hour_block",
        "lag_15",
        "lag_30",
        "lag_45",
        "lag_60",
        "lag_96",
        "lag_weekly",
        "recent_mean",
        "recent_median",
        "profile_mean",
        "profile_median",
    ]
    return frame[columns].astype(float)


def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    champion = results[0]
    champion_accuracy = float(champion.summary["mean_station_accuracy"])
    champion_station = champion.summary["station_metrics"][["station_id", "station_accuracy"]].rename(
        columns={"station_accuracy": "champion_accuracy"}
    )
    rows = []
    for result in results:
        station = result.summary["station_metrics"][["station_id", "station_accuracy"]].rename(
            columns={"station_accuracy": "candidate_accuracy"}
        )
        joined = champion_station.merge(station, on="station_id", how="inner")
        improving = int((joined["candidate_accuracy"] > joined["champion_accuracy"] + 1e-9).sum())
        worsening = int((joined["candidate_accuracy"] < joined["champion_accuracy"] - 1e-9).sum())
        rows.append(
            {
                "model": result.name,
                "accuracy_oficial_simulada": float(result.summary["mean_station_accuracy"]),
                "mejora_pp": float(result.summary["mean_station_accuracy"]) - champion_accuracy,
                "estaciones_mejoran": improving,
                "estaciones_empeoran": worsening,
                "complejidad_operativa": result.complexity,
                "tiempo_entrenamiento_s": result.train_seconds,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(results: list[ModelResult], observations: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        slug = result.name.replace(" ", "_")
        result.predictions.to_csv(TABLE_DIR / f"{slug}_predictions.csv", index=False)
        result.summary["station_metrics"].to_csv(TABLE_DIR / f"{slug}_station_metrics.csv", index=False)
        result.summary["horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_horizon_metrics.csv", index=False)
        result.summary["station_horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_station_horizon_metrics.csv", index=False)
        result.summary["hour_metrics"].to_csv(TABLE_DIR / f"{slug}_hour_metrics.csv", index=False)
        result.summary["weekday_metrics"].to_csv(TABLE_DIR / f"{slug}_weekday_metrics.csv", index=False)

    comparison = comparison_table(results)
    comparison.to_csv(TABLE_DIR / "comparison.csv", index=False)
    payload = {
        "data_min": observations["observed_at"].min().isoformat().replace("+00:00", "Z"),
        "data_max": observations["observed_at"].max().isoformat().replace("+00:00", "Z"),
        "stations": int(observations["station_id"].nunique()),
        "validation_cutoffs": len(cutoffs),
        "validation_start": min(cutoffs).isoformat().replace("+00:00", "Z"),
        "validation_end": max(cutoffs).isoformat().replace("+00:00", "Z"),
        "models": {
            result.name: {
                "mean_station_accuracy": result.summary["mean_station_accuracy"],
                "mean_station_wape": result.summary["mean_station_wape"],
                "worst_station": result.summary["worst_station"],
                "worst_horizon": result.summary["worst_horizon"],
                "worst_station_horizon": result.summary["worst_station_horizon"],
                "train_seconds": result.train_seconds,
                "complexity": result.complexity,
            }
            for result in results
        },
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(results, comparison, payload) + "\n")


def render_markdown(results: list[ModelResult], comparison: pd.DataFrame, payload: dict[str, object]) -> str:
    champion, candidate_a, candidate_b = results
    best = comparison.sort_values("accuracy_oficial_simulada", ascending=False).iloc[0]
    lines = [
        "# Model improvement candidates",
        "",
        "## Scope",
        "",
        "- No production files, scheduler, submission logic, Supabase state, or active champion were modified.",
        "- Validation is temporal walk-forward by daily folds over the last 7 days available locally.",
        "- Every sample uses only sources at or before its `data_cutoff`.",
        "",
        "## Data window",
        "",
        f"- Observations: {payload['data_min']} to {payload['data_max']}.",
        f"- Stations: {payload['stations']}.",
        f"- Validation cutoffs: {payload['validation_cutoffs']} from {payload['validation_start']} to {payload['validation_end']}.",
        "",
        "## Champion diagnosis",
        "",
        _diagnosis_lines(champion),
        "",
        "## Candidate results",
        "",
        _model_lines(candidate_a),
        "",
        _model_lines(candidate_b),
        "",
        "## Comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Recommendation",
        "",
        _recommendation(best, comparison),
        "",
        "## Artifacts",
        "",
        "- `reports/model_improvement/summary.md`",
        "- `reports/model_improvement/summary.json`",
        "- `reports/model_improvement/tables/*.csv`",
    ]
    return "\n".join(lines)


def _diagnosis_lines(result: ModelResult) -> str:
    summary = result.summary
    worst_station = summary["worst_station"]
    worst_horizon = summary["worst_horizon"]
    worst_pair = summary["worst_station_horizon"]
    return "\n".join(
        [
            f"- Mean station accuracy: {summary['mean_station_accuracy']:.3f}%.",
            f"- Mean station WAPE: {summary['mean_station_wape']:.4f}.",
            f"- Worst station: {worst_station['station_id']} with accuracy {worst_station['station_accuracy']:.3f}% and WAPE {worst_station['station_wape']:.4f}.",
            f"- Worst horizon: +{int(worst_horizon['horizon_minutes'])} min with accuracy {worst_horizon['horizon_accuracy']:.3f}% and WAPE {worst_horizon['horizon_wape']:.4f}.",
            f"- Worst station x horizon: {worst_pair['station_id']} +{int(worst_pair['horizon_minutes'])} min with accuracy {worst_pair['station_horizon_accuracy']:.3f}%.",
            "- Highest WAPE tables are saved by station, horizon, hour, weekday, and station x horizon.",
        ]
    )


def _model_lines(result: ModelResult) -> str:
    worst_station = result.summary["worst_station"]
    worst_horizon = result.summary["worst_horizon"]
    return "\n".join(
        [
            f"### {result.name}",
            "",
            f"- Mean station accuracy: {result.summary['mean_station_accuracy']:.3f}%.",
            f"- Mean station WAPE: {result.summary['mean_station_wape']:.4f}.",
            f"- Worst station: {worst_station['station_id']} ({worst_station['station_accuracy']:.3f}%).",
            f"- Worst horizon: +{int(worst_horizon['horizon_minutes'])} min ({worst_horizon['horizon_accuracy']:.3f}%).",
            f"- Training/evaluation time: {result.train_seconds:.2f}s.",
            f"- Complexity: {result.complexity}",
        ]
    )


def _recommendation(best: pd.Series, comparison: pd.DataFrame) -> str:
    if best["model"] == "champion actual":
        return "No promover candidate todavia: el champion sigue siendo el mejor en esta validacion temporal."
    champion = comparison[comparison["model"] == "champion actual"].iloc[0]
    best_degradation = int(best["estaciones_empeoran"])
    if float(best["mejora_pp"]) <= 0:
        return "No promover candidate todavia: no hay mejora positiva contra champion."
    if best_degradation >= 4:
        return "Seguir experimentando: hay mejora agregada, pero demasiadas estaciones empeoran."
    return (
        f"Candidate recomendado para siguiente revision: {best['model']}. "
        "No promover automaticamente; revisar estaciones degradadas y probar en shadow run antes de produccion."
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    observations = load_observations()
    cutoffs = validation_cutoffs(observations)
    if not cutoffs:
        raise SystemExit("No validation cutoffs available.")
    lookup = build_lookup(observations)
    profiles = build_profile_lookup(observations)
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] = {}
    samples = cached_samples(sample_cache, observations, cutoffs, lookup=lookup, profiles=profiles)
    results = [
        champion_predictions(samples),
        candidate_a_predictions(
            observations,
            cutoffs,
            lookup=lookup,
            profiles=profiles,
            sample_cache=sample_cache,
        ),
        candidate_b_predictions(
            observations,
            cutoffs,
            lookup=lookup,
            profiles=profiles,
            sample_cache=sample_cache,
        ),
    ]
    write_outputs(results, observations, cutoffs)
    comparison = comparison_table(results)
    print(comparison.to_string(index=False))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
