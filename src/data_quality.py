from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports/eda")
TABLE_DIR = REPORT_DIR / "tables"


def dataset_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    dataset = metadata.get("dataset", metadata)
    if isinstance(dataset, str):
        return metadata
    return dataset


def read_raw_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ["stations.csv", "observations.csv", "context.csv", "metadata.json"]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos en {raw_dir}: {', '.join(missing)}. "
            "Ejecuta primero: python -m src.ingest"
        )

    stations = pd.read_csv(raw_dir / "stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        raw_dir / "observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(raw_dir / "context.csv", parse_dates=["observed_at"])
    metadata = json.loads((raw_dir / "metadata.json").read_text())

    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    return stations, observations, context, metadata


def _expected_index(metadata: dict[str, Any]) -> pd.DatetimeIndex:
    dataset = dataset_metadata(metadata)
    start = pd.Timestamp(dataset["history_start"]).tz_convert("UTC")
    end = pd.Timestamp(dataset["history_end"]).tz_convert("UTC")
    return pd.date_range(start=start, end=end, freq=f"{dataset['frequency_minutes']}min")


def validate_data(
    stations: pd.DataFrame,
    observations: pd.DataFrame,
    context: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    dataset = dataset_metadata(metadata)
    expected_times = _expected_index(metadata)
    expected_station_ids = set(stations["station_id"].astype("string"))
    errors: list[str] = []

    row_counts = pd.DataFrame(
        [
            {"table": "stations", "actual": len(stations), "expected": dataset["station_count"]},
            {"table": "observations", "actual": len(observations), "expected": dataset["observation_rows"]},
            {"table": "context", "actual": len(context), "expected": dataset["context_rows"]},
        ]
    )
    for row in row_counts.to_dict("records"):
        if int(row["actual"]) != int(row["expected"]):
            errors.append(f"{row['table']}: filas reales {row['actual']} != esperadas {row['expected']}")

    missing_values = pd.concat(
        [
            stations.isna().sum().rename("missing").reset_index().assign(table="stations"),
            observations.isna().sum().rename("missing").reset_index().assign(table="observations"),
            context.isna().sum().rename("missing").reset_index().assign(table="context"),
        ],
        ignore_index=True,
    ).rename(columns={"index": "column"})

    duplicate_obs = int(observations.duplicated(["station_id", "observed_at"]).sum())
    duplicate_context = int(context.duplicated(["observed_at"]).sum())
    duplicates = pd.DataFrame(
        [
            {"check": "observations_station_timestamp", "duplicates": duplicate_obs},
            {"check": "context_timestamp", "duplicates": duplicate_context},
        ]
    )
    if duplicate_obs:
        errors.append(f"observations: {duplicate_obs} duplicados por station_id + observed_at")
    if duplicate_context:
        errors.append(f"context: {duplicate_context} duplicados por observed_at")

    station_lengths = stations["station_id"].astype("string").str.len()
    obs_station_lengths = observations["station_id"].astype("string").str.len()
    if not station_lengths.eq(5).all() or not obs_station_lengths.eq(5).all():
        errors.append("station_id debe conservar 5 caracteres, incluidos ceros iniciales")

    unknown_stations = sorted(set(observations["station_id"].astype("string")) - expected_station_ids)
    if unknown_stations:
        errors.append(f"observations contiene station_id no catalogados: {unknown_stations}")

    coverage_rows = []
    for station_id, group in observations.groupby("station_id", sort=True):
        unique_times = pd.DatetimeIndex(group["observed_at"].drop_duplicates().sort_values())
        missing_periods = len(expected_times.difference(unique_times))
        extra_periods = len(unique_times.difference(expected_times))
        coverage_rows.append(
            {
                "station_id": station_id,
                "rows": len(group),
                "unique_timestamps": len(unique_times),
                "missing_periods": missing_periods,
                "extra_periods": extra_periods,
                "min_observed_at": unique_times.min() if len(unique_times) else pd.NaT,
                "max_observed_at": unique_times.max() if len(unique_times) else pd.NaT,
            }
        )
        if missing_periods or extra_periods:
            errors.append(
                f"station_id {station_id}: {missing_periods} huecos y {extra_periods} timestamps fuera del rango esperado"
            )

    coverage = pd.DataFrame(coverage_rows)
    context_unique_times = pd.DatetimeIndex(context["observed_at"].drop_duplicates().sort_values())
    context_coverage = pd.DataFrame(
        [
            {
                "rows": len(context),
                "unique_timestamps": len(context_unique_times),
                "missing_periods": len(expected_times.difference(context_unique_times)),
                "extra_periods": len(context_unique_times.difference(expected_times)),
                "min_observed_at": context_unique_times.min(),
                "max_observed_at": context_unique_times.max(),
            }
        ]
    )
    if int(context_coverage.loc[0, "missing_periods"]) or int(context_coverage.loc[0, "extra_periods"]):
        errors.append("context no cubre exactamente el rango temporal esperado")

    if (observations["demand"] < 0).any():
        errors.append("observations contiene demanda negativa")

    context_times = set(context_unique_times)
    missing_context_for_observations = sorted(set(observations["observed_at"]) - context_times)
    if missing_context_for_observations:
        errors.append("hay observaciones sin timestamp equivalente en context")

    reports = {
        "row_counts": row_counts,
        "missing_values": missing_values[["table", "column", "missing"]],
        "duplicates": duplicates,
        "station_coverage": coverage,
        "context_coverage": context_coverage,
    }
    return reports, errors


def write_quality_report(reports: dict[str, pd.DataFrame], errors: list[str], report_dir: Path = REPORT_DIR) -> None:
    table_dir = report_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(table_dir / f"quality_{name}.csv", index=False)

    status = "APROBADO" if not errors else "CON ERRORES"
    lines = [
        "# Reporte de calidad de datos",
        "",
        f"Estado: **{status}**",
        "",
        "## Conteos",
        "",
        reports["row_counts"].to_markdown(index=False),
        "",
        "## Duplicados",
        "",
        reports["duplicates"].to_markdown(index=False),
        "",
        "## Cobertura por estacion",
        "",
        reports["station_coverage"][["station_id", "rows", "missing_periods", "extra_periods"]].to_markdown(index=False),
        "",
        "## Resultado",
        "",
    ]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- No se encontraron errores bloqueantes de calidad.")
    (report_dir / "data_quality.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    stations, observations, context, metadata = read_raw_data()
    reports, errors = validate_data(stations, observations, context, metadata)
    write_quality_report(reports, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Validacion de calidad aprobada. Reporte: reports/eda/data_quality.md")


if __name__ == "__main__":
    main()
