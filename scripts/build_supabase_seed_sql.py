from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.features import BOGOTA_TZ, DAY_NAMES_ES


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("supabase/seed/initial_api_data.sql")


def sql_literal(value: object) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def timestamp_literal(value: pd.Timestamp) -> str:
    return sql_literal(value.isoformat())


def value_rows(rows: list[tuple[object, ...]]) -> str:
    return ",\n".join("(" + ", ".join(sql_literal(value) for value in row) + ")" for row in rows)


def chunked(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [frame.iloc[start : start + size] for start in range(0, len(frame), size)]


def build_seed_sql(raw_dir: Path = RAW_DIR, chunk_size: int = 1000) -> str:
    stations = pd.read_csv(raw_dir / "stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        raw_dir / "observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(raw_dir / "context.csv", parse_dates=["observed_at"])

    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    slots = pd.DataFrame({"observed_at": sorted(context["observed_at"].unique())})
    slots["observed_at"] = pd.to_datetime(slots["observed_at"], utc=True)
    local_time = slots["observed_at"].dt.tz_convert(BOGOTA_TZ)
    slots["local_date"] = local_time.dt.date.astype("string")
    slots["hour"] = local_time.dt.hour
    slots["minute"] = local_time.dt.minute
    slots["day_of_week"] = local_time.dt.dayofweek
    slots["day_name"] = slots["day_of_week"].map(DAY_NAMES_ES)
    slots["is_weekend"] = slots["day_of_week"].isin([5, 6])
    slots["quarter_hour_block"] = slots["hour"] * 4 + (slots["minute"] // 15)

    statements = [
        "begin;",
        "truncate table public.data_quality_checks, public.metrics_by_station, public.predictions, public.model_versions, public.pipeline_runs, public.station_time_features, public.observations, public.context_observations, public.time_slots, public.stations restart identity cascade;",
    ]

    station_rows = [
        (
            row.station_id,
            row.station_name,
            row.corridor,
            row.latitude,
            row.longitude,
        )
        for row in stations.itertuples(index=False)
    ]
    statements.append(
        "insert into public.stations (station_id, station_name, corridor, latitude, longitude) values\n"
        + value_rows(station_rows)
        + "\non conflict (station_id) do update set station_name = excluded.station_name, corridor = excluded.corridor, latitude = excluded.latitude, longitude = excluded.longitude;"
    )

    for chunk in chunked(slots, chunk_size):
        slot_rows = [
            (
                row.observed_at.isoformat(),
                row.local_date,
                row.hour,
                row.minute,
                row.day_of_week,
                row.day_name,
                row.is_weekend,
                row.quarter_hour_block,
            )
            for row in chunk.itertuples(index=False)
        ]
        statements.append(
            "insert into public.time_slots (observed_at, local_date, hour, minute, day_of_week, day_name, is_weekend, quarter_hour_block) values\n"
            + value_rows(slot_rows)
            + "\non conflict (observed_at) do nothing;"
        )

    for chunk in chunked(context, chunk_size):
        context_rows = [
            (
                row.observed_at.isoformat(),
                row.rain_mm,
                row.rain_forecast,
                row.temperature_c,
                row.temperature_forecast,
                row.event_intensity,
            )
            for row in chunk.itertuples(index=False)
        ]
        statements.append(
            "insert into public.context_observations (observed_at, rain_mm, rain_forecast, temperature_c, temperature_forecast, event_intensity) values\n"
            + value_rows(context_rows)
            + "\non conflict (observed_at) do update set rain_mm = excluded.rain_mm, rain_forecast = excluded.rain_forecast, temperature_c = excluded.temperature_c, temperature_forecast = excluded.temperature_forecast, event_intensity = excluded.event_intensity;"
        )

    for chunk in chunked(observations, chunk_size):
        observation_rows = [
            (
                row.station_id,
                row.observed_at.isoformat(),
                row.demand,
            )
            for row in chunk.itertuples(index=False)
        ]
        statements.append(
            "insert into public.observations (station_id, observed_at, demand) values\n"
            + value_rows(observation_rows)
            + "\non conflict (station_id, observed_at) do update set demand = excluded.demand;"
        )

    statements.append("commit;")
    return "\n\n".join(statements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_seed_sql(args.raw_dir, args.chunk_size))
    print(f"Seed SQL escrito en {args.output}")


if __name__ == "__main__":
    main()
