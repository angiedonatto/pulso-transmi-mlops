import pandas as pd

from src.data_quality import dataset_metadata, validate_data


def sample_metadata() -> dict:
    return {
        "dataset": "test-dataset",
        "frequency_minutes": 15,
        "periods_per_station": 2,
        "station_count": 1,
        "observation_rows": 2,
        "context_rows": 2,
        "history_start": "2026-01-01T00:00:00-05:00",
        "history_end": "2026-01-01T00:15:00-05:00",
    }


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = pd.DataFrame(
        [{"station_id": "03000", "station_name": "Portal Suba", "corridor": "Suba"}]
    ).astype({"station_id": "string"})
    observations = pd.DataFrame(
        [
            {"station_id": "03000", "observed_at": "2026-01-01T00:00:00-05:00", "demand": 10},
            {"station_id": "03000", "observed_at": "2026-01-01T00:15:00-05:00", "demand": 12},
        ]
    ).astype({"station_id": "string"})
    context = pd.DataFrame(
        [
            {"observed_at": "2026-01-01T00:00:00-05:00", "rain_mm": 0.0},
            {"observed_at": "2026-01-01T00:15:00-05:00", "rain_mm": 0.1},
        ]
    )
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    return stations, observations, context


def test_dataset_metadata_accepts_downloaded_and_api_shapes() -> None:
    downloaded = sample_metadata()
    api_shape = {"project": "Pulso TransMi", "dataset": downloaded}

    assert dataset_metadata(downloaded)["station_count"] == 1
    assert dataset_metadata(api_shape)["station_count"] == 1


def test_validate_data_passes_for_complete_quarter_hour_series() -> None:
    reports, errors = validate_data(*sample_frames(), sample_metadata())

    assert errors == []
    assert reports["row_counts"]["actual"].tolist() == [1, 2, 2]


def test_validate_data_detects_lost_leading_zero_station_id() -> None:
    stations, observations, context = sample_frames()
    observations.loc[0, "station_id"] = "3000"

    _, errors = validate_data(stations, observations, context, sample_metadata())

    assert any("ceros iniciales" in error for error in errors)
