from __future__ import annotations

import pandas as pd


BOGOTA_TZ = "America/Bogota"
DAY_NAMES_ES = {
    0: "lunes",
    1: "martes",
    2: "miercoles",
    3: "jueves",
    4: "viernes",
    5: "sabado",
    6: "domingo",
}


def ensure_observed_at_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with observed_at parsed as timezone-aware UTC."""
    result = frame.copy()
    result["observed_at"] = pd.to_datetime(result["observed_at"], utc=True)
    return result


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic temporal features for EDA and baseline modeling."""
    result = ensure_observed_at_utc(frame)
    local_time = result["observed_at"].dt.tz_convert(BOGOTA_TZ)

    result["observed_at_local"] = local_time
    result["date"] = local_time.dt.date.astype("string")
    result["hour"] = local_time.dt.hour
    result["minute"] = local_time.dt.minute
    result["day_of_week"] = local_time.dt.dayofweek
    result["day_name"] = result["day_of_week"].map(DAY_NAMES_ES)
    result["is_weekend"] = result["day_of_week"].isin([5, 6])
    result["quarter_hour_block"] = result["hour"] * 4 + (result["minute"] // 15)
    result["time_block"] = local_time.dt.strftime("%H:%M")
    return result
