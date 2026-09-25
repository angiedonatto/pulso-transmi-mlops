from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from drift_monitor import DriftConfig, build_drift_report  # noqa: E402


TABLE_SELECTS = {
    "observations": "station_id,observed_at,demand",
    "predictions": "cycle_id,model_id,station_id,target_observed_at,horizon_minutes,prediction,created_at",
    "submission_receipts": "cycle_id,model_id,status,is_official,accepted_at,created_at",
    "forecast_cycles": "cycle_id,data_cutoff,closes_at,expected_predictions",
    "forecast_cycle_targets": "cycle_id,station_id,target_at,horizon_minutes,target_order",
    "model_versions": "model_id,is_active,trained_until,model_name,model_type",
}


class ReadOnlySupabaseClient:
    def __init__(self, *, timeout: float = 30.0) -> None:
        _load_env_file()
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{url}/rest/v1",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ReadOnlySupabaseClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def table(self, name: str, *, select: str, order: str | None = None, page_size: int = 1000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {"select": select, "limit": str(page_size), "offset": str(offset)}
            if order:
                params["order"] = order
            response = self._client.get(f"/{name}", params=params)
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.DataFrame(rows)


def load_monitor_tables() -> dict[str, pd.DataFrame]:
    with ReadOnlySupabaseClient() as client:
        return {
            "observations": client.table("observations", select=TABLE_SELECTS["observations"], order="observed_at.asc,station_id.asc"),
            "predictions": client.table("predictions", select=TABLE_SELECTS["predictions"], order="created_at.asc"),
            "submission_receipts": client.table("submission_receipts", select=TABLE_SELECTS["submission_receipts"], order="created_at.asc"),
            "forecast_cycles": client.table("forecast_cycles", select=TABLE_SELECTS["forecast_cycles"], order="data_cutoff.asc"),
            "forecast_cycle_targets": client.table("forecast_cycle_targets", select=TABLE_SELECTS["forecast_cycle_targets"], order="cycle_id.asc,target_order.asc"),
            "model_versions": client.table("model_versions", select=TABLE_SELECTS["model_versions"], order="trained_until.asc"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline/read-only drift monitor against Supabase data.")
    parser.add_argument("--output", type=Path, help="Optional local JSON report path. Supabase remains read-only.")
    parser.add_argument("--recent-cycles", type=int, default=6)
    parser.add_argument("--reference-cycles", type=int, default=12)
    args = parser.parse_args()

    tables = load_monitor_tables()
    config = DriftConfig(recent_cycles=args.recent_cycles, reference_cycles=args.reference_cycles)
    report = build_drift_report(config=config, **tables)
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


def _load_env_file(path: Path = Path(".env.local")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
