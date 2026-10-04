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

from monitoring_snapshot import build_monitoring_snapshot  # noqa: E402


TABLE_SELECTS = {
    "observations": "station_id,observed_at,demand",
    "predictions": "cycle_id,model_id,station_id,target_observed_at,horizon_minutes,prediction,created_at",
    "submission_receipts": "cycle_id,model_id,status,is_official,accepted_at,created_at,submission_id,receipt_payload",
    "forecast_cycles": "cycle_id,data_cutoff,closes_at,expected_predictions",
    "forecast_cycle_targets": "cycle_id,station_id,target_at,horizon_minutes,target_order",
    "model_versions": "model_id,is_active,trained_until,model_name,model_type,artifact_uri,params,feature_config",
    "pipeline_runs": "run_id,started_at,finished_at,status,git_commit,data_cutoff,trigger_type,error_message",
    "data_quality_checks": "check_id,check_name,status,severity,actual_value,expected_value,details,checked_at",
    "stream_cursors": "stream_name,confirmed_cursor,confirmed_at,last_observed_at,rows_written",
    "runtime_ingestion_events": "event_type,severity,station_id,observed_at,created_at,reason_codes",
    "prediction_traces": "model_id,cycle_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,feedback_samples,data_cutoff,created_at",
}


class SupabaseMonitoringStore:
    def __init__(self, *, timeout: float = 30.0) -> None:
        _load_env_file()
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{url}/rest/v1",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseMonitoringStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def table(self, name: str, *, select: str, order: str | None = None, limit: int | None = None, page_size: int = 1000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch = min(page_size, limit - offset) if limit is not None else page_size
            if batch <= 0:
                break
            params = {"select": select, "limit": str(batch), "offset": str(offset)}
            if order:
                params["order"] = order
            response = self._client.get(f"/{name}", params=params)
            if response.status_code == 404:
                return pd.DataFrame()
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < batch:
                break
            offset += batch
        return pd.DataFrame(rows)

    def load_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "observations": self.table("observations", select=TABLE_SELECTS["observations"], order="observed_at.desc,station_id.asc", limit=75000),
            # most recent rows: with a shadow model writing 48 rows per cycle too, an ascending
            # capped read would soon return only the oldest cycles
            "predictions": self.table("predictions", select=TABLE_SELECTS["predictions"], order="created_at.desc,prediction_id.asc", limit=10000),
            "submission_receipts": self.table("submission_receipts", select=TABLE_SELECTS["submission_receipts"], order="created_at.desc", limit=2000),
            "forecast_cycles": self.table("forecast_cycles", select=TABLE_SELECTS["forecast_cycles"], order="data_cutoff.desc", limit=2000),
            # The old ascending cap dropped the newest targets after 208 cycles, silently
            # freezing the accuracy window even while submissions and observations advanced.
            "forecast_cycle_targets": self.table("forecast_cycle_targets", select=TABLE_SELECTS["forecast_cycle_targets"], order="target_at.desc,cycle_id.desc,target_order.asc", limit=10000),
            "model_versions": self.table("model_versions", select=TABLE_SELECTS["model_versions"], order="trained_until.asc", limit=500),
            "pipeline_runs": self.table("pipeline_runs", select=TABLE_SELECTS["pipeline_runs"], order="started_at.desc,run_id.asc", limit=1000),
            "data_quality_checks": self.table("data_quality_checks", select=TABLE_SELECTS["data_quality_checks"], order="checked_at.desc,check_id.asc", limit=1000),
            "stream_cursors": self.table("stream_cursors", select=TABLE_SELECTS["stream_cursors"], limit=100),
            "runtime_ingestion_events": self.table("runtime_ingestion_events", select=TABLE_SELECTS["runtime_ingestion_events"], order="created_at.desc,event_id.asc", limit=1000),
            "prediction_traces": self.table("prediction_traces", select=TABLE_SELECTS["prediction_traces"], order="created_at.desc,cycle_id.asc,model_id.asc,station_id.asc,target_at.asc", limit=2000),
        }

    def persist_snapshot(self, row: dict[str, Any]) -> None:
        response = self._client.post("/monitoring_snapshots", json=row)
        response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and persist a canonical monitoring snapshot.")
    parser.add_argument("--output", type=Path, help="Optional local JSON copy.")
    parser.add_argument("--no-persist", action="store_true", help="Build the snapshot without writing to Supabase.")
    args = parser.parse_args()

    with SupabaseMonitoringStore() as store:
        snapshot = build_monitoring_snapshot(**store.load_tables())
        row = snapshot.row()
        if not args.no_persist:
            store.persist_snapshot(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": row["overall_status"], "drift_status": row["drift_status"], "generated_at": row["generated_at"]}, separators=(",", ":")))


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
