"""Read-only refresh of the local observations snapshot from Supabase.

Pulls the current contents of the production `observations` table (via a single GET
per page against PostgREST -- no writes, ever) and compares it against the local
`data/raw/observations.csv` snapshot Candidate D1 and Challenger Lab v2 were evaluated
against.

Quarantine note: `src/stream_sync.py::sync_observation_stream` only ever upserts
`validation.valid_rows` into `observations` (see `_validate_runtime_page` /
`store.upsert_observations(rows_to_write)`); rows that fail validation are logged to
`runtime_ingestion_events` with `severity='quarantine'` and are never written to
`observations`. Reading `/observations` directly therefore already excludes quarantined
and invalid rows by construction -- no extra filtering is required here.

The refreshed snapshot is written to a NEW file (`data/raw/observations_refresh.csv`),
not overwritten in place, so the original `data/raw/observations.csv` -- and every
number already reported against it (Candidate D1, Challenger Lab v2) -- stays exactly
reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT_DIR = ROOT / "reports" / "challenger_lab_v2_refresh"
OLD_PATH = ROOT / "data" / "raw" / "observations.csv"
NEW_PATH = ROOT / "data" / "raw" / "observations_refresh.csv"


def load_env(path: Path = ROOT / ".env.local") -> None:
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def fetch_supabase_observations() -> pd.DataFrame:
    """Read-only: paginated GET against /observations, exactly like
    candidate_d1_residual_calibration.load_observations_from_supabase_read_only() and
    candidate_c_experiments.load_observations_from_supabase() already do."""
    load_env()
    from candidate_b_shadow import SupabaseShadowStore

    with SupabaseShadowStore() as store:
        frame = store.observations_until("2100-01-01T00:00:00Z")
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def load_old_snapshot() -> pd.DataFrame:
    frame = pd.read_csv(OLD_PATH, dtype={"station_id": "string"})
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def audit_continuity(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_mask = frame.duplicated(subset=["station_id", "observed_at"], keep=False)
    duplicates = int(duplicate_mask.sum())

    gaps: list[dict[str, Any]] = []
    for station_id, group in frame.groupby("station_id", sort=True):
        times = group["observed_at"].sort_values()
        deltas = times.diff().dropna()
        bad = deltas[deltas != pd.Timedelta(minutes=15)]
        for idx, delta in bad.items():
            gaps.append(
                {
                    "station_id": str(station_id),
                    "after": times.loc[times.index < idx].iloc[-1].isoformat().replace("+00:00", "Z") if (times.index < idx).any() else None,
                    "before": times.loc[idx].isoformat().replace("+00:00", "Z"),
                    "gap_minutes": delta.total_seconds() / 60.0,
                }
            )
    return {
        "duplicate_rows": duplicates,
        "stations": int(frame["station_id"].nunique()),
        "gap_count": len(gaps),
        "gaps": gaps[:50],
        "gaps_truncated": len(gaps) > 50,
    }


def audit_history_unchanged(old: pd.DataFrame, new: pd.DataFrame, old_max: pd.Timestamp) -> dict[str, Any]:
    """Confirms the refresh did not retroactively alter any previously-reported row."""
    new_history = new[new["observed_at"] <= old_max].copy()
    old_sorted = old.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    new_history_sorted = new_history.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    if len(old_sorted) != len(new_history_sorted):
        return {"identical": False, "reason": f"row count mismatch: old={len(old_sorted)} new_history={len(new_history_sorted)}"}
    merged = old_sorted.merge(new_history_sorted, on=["station_id", "observed_at"], suffixes=("_old", "_new"), how="outer", indicator=True)
    mismatched = merged[merged["_merge"] != "both"]
    if not mismatched.empty:
        return {"identical": False, "reason": f"{len(mismatched)} rows present in only one of old/new-history"}
    diff = (merged["demand_old"] - merged["demand_new"]).abs()
    max_diff = float(diff.max()) if len(diff) else 0.0
    return {"identical": bool(max_diff < 1e-9), "max_abs_demand_diff": max_diff, "rows_compared": int(len(merged))}


def render_audit_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Challenger Lab v2 Refresh -- Data Refresh Audit",
        "",
        "## Scope",
        "",
        "- Read-only. A single paginated GET against Supabase `/observations` (via `SupabaseShadowStore.observations_until`, the same read path Candidate D1 and Candidate C already use with `--source supabase`). No row was written to Supabase.",
        "- The refreshed snapshot is saved to a NEW file (`data/raw/observations_refresh.csv`); the original `data/raw/observations.csv` was not modified, so every previously-reported number against it stays reproducible.",
        "- Quarantined/invalid rows are excluded by construction: `src/stream_sync.py` only upserts `validation.valid_rows` into `observations`; quarantined events are logged separately to `runtime_ingestion_events` (`severity='quarantine'`) and never reach `observations`. No additional filtering was applied here.",
        "",
        "## Old vs new",
        "",
        f"- `old_observations_max`: {payload['old_observations_max']}",
        f"- `new_observations_max`: {payload['new_observations_max']}",
        f"- `old_row_count`: {payload['old_row_count']}",
        f"- `new_row_count`: {payload['new_row_count']}",
        f"- `new_rows_added`: {payload['new_rows_added']}",
        f"- `stations`: {payload['stations']}",
        "",
        "## Continuity / duplicates (full refreshed dataset)",
        "",
        f"- Duplicate `(station_id, observed_at)` rows: {payload['continuity']['duplicate_rows']}",
        f"- Timestamp gaps (deltas != 15 minutes) across all stations: {payload['continuity']['gap_count']}"
        + (" (showing first 50)" if payload["continuity"]["gaps_truncated"] else ""),
        "",
        "## Historical rows unchanged by the refresh",
        "",
        f"- `identical` (old snapshot vs the <= old_observations_max slice of the new fetch): {payload['history_unchanged']['identical']}",
        f"- Detail: {json.dumps({k: v for k, v in payload['history_unchanged'].items() if k != 'identical'})}",
        "",
        "## Files",
        "",
        "- `data/raw/observations_refresh.csv` (new; full refreshed snapshot)",
        "- `data/raw/observations.csv` (unchanged; original snapshot Candidate D1 / Challenger Lab v2 were evaluated against)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading old local snapshot (data/raw/observations.csv)...", flush=True)
    old = load_old_snapshot()
    old_max = old["observed_at"].max()
    print(f"  old_observations_max = {old_max.isoformat()}, rows = {len(old)}", flush=True)

    print("Fetching current Supabase observations (read-only GET)...", flush=True)
    new = fetch_supabase_observations()
    new_max = new["observed_at"].max()
    print(f"  new_observations_max = {new_max.isoformat()}, rows = {len(new)}", flush=True)

    continuity = audit_continuity(new)
    history_check = audit_history_unchanged(old, new, old_max)

    payload = {
        "old_observations_max": old_max.isoformat().replace("+00:00", "Z"),
        "new_observations_max": new_max.isoformat().replace("+00:00", "Z"),
        "old_row_count": int(len(old)),
        "new_row_count": int(len(new)),
        "new_rows_added": int(len(new)) - int(len(old)),
        "stations": int(new["station_id"].nunique()),
        "continuity": continuity,
        "history_unchanged": history_check,
    }

    new.to_csv(NEW_PATH, index=False)
    (REPORT_DIR / "data_refresh_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (REPORT_DIR / "data_refresh_audit.md").write_text(render_audit_md(payload))

    print(json.dumps(payload, indent=2, default=str))
    print(f"\nRefreshed snapshot written to {NEW_PATH}")
    print(f"Audit written to {REPORT_DIR / 'data_refresh_audit.md'}")

    if not history_check.get("identical", False):
        print("\n*** WARNING: historical rows differ from the original snapshot. Investigate before trusting the refresh. ***")
    if continuity["duplicate_rows"] > 0:
        print(f"\n*** WARNING: {continuity['duplicate_rows']} duplicate rows found in the refreshed data. ***")


if __name__ == "__main__":
    main()
