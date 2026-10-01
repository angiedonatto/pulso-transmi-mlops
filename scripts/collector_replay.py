"""Collector audit + replay of the last N official cycles (read-only, no submissions).

1. Downloads the full observation stream (GET) and the operational tables from Supabase (GET).
2. Compares stream vs stored data: missing rows, value differences (revisions), duplicates, order.
3. Freshness per official Candidate G cycle: was every slot <= data_cutoff ingested before the
   cycle's predictions were created?
4. Replays the last N cycles with the NEW collector (`stream_sync.sync_observation_stream`) into an
   in-memory store, exposing only the stream rows released before production predicted each cycle,
   and compares the dataset <= data_cutoff: production-at-prediction vs Supabase-now vs replay.
5. Writes `reports/collector_audit/` with a verdict.

Never writes to Supabase, never submits, never touches Predict or `model_versions`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import stream_sync as ss  # noqa: E402
from drift_monitor import accuracy_summary  # noqa: E402
from stream_sync import CursorState, ObservationRow, RetryPolicy, StreamPage  # noqa: E402


REPORT_DIR = ROOT / "reports" / "collector_audit"
INTERIM_DIR = ROOT / "data" / "interim" / "collector_audit"
CHAMPION = "candidate_g_f4_profile_ratio_g2_ewma"
TABLES = {
    "stream_cursors": ("*", None),
    "observations": ("station_id,observed_at,demand,ingested_at", "observed_at.asc,station_id.asc"),
    "forecast_cycles": ("cycle_id,data_cutoff,closes_at,expected_predictions,first_seen_at", "data_cutoff.asc"),
    "forecast_cycle_targets": ("cycle_id,station_id,target_at,horizon_minutes", "cycle_id.asc"),
    "predictions": ("cycle_id,model_id,station_id,target_observed_at,horizon_minutes,prediction,created_at", "created_at.asc"),
    "submission_receipts": ("cycle_id,model_id,status,is_official,accepted_at,created_at", "created_at.asc"),
    "model_versions": ("model_id,model_name,is_active", None),
    "runtime_ingestion_events": ("event_type,severity,created_at,reason_codes,details", "created_at.asc"),
    "pipeline_runs": ("run_id,started_at,status,trigger_type", "started_at.asc"),
}


# --------------------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------------------


def download_stream(limit: int = 5000) -> pd.DataFrame:
    rows, cursor, seen = [], None, set()
    with ss.PulsoStreamApi(timeout=60) as api:
        while True:
            page, _ = ss._fetch_page(api, None, ss.STREAM_NAME, cursor, limit, RetryPolicy(attempts=4), persist_errors=False)
            rows.extend(page.rows)
            if not page.rows or page.next_cursor is None or page.next_cursor in seen:
                break
            seen.add(page.next_cursor)
            cursor = page.next_cursor
    return pd.DataFrame(rows)


def load_supabase() -> dict[str, pd.DataFrame]:
    from run_monitoring_snapshot import SupabaseMonitoringStore

    with SupabaseMonitoringStore(timeout=60.0) as store:
        return {name: store.table(name, select=select, order=order, limit=500_000) for name, (select, order) in TABLES.items()}


def normalize(stream: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    stream = stream.copy()
    stream["station_id"] = stream["station_id"].astype(str).str.zfill(5)
    stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
    stream["released_at"] = pd.to_datetime(stream["released_at"], utc=True, format="mixed")
    out = {}
    for name, frame in tables.items():
        frame = frame.copy()
        if "station_id" in frame:
            frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
        for column in ("observed_at", "ingested_at", "data_cutoff", "first_seen_at", "closes_at", "created_at", "accepted_at", "started_at", "target_at", "target_observed_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed")
        out[name] = frame
    return stream, out


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------


def stream_vs_store(stream: pd.DataFrame, observations: pd.DataFrame) -> dict[str, Any]:
    key = ["station_id", "observed_at"]
    versions = stream.groupby(key).size()
    latest = stream.sort_values("released_at").drop_duplicates(key, keep="last")
    stored = observations[observations["observed_at"] >= stream["observed_at"].min()]
    merged = latest.merge(stored, on=key, how="outer", suffixes=("_stream", "_db"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    grid = pd.date_range(stream["observed_at"].min(), stream["observed_at"].max(), freq="15min")
    stations = sorted(stream["station_id"].unique())
    expected = pd.MultiIndex.from_product([stations, grid])
    ordered = stream.sort_values(["released_at", "observed_at", "station_id"])
    return {
        "stream_rows": int(len(stream)),
        "stream_unique_keys": int(len(versions)),
        "keys_published_more_than_once": int((versions > 1).sum()),
        "stored_rows_in_stream_range": int(len(stored)),
        "missing_in_store": int((merged["_merge"] == "left_only").sum()),
        "only_in_store": int((merged["_merge"] == "right_only").sum()),
        "value_mismatches": int((both["demand_stream"] != both["demand_db"]).sum()),
        "grid_slots_expected": int(len(expected)),
        "grid_slots_missing_in_store": int(len(expected.difference(pd.MultiIndex.from_frame(stored[key])))),
        "release_batches": int(stream["released_at"].nunique()),
        "observed_at_backwards_in_release_order": int((ordered["observed_at"].diff().dt.total_seconds() < 0).sum()),
        "stream_observed_range": [str(stream["observed_at"].min()), str(stream["observed_at"].max())],
    }


def freshness_by_cycle(stream: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    models = t["model_versions"]
    g_id = str(models.loc[models["model_name"] == CHAMPION, "model_id"].iloc[0])
    preds = t["predictions"][t["predictions"]["model_id"].astype(str) == g_id]
    created = preds.groupby("cycle_id")["created_at"].min().rename("pred_created")
    cycles = t["forecast_cycles"].merge(created, left_on="cycle_id", right_index=True, how="inner").sort_values("data_cutoff")
    obs = t["observations"].merge(stream[["station_id", "observed_at", "released_at"]], on=["station_id", "observed_at"], how="left")
    rows = []
    for c in cycles.itertuples(index=False):
        window = obs[obs["observed_at"] <= c.data_cutoff]
        last = obs[obs["observed_at"] == c.data_cutoff]
        late = window[window["ingested_at"] > c.pred_created]
        rows.append({
            "cycle_id": c.cycle_id, "data_cutoff": c.data_cutoff, "pred_created": c.pred_created,
            "last_slot_released": last["released_at"].max(), "last_slot_ingested": last["ingested_at"].max(),
            "slack_minutes": (c.pred_created - last["ingested_at"].max()).total_seconds() / 60 if len(last) else math.nan,
            "rows_le_cutoff_missing_at_prediction": int(len(late)),
            "newest_slot_available_at_prediction": window.loc[window["ingested_at"] <= c.pred_created, "observed_at"].max(),
            "fresh": bool(len(late) == 0),
        })
    return pd.DataFrame(rows)


def missed_cycles(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cycles = t["forecast_cycles"].sort_values("data_cutoff").reset_index(drop=True)
    events = t["runtime_ingestion_events"]
    runs = t["pipeline_runs"]
    rows = []
    for i in range(1, len(cycles)):
        prev, cur = cycles.loc[i - 1], cycles.loc[i]
        gap = int((cur["data_cutoff"] - prev["data_cutoff"]) / pd.Timedelta(hours=1)) - 1
        if gap <= 0:
            continue
        lo, hi = prev["first_seen_at"], cur["first_seen_at"]
        rows.append({
            "missing_cutoffs_from": prev["data_cutoff"] + pd.Timedelta(hours=1), "missing_cutoffs_to": cur["data_cutoff"] - pd.Timedelta(hours=1), "missing_cycles": gap,
            "real_window_from": lo, "real_window_to": hi,
            "collector_runs_in_window": int(((runs["started_at"] > lo) & (runs["started_at"] < hi)).sum()),
            "collector_api_errors_in_window": int(((events["created_at"] > lo) & (events["created_at"] < hi) & (events["severity"] == "critical")).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Replay with the new collector
# --------------------------------------------------------------------------------------


class TimedStreamApi:
    """Serves the real stream in its own order, exposing only rows released at or before `now`.
    Cursor = row offset (pages of `limit` rows, like the real endpoint)."""

    def __init__(self, stream: pd.DataFrame) -> None:
        ordered = stream.sort_values(["released_at", "observed_at", "station_id"]).reset_index(drop=True)
        self.rows = [
            {"station_id": r.station_id, "observed_at": r.observed_at.isoformat().replace("+00:00", "Z"), "demand": int(r.demand), "released_at": r.released_at.isoformat().replace("+00:00", "Z")}
            for r in ordered.itertuples(index=False)
        ]
        self.released = list(ordered["released_at"])
        self.now: pd.Timestamp | None = None
        self.calls = 0

    def visible(self) -> int:
        return sum(1 for stamp in self.released if self.now is None or stamp <= self.now)

    def offset_after(self, observed_through: pd.Timestamp) -> str:
        return str(next((i for i, r in enumerate(self.rows) if pd.Timestamp(r["observed_at"]) > observed_through), len(self.rows)))

    def observations_page(self, *, cursor: str | None, limit: int) -> StreamPage:
        self.calls += 1
        start = int(cursor or 0)
        end = min(self.visible(), start + limit)
        rows = self.rows[start:end]
        return StreamPage(rows=rows, next_cursor=str(end) if end < self.visible() else None, raw_payload={"data": rows})


class ReplayStore:
    """In-memory ObservationStore (same surface as SupabaseRestObservationStore, no network)."""

    def __init__(self, rows: list[ObservationRow], stations: set[str], cursor: str | None) -> None:
        self.rows = {(r.station_id, r.observed_at): r for r in rows}
        self.stations = stations
        self.cursor = cursor
        self.is_exhausted = False
        self.last_observed_at = max((r.observed_at for r in rows), default=None)
        self.validations: list[Any] = []

    def get_cursor(self, stream_name: str = ss.STREAM_NAME) -> CursorState:
        return CursorState(self.cursor, self.last_observed_at, self.is_exhausted)

    def upsert_observations(self, rows: list[ObservationRow]) -> int:
        for row in rows:
            self.rows[(row.station_id, row.observed_at)] = row
        return len(rows)

    def confirm_cursor(self, *, stream_name: str, cursor: str | None, rows_written: int, last_observed_at: str | None, is_exhausted: bool) -> None:
        self.cursor, self.last_observed_at, self.is_exhausted = cursor, last_observed_at, is_exhausted

    def persist_runtime_validation(self, *, stream_name: str, cursor: str | None, result: Any) -> None:
        self.validations.append(result)

    def known_station_ids(self) -> set[str]:
        return self.stations

    def active_model_station_ids(self) -> set[str]:
        return self.stations

    def existing_observations(self, rows: list[ObservationRow]) -> dict[tuple[str, str], int]:
        return {(r.station_id, r.observed_at): self.rows[(r.station_id, r.observed_at)].demand for r in rows if (r.station_id, r.observed_at) in self.rows}

    def recent_observation_history(self, limit: int = 10000) -> list[ObservationRow]:
        return sorted(self.rows.values(), key=lambda r: r.observed_at, reverse=True)[:limit]

    def consecutive_empty_response_count(self, stream_name: str, cursor: str | None) -> int:
        return 0


def replay_last_cycles(stream: pd.DataFrame, t: dict[str, pd.DataFrame], fresh: pd.DataFrame, n_cycles: int, lookback_hours: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resolved_until = t["observations"]["observed_at"].max()
    candidates = fresh[fresh["data_cutoff"] + pd.Timedelta(minutes=60) <= resolved_until].tail(n_cycles)
    first_cutoff = candidates["data_cutoff"].min()
    seed_until = first_cutoff - pd.Timedelta(hours=24)
    stations = set(t["observations"]["station_id"].unique())
    obs = t["observations"]
    seed = [ObservationRow(r.station_id, r.observed_at.isoformat().replace("+00:00", "Z"), int(r.demand)) for r in obs[obs["observed_at"] <= seed_until].itertuples(index=False)]
    api = TimedStreamApi(stream)
    store = ReplayStore(seed, stations, cursor=api.offset_after(seed_until))
    rows, runs = [], []
    for c in candidates.itertuples(index=False):
        api.now = c.pred_created  # exactly what had been released when production predicted
        result = ss.sync_observation_stream(api, store, reconcile_lookback_hours=lookback_hours, retry=RetryPolicy(attempts=1, base_delay_s=0))
        runs.append({"data_cutoff": str(c.data_cutoff), "pages": result.pages, "received": result.received, "written": result.written,
                     "reconcile": vars(result.reconcile) if result.reconcile else None,
                     "observed_through": result.completeness.observed_through if result.completeness else None})
        cut = c.data_cutoff
        after = {(k[0], k[1]): v.demand for k, v in store.rows.items() if pd.Timestamp(k[1]) <= cut}
        now_db = {(r.station_id, r.observed_at.isoformat().replace("+00:00", "Z")): int(r.demand) for r in obs[obs["observed_at"] <= cut].itertuples(index=False)}
        at_pred = {(r.station_id, r.observed_at.isoformat().replace("+00:00", "Z")): int(r.demand) for r in obs[(obs["observed_at"] <= cut) & (obs["ingested_at"] <= c.pred_created)].itertuples(index=False)}
        replay_rows = [ObservationRow(k[0], k[1], v) for k, v in after.items()]
        rows.append({
            "cycle_id": c.cycle_id, "data_cutoff": cut, "pred_created": c.pred_created,
            "rows_production_at_prediction": len(at_pred), "rows_supabase_now": len(now_db), "rows_replay_new_collector": len(after),
            "replay_minus_production_rows": len(set(after) - set(at_pred)), "production_minus_replay_rows": len(set(at_pred) - set(after)),
            "value_differences_replay_vs_production": sum(1 for k in set(after) & set(at_pred) if after[k] != at_pred[k]),
            "value_differences_supabase_vs_production": sum(1 for k in set(now_db) & set(at_pred) if now_db[k] != at_pred[k]),
            "replay_fresh_for_cutoff": ss.freshness_for_cutoff(replay_rows, stations, cut.isoformat().replace("+00:00", "Z"), lookback_hours=lookback_hours)["fresh"],
            "production_fresh_at_prediction": bool(c.fresh),
        })
    return pd.DataFrame(rows), runs


def g_accuracy(t: dict[str, pd.DataFrame], cycle_ids: list[str]) -> pd.DataFrame:
    models = t["model_versions"]
    g_id = str(models.loc[models["model_name"] == CHAMPION, "model_id"].iloc[0])
    preds = t["predictions"][(t["predictions"]["model_id"].astype(str) == g_id) & t["predictions"]["cycle_id"].isin(cycle_ids)]
    actual = t["observations"].rename(columns={"observed_at": "target_observed_at", "demand": "actual"})[["station_id", "target_observed_at", "actual"]]
    scored = preds.merge(actual, on=["station_id", "target_observed_at"], how="inner")
    rows = []
    for cycle_id, group in scored.groupby("cycle_id"):
        summary = accuracy_summary(group.assign(abs_error=(group["prediction"] - group["actual"]).abs()))
        rows.append({"cycle_id": cycle_id, "rows": summary["rows"], "mean_station_accuracy": summary["mean_station_accuracy"], "global_wape_accuracy": summary["global_wape_accuracy"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Verdict (rules fixed before running)
# --------------------------------------------------------------------------------------


def verdict(store_check: dict[str, Any], replay: pd.DataFrame, fresh: pd.DataFrame) -> dict[str, Any]:
    if replay.empty or store_check["stream_rows"] == 0:
        return {"verdict": "insufficient_evidence", "reason": "no stream data or no resolved cycle to replay"}
    window_problems = int(
        (replay["production_minus_replay_rows"] > 0).sum() + (replay["replay_minus_production_rows"] > 0).sum()
        + (replay["value_differences_replay_vs_production"] > 0).sum() + (~replay["production_fresh_at_prediction"]).sum()
    )
    stored_problems = store_check["missing_in_store"] + store_check["value_mismatches"] + store_check["grid_slots_missing_in_store"]
    stale_elsewhere = int((~fresh["fresh"]).sum())
    if window_problems > 0:
        return {"verdict": "collector_issue_confirmed", "reason": f"{window_problems} problems in the replayed cycles' inputs"}
    if stored_problems > 0:
        return {"verdict": "collector_issue_partial", "reason": f"stored data differs from the stream ({stored_problems} rows/slots) outside the replayed cycles"}
    return {
        "verdict": "collector_healthy_model_issue_likely",
        "reason": "the replayed cycles had complete, fresh inputs identical to the stream; the new collector reproduces the same dataset"
                  + (f" ({stale_elsewhere} older cycle(s) predicted on stale data, outside the window)" if stale_elsewhere else ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--lookback-hours", type=float, default=ss.DEFAULT_RECONCILE_LOOKBACK_HOURS)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    raw_stream = download_stream()
    raw_stream.to_csv(INTERIM_DIR / "stream_full.csv", index=False)
    raw_tables = load_supabase()
    for name, frame in raw_tables.items():
        frame.to_csv(INTERIM_DIR / f"supabase_{name}.csv", index=False)
    stream, t = normalize(raw_stream, raw_tables)

    store_check = stream_vs_store(stream, t["observations"])
    fresh = freshness_by_cycle(stream, t)
    missed = missed_cycles(t)
    replay, runs = replay_last_cycles(stream, t, fresh, args.cycles, args.lookback_hours)
    accuracy = g_accuracy(t, list(replay["cycle_id"]))
    errors = t["runtime_ingestion_events"][t["runtime_ingestion_events"]["event_type"] == "api_error"]
    decision = verdict(store_check, replay, fresh)

    out = args.output_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name, frame in {"freshness_by_cycle": fresh, "replay_last_cycles": replay.merge(accuracy, on="cycle_id", how="left"), "missed_cycles": missed, "api_errors": errors}.items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
    cursor = t["stream_cursors"].iloc[0].to_dict() if not t["stream_cursors"].empty else {}
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stream_vs_store": store_check,
        "cursor": {k: str(v) for k, v in cursor.items()},
        "freshness": {"cycles": int(len(fresh)), "fresh": int(fresh["fresh"].sum()), "stale_cycles": [str(c) for c in fresh.loc[~fresh["fresh"], "data_cutoff"]],
                      "median_slack_minutes": float(fresh["slack_minutes"].median())},
        "api_errors": [{"created_at": str(r.created_at), "reason_codes": str(r.reason_codes)} for r in errors.itertuples()],
        "missed_cycles": missed.astype(str).to_dict("records"),
        "replay_runs": runs,
        "verdict": decision,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "latest.md").write_text(render(summary, replay.merge(accuracy, on="cycle_id", how="left"), fresh, missed) + "\n")
    print(json.dumps(decision))


def render(summary: dict[str, Any], replay: pd.DataFrame, fresh: pd.DataFrame, missed: pd.DataFrame) -> str:
    sv, fr = summary["stream_vs_store"], summary["freshness"]
    lines = [
        "# Collector audit — Pulso TransMi",
        "",
        f"- Generated (execution clock): `{summary['generated_at_utc']}` · read-only (GET stream + GET Supabase), no submissions, nothing written.",
        f"- **Verdict: `{summary['verdict']['verdict']}`** — {summary['verdict']['reason']}.",
        "",
        "## Stream vs stored data",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sv.items()]
    lines += [
        "",
        f"## Freshness at prediction time ({fr['fresh']}/{fr['cycles']} Candidate G cycles fresh, median slack {fr['median_slack_minutes']:.1f} min)",
        "",
        f"Stale cycles: {', '.join(fr['stale_cycles']) or 'none'}.",
        "",
        _table(fresh.tail(10), ["data_cutoff", "last_slot_released", "last_slot_ingested", "pred_created", "slack_minutes", "rows_le_cutoff_missing_at_prediction", "fresh"]),
        "",
        f"## Replay of the last {len(replay)} cycles with the new collector (no submissions)",
        "",
        _table(replay, ["data_cutoff", "rows_production_at_prediction", "rows_supabase_now", "rows_replay_new_collector", "replay_minus_production_rows",
                        "production_minus_replay_rows", "value_differences_replay_vs_production", "production_fresh_at_prediction", "replay_fresh_for_cutoff",
                        "mean_station_accuracy", "global_wape_accuracy"]),
        "",
        "## Missed cycles (gaps in forecast_cycles)",
        "",
        _table(missed, list(missed.columns)) if not missed.empty else "none",
        "",
        "## API errors seen by the collector",
        "",
    ]
    lines += [f"- {e['created_at']}: {e['reason_codes']}" for e in summary["api_errors"]] or ["none"]
    return "\n".join(lines)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    columns = [c for c in columns if c in frame.columns]
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in frame[columns].itertuples(index=False):
        out.append("| " + " | ".join(_cell(v) for v in record) + " |")
    return "\n".join(out)


def _cell(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%m-%d %H:%M:%S") + "Z"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    main()
