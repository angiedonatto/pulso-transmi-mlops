"""Trace official forecast cycles that were never submitted (read-only).

For every hourly cutoff missing from `forecast_cycles` inside the analysed range, rebuild the
pipeline stage by stage with direct evidence:

    cycle opened (stream release tick of the cutoff slot)
    -> Predict runs inside the cycle's submission window (GitHub Actions, via `gh`)
    -> run outcome and exception (run logs)
    -> predictions persisted (Supabase `predictions`)
    -> submission attempted / receipt (Supabase `submission_receipts`)

Needs `gh` authenticated for the repository. Never writes to Supabase or GitHub.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

REPORT_DIR = ROOT / "reports" / "missed_cycles"
DEFAULT_REPO = "angiedonatto/pulso-transmi-mlops"
STREAM_FILE = ROOT / "data" / "interim" / "collector_audit" / "stream_full.csv"
EXCEPTION_RE = re.compile(r"^([\w.]+(?:Error|Exception|Timeout)):\s?(.*)$")
RUN_START_MARGIN = pd.Timedelta(minutes=5)  # a run triggered just before the tick executes after it opened
FRAME_RE = re.compile(r'File "[^"]*/(src/[\w_]+\.py)", line (\d+), in (\w+)')


def gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def predict_runs(repo: str, days: list[str]) -> pd.DataFrame:
    rows = []
    for day in days:
        out = gh("run", "list", "-R", repo, "--workflow", "Predict", "--created", day, "--limit", "400",
                 "--json", "databaseId,createdAt,conclusion,headSha,event")
        rows.extend(json.loads(out))
    frame = pd.DataFrame(rows).drop_duplicates("databaseId")
    frame["createdAt"] = pd.to_datetime(frame["createdAt"], utc=True)
    return frame.sort_values("createdAt").reset_index(drop=True)


def run_outcome(repo: str, run_id: int) -> dict[str, Any]:
    """Status line and exception of a Predict run, read from its log."""
    log = gh("run", "view", str(run_id), "-R", repo, "--log")
    lines = [re.sub(r"^.*?\d{4}-\d{2}-\d{2}T[\d:.]+Z ", "", line) for line in log.splitlines()]
    status = next((line.split("status=")[1].split()[0] for line in lines if line.startswith("predict complete: status=")), None)
    raised = [f"{m.group(1)}: {m.group(2)}" for line in lines if (m := EXCEPTION_RE.match(line.strip()))]
    exception = raised[-1] if raised else None  # the last one is what aborted the run
    frames = [f"{m.group(1)}:{m.group(2)}:{m.group(3)}" for line in lines if (m := FRAME_RE.search(line))]
    return {"status": status, "exception": exception, "stack": frames}


def git_commit_time(sha: str) -> str | None:
    result = subprocess.run(["git", "show", "-s", "--format=%cI %s", sha], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() or None


def load_supabase() -> dict[str, pd.DataFrame]:
    from run_monitoring_snapshot import SupabaseMonitoringStore

    selects = {
        "forecast_cycles": ("cycle_id,data_cutoff,closes_at,first_seen_at", "data_cutoff.asc"),
        "predictions": ("cycle_id,model_id,created_at", "created_at.asc"),
        "submission_receipts": ("cycle_id,model_id,status,is_official,accepted_at,created_at", "created_at.asc"),
    }
    with SupabaseMonitoringStore(timeout=60.0) as store:
        return {name: store.table(name, select=select, order=order, limit=500_000) for name, (select, order) in selects.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--start", default="2026-09-12T19:00:00Z", help="First cutoff of the analysed range (leaderboard cumulative window).")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    tables = load_supabase()
    cycles = tables["forecast_cycles"].copy()
    for column in ("data_cutoff", "closes_at", "first_seen_at"):
        cycles[column] = pd.to_datetime(cycles[column], utc=True, format="mixed")
    stream = pd.read_csv(STREAM_FILE)
    stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
    stream["released_at"] = pd.to_datetime(stream["released_at"], utc=True, format="mixed")
    tick = stream.groupby("observed_at")["released_at"].min()  # when the cutoff slot was published

    start = pd.Timestamp(args.start)
    known = set(cycles["data_cutoff"])
    grid = pd.date_range(start, cycles["data_cutoff"].max(), freq="h")
    missing = [cutoff for cutoff in grid if cutoff not in known]
    prefix = cycles["cycle_id"].iloc[-1].rsplit("_", 1)[0]
    # submission window observed on the persisted cycles: release tick -> closes_at
    persisted = cycles[cycles["data_cutoff"].isin(tick.index)].assign(opened=lambda f: f["data_cutoff"].map(tick))
    window = (persisted["closes_at"] - persisted["opened"]).median()

    days = sorted({(tick[c] + delta).strftime("%Y-%m-%d") for c in missing for delta in (pd.Timedelta(0), window)})
    runs = predict_runs(args.repo, days)
    predictions = tables["predictions"]
    receipts = tables["submission_receipts"]
    outcome_cache: dict[int, dict[str, Any]] = {}
    rows = []
    for cutoff in missing:
        cycle_id = f"{prefix}_{cutoff:%Y%m%dT%H%M%SZ}"
        opened = tick.get(cutoff)
        inside = runs[(runs["createdAt"] >= opened - RUN_START_MARGIN) & (runs["createdAt"] <= opened + window)] if opened is not None else runs.iloc[:0]
        outcomes = []
        for run in inside.itertuples(index=False):
            if run.databaseId not in outcome_cache:
                outcome_cache[run.databaseId] = run_outcome(args.repo, run.databaseId)
            outcomes.append({"run_id": run.databaseId, "created_at": run.createdAt.isoformat(), "conclusion": run.conclusion, "sha": run.headSha[:7], **outcome_cache[run.databaseId]})
        exceptions = sorted({o["exception"] for o in outcomes if o["exception"]})
        key_error_runs = sum("not in index" in str(o["exception"]) for o in outcomes)
        network_runs = sum("Timeout" in str(o["exception"]) or "ConnectError" in str(o["exception"]) for o in outcomes)
        stacks = [o["stack"] for o in outcomes if o["stack"]]
        n_pred = int((predictions["cycle_id"] == cycle_id).sum())
        receipt = receipts[receipts["cycle_id"] == cycle_id]
        attempted = bool(len(receipt)) or any(frame.startswith("src/official_submission.py") and "submit_for_cycle" in frame for stack in stacks for frame in stack)
        rows.append({
            "cycle_id": cycle_id,
            "cycle_id_source": "inferred (never persisted)" if cycle_id not in set(cycles["cycle_id"]) else "forecast_cycles",
            "data_cutoff": cutoff.isoformat(),
            "cycle_opened_at": opened.isoformat() if opened is not None else None,
            "window_closes_at": (opened + window).isoformat() if opened is not None else None,
            "predict_runs_during_window": len(outcomes),
            "runs_failed": sum(o["conclusion"] == "failure" for o in outcomes),
            "runs_failed_g_feedback_keyerror": key_error_runs,
            "runs_failed_network_timeout": network_runs,
            "run_shas": ",".join(sorted({o["sha"] for o in outcomes})),
            "predictions_generated": n_pred,
            "submission_attempted": attempted,
            "receipt_exists": bool(len(receipt)),
            "exceptions": " | ".join(exceptions),
            "failing_frame": next((stack[-1] for stack in stacks if "candidate_g_adaptive" in stack[-1]), stacks[0][-1] if stacks else None),
            "network_failing_frames": sorted({stack[-1] for o, stack in ((o, o["stack"]) for o in outcomes) if stack and "Timeout" in str(o["exception"])}),
            "runs": outcomes,
        })
    table = pd.DataFrame(rows)
    shas = sorted({sha for value in table["run_shas"] for sha in str(value).split(",") if sha})
    commits = {sha: git_commit_time(sha) for sha in shas}
    fix = git_commit_time("023ef71")

    def classify(row: pd.Series) -> pd.Series:
        if row["predict_runs_during_window"] == 0:
            return pd.Series({"failure_stage": "no Predict run in the window", "root_cause": "workflow did not run", "avoidable": True,
                              "recommended_fix": "alert when an open cycle has no run; schedule redundancy"})
        if row["predictions_generated"] == 0 and "candidate_g_adaptive.py" in str(row["failing_frame"]) or "not in index" in str(row["exceptions"]):
            return pd.Series({
                "failure_stage": "prediction generation (Candidate G feedback reconstruction), before persistence/payload/submission",
                "root_cause": "build_candidate_f_base_frame received raw Supabase observations (station_id, observed_at, demand) without _prepare_observations; "
                              "the production-only G2 feedback path (runs once G has earlier predictions) raised KeyError on the calendar columns in every attempt; "
                              f"fixed by 023ef71 ({fix})",
                "avoidable": True,
                "recommended_fix": "fallback chain so an open cycle is never left without a submission; end-to-end test of >= 2 consecutive cycles before promotion; alert on failed runs with an open cycle",
            })
        return pd.Series({"failure_stage": "undetermined", "root_cause": row["exceptions"] or "unknown", "avoidable": None, "recommended_fix": "inspect run logs"})

    table = pd.concat([table, table.apply(classify, axis=1)], axis=1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    columns = ["cycle_id", "cycle_opened_at", "predict_runs_during_window", "predictions_generated", "submission_attempted",
               "receipt_exists", "failure_stage", "root_cause", "avoidable", "recommended_fix", "runs_failed", "runs_failed_g_feedback_keyerror",
               "runs_failed_network_timeout", "exceptions", "failing_frame", "network_failing_frames", "run_shas", "data_cutoff", "cycle_id_source"]
    table[columns].to_csv(out / "missed_cycles.csv", index=False)
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "analysed_cutoffs": [grid[0].isoformat(), grid[-1].isoformat()],
        "missing_cycles": len(table),
        "submission_window_minutes": window / pd.Timedelta(minutes=1),
        "commits": commits,
        "cycles": json.loads(table.to_json(orient="records", date_format="iso")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "latest.md").write_text(render(table, summary, columns) + "\n")
    print(json.dumps({"missing_cycles": len(table), "stages": table["failure_stage"].value_counts().to_dict()}))


def render(table: pd.DataFrame, summary: dict[str, Any], columns: list[str]) -> str:
    lines = [
        "# Missed official cycles — trace",
        "",
        f"- Generated: `{summary['generated_at_utc']}` · cutoffs analysed {summary['analysed_cutoffs'][0]} → {summary['analysed_cutoffs'][1]} · read-only (Supabase GET, `gh` logs).",
        f"- Missing cycles: **{summary['missing_cycles']}**. Submission window per cycle (release tick → closes_at, median of persisted cycles): {summary['submission_window_minutes']:.1f} min.",
        f"- Commits that ran in those windows: " + "; ".join(f"`{k}` {v}" for k, v in summary["commits"].items()),
        "",
        "| " + " | ".join(columns[:13]) + " |",
        "|" + "---|" * 13,
    ]
    for row in table[columns[:13]].itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Runs per cycle", ""]
    for record in summary["cycles"]:
        lines.append(f"- `{record['cycle_id']}`: " + "; ".join(f"{r['created_at'][11:16]}Z {r['conclusion']} ({r['sha']}) {r['exception'] or r['status']}" for r in record["runs"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
