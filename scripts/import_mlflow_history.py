"""Import immutable historical reports; never train, submit or contact Supabase."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pulso-history-v1"


@dataclass
class HistoricalRun:
    experiment: str
    name: str
    locator: str
    sources: tuple[str, ...]
    metrics: dict[str, float]
    params: dict
    tags: dict[str, str]


def numeric(values: dict) -> dict[str, float]:
    """Keep reported finite numeric values, never synthesize missing metrics."""
    result = {}
    for key, value in values.items():
        if isinstance(value, bool) or value is None or value == "":
            continue
        try:
            number = float(value)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(number):
            raise ValueError(f"Non-finite metric {key}: {value}")
        result[key] = number
    return result


def read_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text())


def read_csv(root: Path, relative: str) -> list[dict]:
    with (root / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_runs(root: Path = ROOT) -> list[HistoricalRun]:
    runs = []
    source = "reports/model_improvement/summary.json"
    summary = read_json(root, source)
    context = {k: v for k, v in summary.items() if k != "models"}
    for model, row in summary["models"].items():
        runs.append(HistoricalRun(
            "Pulso / AB / temporal-walk-forward", model, f"models/{model}", (source,),
            numeric(row), context | {"complexity": row["complexity"]},
            {"evaluation": "temporal_walk_forward", "model_version": "not_reported"}))

    source = "reports/candidate_c/tables/model_trials.csv"
    metadata = "reports/candidate_c/summary.json"
    summary = read_json(root, metadata)
    for row in read_csv(root, source):
        # These are model-selection scores, not the later hard holdout.
        runs.append(HistoricalRun(
            "Pulso / C / walk-forward-selection", f"{row['family']} / {row['trial']}",
            f"trial/{row['family']}/{row['trial']}", (source, metadata),
            {"mean_station_accuracy": float(row["accuracy"])},
            ast.literal_eval(row["params"]) | {
                "walk_forward_start": summary["walk_forward_start"],
                "walk_forward_end": summary["walk_forward_end"], "trial": row["trial"]},
            {"evaluation": "walk_forward_model_selection", "model_version": summary["candidate_version"]}))

    source = "reports/candidate_d1/tables/comparison.csv"
    metadata = "reports/candidate_d1/summary.json"
    summary = read_json(root, metadata)
    for row in read_csv(root, source):
        for prefix, protocol in [("wf", "walk-forward"), ("hard", "hard-holdout")]:
            metrics = {k.removeprefix(prefix + "_"): v for k, v in row.items() if k.startswith(prefix + "_")}
            runs.append(HistoricalRun(
                f"Pulso / D1 / {protocol}", row["model"], f"{prefix}/{row['model']}", (source, metadata),
                numeric(metrics), {"complexity": row["complexity"],
                    "train_seconds_reported": row["train_seconds"],
                    "validation_start": summary["walk_forward_start" if prefix == "wf" else "hard_holdout_start"],
                    "validation_end": summary["walk_forward_end" if prefix == "wf" else "hard_holdout_end"]},
                {"evaluation": "temporal_walk_forward" if prefix == "wf" else "temporal_hard_holdout",
                 "model_version": summary["candidate_version"] if "D1" in row["model"] else "Candidate B v2"}))

    source = "reports/candidate_k/summary.json"
    summary = read_json(root, source)
    for table, dimension in [("overall", "phase"), ("pooled", "window")]:
        for row in summary[table]:
            group, model = row[dimension], row["model"]
            runs.append(HistoricalRun(
                f"Pulso / K / retrospective-{dimension}-{group}", model,
                f"{table}/{group}/{model}", (source,), numeric(row),
                summary["variants"].get(model, {}) | {"variant_config_status": "reported" if model in summary["variants"] else "not_reported",
                 "protocol": summary["protocol"], "evaluation_slice": group},
                {"evaluation": "retrospective_backtest_adaptive_design",
                 "model_version": "k_v2.1" if model == "k_staged_re" else "not_reported",
                 "report_generated_at": summary["generated_at"],
                 "reference_origin": "J replay before J freeze; official predictions after J freeze",
                 "warning": "Design informed by this window; oos_shock is relative to J freeze, not an untouched K holdout"}))

    source = "reports/candidate_l_validation/holdout.md"
    text = (root / source).read_text()
    pattern = r"^\| (2026-[^|]+|Pooled canonical station accuracy) \| (\d+)/(\d+) \| ([\d.]+)% \| ([\d.]+)% \|$"
    rows = re.findall(pattern, text, re.MULTILINE)
    if len(rows) != 3:
        raise ValueError("L holdout table changed; review parser before importing")
    spec = re.search(r"Spec unchanged from `([0-9a-f]{40})`", text)
    if spec is None:
        raise ValueError("Missing L specification commit")
    for cutoff, observed, total, k_accuracy, l_accuracy in rows:
        group = "pooled" if cutoff.startswith("Pooled") else cutoff.strip().replace(" ", "T")
        for model, accuracy in [("K", k_accuracy), ("L", l_accuracy)]:
            runs.append(HistoricalRun(
                f"Pulso / L / retrospective-holdout-{group}", f"Candidate {model}",
                f"holdout/{group}/{model}", (source,),
                {"mean_station_accuracy": float(accuracy), "observed_truth": float(observed), "target_count": float(total)},
                ({"recent_observations": 40, "minimum_observations": 32, "harmonics": 2,
                  "period_search_slots": "8-64", "ridge": 0.05} if model == "L" else {}),
                {"evaluation": "retrospective_untouched_label_holdout", "report_date": "2026-10-04",
                 "model_version": "spec-" + spec.group(1) if model == "L" else "not_reported",
                 "warning": "Two retrospective cycles, incomplete truth; not prospective production or complete leaderboard"}))
    return runs


def provenance(root: Path, run: HistoricalRun) -> dict:
    sources = {}
    for relative in run.sources:
        content = (root / relative).read_bytes()
        revision = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%H", "--", relative],
            capture_output=True, text=True, check=False).stdout.strip()
        sources[relative] = {"sha256": hashlib.sha256(content).hexdigest(),
                             "last_file_commit": revision or "untracked"}
    payload = json.loads(json.dumps({"schema": SCHEMA, "record": asdict(run), "sources": sources}))
    # Commit metadata is evidence, not identity: re-committing identical reports
    # must not duplicate runs. Changed source bytes create a new historical revision.
    identity = {"schema": SCHEMA, "record": asdict(run),
                "source_hashes": {k: v["sha256"] for k, v in sources.items()}}
    payload["import_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return payload


def string_params(params: dict) -> dict[str, str]:
    return {k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            for k, v in params.items()}


def verify_run(client, run_id: str, record: HistoricalRun, payload: dict) -> None:
    actual = client.get_run(run_id)
    if actual.info.status != "FINISHED" or actual.data.metrics != record.metrics:
        raise ValueError(f"Metrics/status mismatch: {record.name}")
    if actual.data.params != string_params(record.params):
        raise ValueError(f"Parameters mismatch: {record.name}")
    if actual.data.tags.get("import_id") != payload["import_id"]:
        raise ValueError(f"Provenance mismatch: {record.name}")
    downloaded = client.download_artifacts(run_id, "provenance.json")
    evidence = json.loads(Path(downloaded).read_text())
    # Same bytes may have been recommitted since import; keep original commit evidence.
    for candidate in (evidence, payload):
        for source in candidate["sources"].values():
            source.pop("last_file_commit", None)
    if evidence != payload:
        raise ValueError(f"Source evidence mismatch: {record.name}")
    for relative, expected in payload["sources"].items():
        downloaded = client.download_artifacts(run_id, str(Path("sources") / relative))
        if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source artifact mismatch: {relative}")


def import_history(root: Path = ROOT, store: Path | None = None) -> dict:
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    root = root.resolve()
    store = (store or root / ".mlflow").resolve()
    store.mkdir(parents=True, exist_ok=True)
    records = load_runs(root)  # Validate sources before creating any run.
    uri = "sqlite:///" + str(store / "mlflow.db")
    created = reused = 0
    with (store / "import.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = MlflowClient(tracking_uri=uri)
        for record in records:
            evidence = provenance(root, record)
            experiment = client.get_experiment_by_name(record.experiment)
            experiment_id = experiment.experiment_id if experiment else client.create_experiment(
                record.experiment, artifact_location=(store / "artifacts" / hashlib.sha256(record.experiment.encode()).hexdigest()[:16]).as_uri())
            found = client.search_runs([experiment_id], filter_string=f"tags.import_id = '{evidence['import_id']}'")
            if len(found) > 1:
                raise ValueError(f"Duplicate import identity: {record.name}")
            if found and found[0].info.status == "FINISHED":
                verify_run(client, found[0].info.run_id, record, evidence)
                reused += 1
                continue
            tags = record.tags | {"mlflow.runName": record.name, "import_id": evidence["import_id"],
                "historical": "true", "execution": "historical_import_no_training",
                "source_path": record.sources[0], "source_locator": record.locator,
                "source_sha256": evidence["sources"][record.sources[0]]["sha256"],
                "source_file_commit": evidence["sources"][record.sources[0]]["last_file_commit"],
                "training_commit": "not_reported", "model_variant": record.name, "import_schema": SCHEMA}
            run_id = found[0].info.run_id if found else client.create_run(experiment_id, tags=tags).info.run_id
            try:
                now = int(time.time() * 1000)  # Import time, not historical training time.
                client.log_batch(run_id, metrics=[Metric(k, v, now, 0) for k, v in record.metrics.items()],
                                 params=[Param(k, v) for k, v in string_params(record.params).items()])
                for relative in record.sources:
                    client.log_artifact(run_id, str(root / relative), artifact_path=str(Path("sources") / Path(relative).parent))
                client.log_dict(run_id, evidence, "provenance.json")
                client.set_terminated(run_id, "FINISHED")
            except Exception:
                client.set_terminated(run_id, "FAILED")
                raise
            verify_run(client, run_id, record, evidence)
            created += 1
    return {"tracking_uri": uri, "created": created, "reused": reused,
            "verified": len(records), "experiments": dict(sorted(Counter(r.experiment for r in records).items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--store", type=Path, help="Local persistent store (default: <root>/.mlflow)")
    args = parser.parse_args()
    print(json.dumps(import_history(args.root, args.store), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
