"""Real SQLite tracking tests against the original checked-in historical evidence."""
import csv
import json
import shutil

import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient

from scripts.import_mlflow_history import ROOT, import_history, load_runs, provenance


def test_report_values_and_protocol_boundaries():
    records = load_runs()
    assert len(records) == 108
    assert len({r.experiment for r in records}) == 16
    c = [r for r in records if "/ C /" in r.experiment]
    with (ROOT / "reports/candidate_c/tables/model_trials.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r.metrics["mean_station_accuracy"] for r in c] == [float(r["accuracy"]) for r in rows]
    assert all(set(r.metrics) == {"mean_station_accuracy"} for r in c)
    l = [r for r in records if "/ L /" in r.experiment]
    assert len(l) == 6
    assert next(r for r in l if "pooled" in r.experiment and r.name == "Candidate L").metrics["mean_station_accuracy"] == 91.9613
    assert all("wape" not in r.metrics and "mae" not in r.metrics for r in l)
    d = [r for r in records if "/ D1 /" in r.experiment]
    assert len(d) == 12 and len({r.experiment for r in d}) == 2
    with (ROOT / "reports/candidate_d1/tables/comparison.csv").open() as stream:
        for row in csv.DictReader(stream):
            for prefix, protocol in [("wf", "walk-forward"), ("hard", "hard-holdout")]:
                record = next(r for r in d if r.name == row["model"] and r.experiment.endswith(protocol))
                for key in ["wape", "mae", "bias", "mean_station_accuracy"]:
                    assert record.metrics[key] == float(row[f"{prefix}_{key}"])
    k = [r for r in records if "/ K /" in r.experiment]
    assert len(k) == 81 and all("adaptive_design" in r.tags["evaluation"] for r in k)
    original_k = json.loads((ROOT / "reports/candidate_k/summary.json").read_text())
    for table, dimension in [("overall", "phase"), ("pooled", "window")]:
        for row in original_k[table]:
            record = next(r for r in k if r.locator == f"{table}/{row[dimension]}/{row['model']}")
            for key in ["mean_station_accuracy", "global_wape", "mae", "signed_bias"]:
                assert record.metrics[key] == row[key]
    original_ab = json.loads((ROOT / "reports/model_improvement/summary.json").read_text())
    for record in [r for r in records if "/ AB /" in r.experiment]:
        for key in ["mean_station_accuracy", "mean_station_wape"]:
            assert record.metrics[key] == original_ab["models"][record.name][key]
    for record in records:
        assert record.metrics


def test_import_is_persistent_idempotent_and_exact(tmp_path):
    store = tmp_path / "tracking"
    first = import_history(store=store)
    assert first["created"] == first["verified"] == 108
    assert first["reused"] == 0
    # New client and invocation read persistent SQLite and copied artifacts.
    second = import_history(store=store)
    assert second["created"] == 0 and second["reused"] == 108
    client = MlflowClient(tracking_uri=first["tracking_uri"])
    experiments = client.search_experiments()
    ids = [e.experiment_id for e in experiments if e.name.startswith("Pulso /")]
    runs = client.search_runs(ids, max_results=1000)
    assert len(runs) == 108
    assert all(r.data.tags["historical"] == "true" for r in runs)
    assert len({r.data.tags["import_id"] for r in runs}) == 108
    # Resume an interrupted run rather than creating a duplicate.
    client.set_terminated(runs[0].info.run_id, "FAILED")
    resumed = import_history(store=store)
    assert resumed["created"] == 1 and resumed["reused"] == 107
    # A tampered metric must fail verification, not silently pass as an existing import.
    client.log_metric(runs[0].info.run_id, "unexpected_metric", 123)
    with pytest.raises(ValueError, match="Metrics/status mismatch"):
        import_history(store=store)


def test_source_change_creates_new_revision_not_overwrite(tmp_path):
    record = load_runs()[0]
    copied_root = tmp_path / "copy"
    source = copied_root / record.sources[0]
    source.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / record.sources[0], source)
    original = provenance(copied_root, record)
    source.write_text(source.read_text() + "\n")
    changed = provenance(copied_root, record)
    assert original["import_id"] != changed["import_id"]
    # A new source revision changes identity even if a metric happens to be unchanged.
    assert original["record"]["metrics"] == changed["record"]["metrics"]
