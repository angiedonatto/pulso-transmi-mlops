"""Register Candidate K v2.1 in model_versions as a SHADOW model (is_active=false).

`--build-artifact` freezes K v2.1 into artifacts/shadow/candidate_k_v2_1/: the synthetic-pretrained
local expert (checked against the experiment summary), the research config read from
`candidate_k_deep` (needs PyTorch), the walk-forward procedure, sha256 pins, the parity reference
(per-seed experiment predictions of the 2026-09-18 05:00Z refit) and, if present, the parity result.
Dry-run by default: prints the rows it would insert. `--apply` writes them, and only if the K code
and artifact are committed (so pipeline_runs.git_commit is real provenance). It never changes
is_active of any model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_k_shadow as cks  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "shadow" / cks.MODEL_VERSION
EXPERIMENT = ROOT / "data" / "interim" / "candidate_k"
EXPERIMENT_SUMMARY = ROOT / "reports" / "candidate_k" / "summary.json"
PARITY_REFIT = pd.Timestamp("2026-09-18T05:00:00Z")
PROVENANCE_PATHS = [
    "src/candidate_k_deep.py",
    "src/candidate_k_shadow.py",
    "src/candidate_k_runtime.py",
    f"artifacts/shadow/{cks.MODEL_VERSION}/metadata.json",
    f"artifacts/shadow/{cks.MODEL_VERSION}/{cks.LOCAL_WEIGHTS}",
    f"artifacts/shadow/{cks.MODEL_VERSION}/parity_reference.csv",
]


def build_artifact() -> None:
    import candidate_k_deep as ck  # PyTorch

    summary = json.loads(EXPERIMENT_SUMMARY.read_text())
    expected = summary["local_pretrained"]["sha256"]
    source = EXPERIMENT / "local_pretrained.pt"
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, ARTIFACT / cks.LOCAL_WEIGHTS)
    if cks._sha256(ARTIFACT / cks.LOCAL_WEIGHTS) != expected:
        raise SystemExit("local expert weights do not match the experiment summary")
    if summary["code_sha256"]["src/candidate_k_deep.py"] != cks._sha256(ROOT / cks.FROZEN_MODULE):
        raise SystemExit("src/candidate_k_deep.py changed since the K v2.1 experiment; refusing to freeze")

    runs = pd.read_csv(EXPERIMENT / "runs" / "k_staged_re.csv", dtype={"station_id": str}, parse_dates=["refit_at", "data_cutoff", "target_at"])
    reference = runs[runs["refit_at"] == PARITY_REFIT][["seed", "station_id", "data_cutoff", "target_at", "horizon_minutes", "prediction", "seasonal_weight"]]
    reference.sort_values(["data_cutoff", "station_id", "horizon_minutes", "seed"]).to_csv(ARTIFACT / "parity_reference.csv", index=False)

    parity_file = ARTIFACT / "parity_local.json"
    metadata = cks.build_artifact_metadata(
        config=json.loads(json.dumps(ck.CandidateKConfig(architecture="staged_re").to_dict())),
        artifact_dir=ARTIFACT,
        parity=json.loads(parity_file.read_text()) if parity_file.exists() else None,
        local_pretrain=summary["local_pretrained"],
    )
    metadata["files"]["parity_reference.csv"] = cks._sha256(ARTIFACT / "parity_reference.csv")
    metadata["experiment"] = {
        "variant": "k_staged_re",
        "summary": "reports/candidate_k/summary.json",
        "runs_sha256": cks._sha256(EXPERIMENT / "runs" / "k_staged_re.csv"),
        "snapshot_manifest": summary["snapshot_manifest"]["tables"]["observations"],
        "parity_reference": {"refit_at": cks._ts(PARITY_REFIT), "cycles": int(reference["data_cutoff"].nunique()), "rows": int(len(reference))},
    }
    (ARTIFACT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def git_provenance() -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)

    head = git("rev-parse", "HEAD")
    untracked = [path for path in PROVENANCE_PATHS if git("ls-files", "--error-unmatch", path).returncode != 0]
    dirty = git("status", "--porcelain", "--", *PROVENANCE_PATHS).stdout.strip().splitlines()
    return {"git_commit": head.stdout.strip() if head.returncode == 0 else None, "untracked": untracked, "uncommitted_changes": dirty,
            "clean": head.returncode == 0 and not untracked and not dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-artifact", action="store_true", help="Freeze the artifact from the K v2.1 experiment (needs PyTorch).")
    parser.add_argument("--apply", action="store_true", help="Insert into Supabase (default: dry-run).")
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file.")
    args = parser.parse_args()

    if args.build_artifact:
        build_artifact()
    metadata = cks.load_artifact()
    payload = cks.model_version_payload(metadata)
    provenance = git_provenance()
    plan = {"mode": "apply" if args.apply else "dry-run", "model_versions": payload, "provenance": provenance,
            "pipeline_runs": {"status": "running -> success", "trigger_type": "shadow_registration", "git_commit": provenance["git_commit"]}}
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return
    if not provenance["clean"]:
        raise SystemExit(f"refusing to register: commit {PROVENANCE_PATHS} first (artifacts/ needs `git add -f`). {provenance}")
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                os.environ.setdefault(key, value)
    with cks.SupabaseCandidateKShadowStore() as store:
        result = store.register_model_version(payload, git_commit=str(provenance["git_commit"]))
    print(json.dumps({**plan, "result": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
