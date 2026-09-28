"""Register Candidate I in model_versions as a SHADOW model (is_active=false).

Dry-run by default: prints the pipeline_runs / model_versions rows it would insert.
`--apply` writes them to Supabase, and only if the model code and artifact are committed
(so pipeline_runs.git_commit is real provenance). It never changes is_active of any model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_i_snr_gate as ci  # noqa: E402
from candidate_i_shadow import SupabaseCandidateIShadowStore, candidate_i_model_version_payload  # noqa: E402

PROVENANCE_PATHS = [
    "src/candidate_i_snr_gate.py",
    "src/candidate_i_shadow.py",
    "artifacts/shadow/candidate_i_v1/metadata.json",
]


def git_provenance() -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)

    head = git("rev-parse", "HEAD")
    untracked = [path for path in PROVENANCE_PATHS if git("ls-files", "--error-unmatch", path).returncode != 0]
    dirty = git("status", "--porcelain", "--", *PROVENANCE_PATHS).stdout.strip().splitlines()
    return {
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "untracked": untracked,
        "uncommitted_changes": dirty,
        "clean": head.returncode == 0 and not untracked and not dirty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Insert into Supabase (default: dry-run).")
    args = parser.parse_args()

    artifact = ci.load_candidate_i_artifact(ci.DEFAULT_ARTIFACT_URI)
    payload = candidate_i_model_version_payload(artifact)
    provenance = git_provenance()
    plan = {
        "mode": "apply" if args.apply else "dry-run",
        "pipeline_runs": {"status": "running -> success", "trigger_type": "shadow_registration", "git_commit": provenance["git_commit"], "data_cutoff": payload["trained_until"]},
        "model_versions": payload,
        "provenance": provenance,
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return
    if not provenance["clean"]:
        raise SystemExit(f"refusing to register: commit {PROVENANCE_PATHS} first (artifacts/ needs `git add -f`). {provenance}")
    with SupabaseCandidateIShadowStore() as store:
        registered = store.register_candidate_i_model_version(payload, git_commit=str(provenance["git_commit"]))
    print(json.dumps({**plan, "result": registered}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
