"""Register Candidate J v1.1 in model_versions as a SHADOW model (is_active=false).

`--build-artifact` freezes J v1.1 into artifacts/shadow/candidate_j_v1_1/metadata.json from the
replay summary (reports/candidate_j/summary.json). Dry-run by default: prints the rows it would
insert. `--apply` writes them, and only if the J code and artifact are committed (so
pipeline_runs.git_commit is real provenance). It never changes is_active of any model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_j_shadow as cjs  # noqa: E402

REPLAY_SUMMARY = ROOT / "reports" / "candidate_j" / "summary.json"
ARTIFACT = ROOT / "artifacts" / "shadow" / cjs.MODEL_VERSION / "metadata.json"
PROVENANCE_PATHS = ["src/candidate_j_regime.py", "src/candidate_j_shadow.py", f"artifacts/shadow/{cjs.MODEL_VERSION}/metadata.json"]


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
    parser.add_argument("--build-artifact", action="store_true", help="Freeze the artifact from the replay summary.")
    parser.add_argument("--apply", action="store_true", help="Insert into Supabase (default: dry-run).")
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file.")
    args = parser.parse_args()

    if args.build_artifact:
        metadata = cjs.build_artifact_metadata(json.loads(REPLAY_SUMMARY.read_text()))
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata = cjs.load_artifact()
    payload = cjs.model_version_payload(metadata)
    provenance = git_provenance()
    plan = {"mode": "apply" if args.apply else "dry-run", "model_versions": payload, "provenance": provenance,
            "pipeline_runs": {"status": "running -> success", "trigger_type": "shadow_registration", "git_commit": provenance["git_commit"]}}
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return
    if not provenance["clean"]:
        raise SystemExit(f"refusing to register: commit {PROVENANCE_PATHS} first (artifacts/ needs `git add -f`). {provenance}")
    if args.env_file:
        _load_supabase_env(args.env_file)
    with cjs.SupabaseCandidateJShadowStore() as store:
        result = store.register_model_version(payload, git_commit=str(provenance["git_commit"]))
    print(json.dumps({**plan, "result": result}, indent=2, sort_keys=True, default=str))


def _load_supabase_env(path: Path) -> None:
    import os

    for line in path.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
            os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
