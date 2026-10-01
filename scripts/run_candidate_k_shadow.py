"""Candidate K v2.1 shadow inference (best effort). Never submits; exits 0 unless --strict.

Writes K's 48 predictions (and trace rows) for every cycle since the K design freeze whose
data_cutoff is already observed and that has no complete K predictions yet (oldest first, at most
--max-cycles). Runs without PULSO_API_KEY. `--check-only` only lists pending cycles (no PyTorch).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidate_k_shadow as cks  # noqa: E402

KEEP_REFITS = 3


class DryRunStore(cks.SupabaseCandidateKShadowStore):
    """Reads Supabase like the real job but never writes (local validation before registering)."""

    def candidate_k_model_version(self) -> dict | None:
        return super().candidate_k_model_version() or {"model_id": "dry-run", "run_id": "dry-run", "is_active": False}

    def prediction_counts(self, model_id: str, cycle_ids: list[str]):  # type: ignore[override]
        return super().prediction_counts(model_id, cycle_ids) if model_id != "dry-run" else cks.Counter()

    def persist_predictions(self, model, cycle_id, frame, *, data_cutoff, train_rows) -> int:  # type: ignore[override]
        return len(frame)  # the runner already validated the 48 rows; nothing is written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", help="Only this cycle (must be after the K design freeze).")
    parser.add_argument("--max-cycles", type=int, default=12, help="Catch-up limit per run.")
    parser.add_argument("--cache-dir", type=Path, help="Directory for trained daily refits (Actions cache in CI).")
    parser.add_argument("--check-only", action="store_true", help="List pending cycles without predicting (no PyTorch needed).")
    parser.add_argument("--github-output", type=Path, help="Append step outputs (pending / cache_updated) to this file.")
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file (local runs).")
    parser.add_argument("--dry-run", action="store_true", help="Read Supabase and predict, but write nothing (local validation).")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on failure (manual runs only).")
    args = parser.parse_args()

    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                os.environ.setdefault(key, value)
    started = time.perf_counter()
    try:
        with (DryRunStore() if args.dry_run else cks.SupabaseCandidateKShadowStore()) as store:
            result = cks.run_candidate_k_shadow(store, cache_dir=args.cache_dir, max_cycles=args.max_cycles, cycle_id=args.cycle_id, check_only=args.check_only)
    except Exception as exc:  # noqa: BLE001 - store construction / connection errors stay best effort
        result = cks.ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}")
    if args.cache_dir and result.refits_trained:
        _prune(args.cache_dir)
    print(json.dumps({**result.to_dict(), "process_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True, default=str))
    if args.github_output:
        with args.github_output.open("a") as handle:
            handle.write(f"pending={'true' if result.pending and args.check_only else 'false'}\n")
            handle.write(f"cache_updated={'true' if result.refits_trained else 'false'}\n")
    failed = result.status in cks.FAILURE_STATUSES or any(c.status == "failed" for c in result.cycles)
    if failed:
        print(f"::warning title=Candidate K shadow {result.status}::{result.reason or [c.reason for c in result.cycles if c.reason]}")
        if args.strict:
            sys.exit(1)


def _prune(cache_dir: Path) -> None:
    """Keep only the newest refits of each spec (older days are never needed again)."""
    for spec in (p for p in cache_dir.iterdir() if p.is_dir()):
        for old in sorted(p for p in spec.iterdir() if p.is_dir())[:-KEEP_REFITS]:
            shutil.rmtree(old)


if __name__ == "__main__":
    main()
