"""Candidate J v1.1 shadow inference (best effort). Never submits; exits 0 unless --strict.

Writes J's 48 predictions for every Candidate G cycle since the design freeze that has a complete
G trace and no complete J predictions yet (newest --max-cycles). Runs without PULSO_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidate_j_shadow as cjs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", help="Only this cycle (must be a G cycle after the design freeze).")
    parser.add_argument("--max-cycles", type=int, default=6, help="Catch-up limit per run.")
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file (local runs).")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on failure (manual runs only).")
    args = parser.parse_args()

    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                os.environ.setdefault(key, value)
    started = time.perf_counter()
    try:
        with cjs.SupabaseCandidateJShadowStore() as store:
            result = cjs.run_candidate_j_shadow(store, max_cycles=args.max_cycles, cycle_id=args.cycle_id)
    except Exception as exc:  # noqa: BLE001 - store construction / connection errors stay best effort
        result = cjs.ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}")
    print(json.dumps({**result.to_dict(), "process_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True, default=str))
    failed = result.status in cjs.FAILURE_STATUSES or any(c.status == "failed" for c in result.cycles)
    if failed:
        print(f"::warning title=Candidate J shadow {result.status}::{result.reason or [c.reason for c in result.cycles if c.reason]}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
