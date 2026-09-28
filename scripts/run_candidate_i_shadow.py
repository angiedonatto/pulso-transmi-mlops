"""Candidate I shadow inference (best-effort). Never submits; exits 0 unless --strict."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candidate_i_shadow import (  # noqa: E402
    DEFAULT_TIME_BUDGET_S,
    FAILURE_STATUSES,
    ShadowRunResult,
    SupabaseCandidateIShadowStore,
    run_candidate_i_shadow,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", help="Shadow this cycle instead of the champion's latest one.")
    parser.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET_S, help="Hard limit in seconds for resolve + predict.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on failure/timeout (manual runs only).")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        with SupabaseCandidateIShadowStore() as store:
            result = run_candidate_i_shadow(store, cycle_id=args.cycle_id, time_budget_s=args.time_budget)
    except Exception as exc:  # noqa: BLE001 - store construction / connection errors stay best-effort
        result = ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}")
    payload = {**result.to_dict(), "process_seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps(payload, sort_keys=True, default=str))
    if result.status in FAILURE_STATUSES:
        print(f"::warning title=Candidate I shadow {result.status}::{result.reason}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
