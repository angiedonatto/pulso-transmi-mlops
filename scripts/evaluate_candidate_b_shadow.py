from __future__ import annotations

import argparse
import json

from candidate_b_shadow import SupabaseShadowStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Candidate B shadow predictions against champion predictions.")
    parser.add_argument("--cycle-id", help="Evaluate a single cycle_id.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum recent shadow cycles to evaluate.")
    args = parser.parse_args()

    with SupabaseShadowStore() as store:
        if args.cycle_id:
            evaluations = [evaluation] if (evaluation := store.evaluate_shadow_cycle(args.cycle_id)) else []
        else:
            evaluations = store.shadow_cycle_evaluations(limit=args.limit)

    payload = [
        {
            "cycle_id": evaluation.cycle_id,
            "champion_accuracy": evaluation.champion_accuracy,
            "candidate_b_accuracy": evaluation.candidate_b_accuracy,
            "delta_pp": evaluation.delta_pp,
            "stations_candidate_b_wins": evaluation.stations_candidate_b_wins,
            "stations_candidate_b_loses": evaluation.stations_candidate_b_loses,
            "coverage": evaluation.coverage,
        }
        for evaluation in evaluations
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
