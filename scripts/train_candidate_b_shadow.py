from __future__ import annotations

import argparse
import json

from candidate_b_shadow import DEFAULT_ARTIFACT_URI, MODEL_VERSION, SupabaseShadowStore, train_and_register_candidate_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register fixed Candidate B shadow model version.")
    parser.add_argument("--training-data-end", required=True, help="UTC cutoff used for training, e.g. 2026-09-10T16:30:00Z.")
    parser.add_argument("--version", default=MODEL_VERSION)
    parser.add_argument("--artifact-uri", default=DEFAULT_ARTIFACT_URI)
    args = parser.parse_args()

    with SupabaseShadowStore() as store:
        result = train_and_register_candidate_b(
            store,
            training_data_end=args.training_data_end,
            version=args.version,
            artifact_uri=args.artifact_uri,
        )
    print(
        json.dumps(
            {
                "status": result.status,
                "model_id": result.model_id,
                "version": result.version,
                "artifact_uri": result.artifact_uri,
                "training_data_end": result.training_data_end,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
