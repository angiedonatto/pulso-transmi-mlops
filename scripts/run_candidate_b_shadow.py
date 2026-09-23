from __future__ import annotations

import json

from candidate_b_shadow import SupabaseShadowStore, run_shadow_cycle
from forecast_cycle_sync import PulsoForecastCycleApi


def main() -> None:
    with PulsoForecastCycleApi() as api, SupabaseShadowStore() as store:
        result = run_shadow_cycle(api, store)
    print(
        json.dumps(
            {
                "status": result.status,
                "cycle_id": result.cycle_id,
                "data_cutoff": result.data_cutoff,
                "model_id": result.model_id,
                "expected_predictions": result.expected_predictions,
                "predictions_generated": result.predictions_generated,
                "artifact_uri": result.artifact_uri,
                "target_validation": result.target_validation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
