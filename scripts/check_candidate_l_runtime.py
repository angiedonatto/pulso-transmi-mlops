"""Read-only hosted validation: train L and validate the latest exact target set.

Never submits or writes to Supabase. Uses the production adapter with a capture store.
"""
import json
from pathlib import Path

from candidate_l_spectral import MODEL_NAME, MODEL_TYPE
from cycle_predictions import SupabaseCyclePredictionStore, generate_predictions_for_cycle
from forecast_cycle_sync import ForecastCycle, CycleTarget
from official_submission import StoredPrediction, build_submission_payload


class ReadOnlyStore(SupabaseCyclePredictionStore):
    enable_k_comparator = False

    def persist_cycle_targets(self, cycle):
        pass

    def persist_l_refit(self, model, manifest):
        self.manifest = manifest

    def persist_predictions(self, cycle, model, predictions):
        if model.get('model_type') == MODEL_TYPE:
            self.l_predictions = predictions
        return len(predictions)


def main():
    with ReadOnlyStore() as store:
        response = store._client.get('/forecast_cycles', params={'order':'data_cutoff.desc','limit':'1'})
        response.raise_for_status()
        row = response.json()[0]
        response = store._client.get('/forecast_cycle_targets', params={'cycle_id':f"eq.{row['cycle_id']}",'order':'target_order.asc'})
        response.raise_for_status()
        targets = [CycleTarget(r['station_id'],r['target_at'],r['horizon_minutes'],r['target_order'],r['raw_target']) for r in response.json()]
        cycle = ForecastCycle(row['cycle_id'],row['data_cutoff'],row['closes_at'],row['expected_predictions'],targets,row['raw_cycle'])
        model = {'model_id':'00000000-0000-4000-8000-000000000001','run_id':'00000000-0000-4000-8000-000000000002',
                 'model_name':MODEL_NAME,'model_type':MODEL_TYPE,'is_active':True}
        result = generate_predictions_for_cycle(cycle,store,model=model)
        assert result.model_id == '00000000-0000-4000-8000-000000000001', 'L unexpectedly fell back to K'
        # Database timestamps can use +00:00; submission requires identical normalization.
        payload = build_submission_payload(cycle,model,[StoredPrediction(p.station_id,p.target_at,p.prediction) for p in store.l_predictions])
        assert len(payload['predictions']) == cycle.expected_predictions
        print(json.dumps({'status':'passed','cycle_id':cycle.cycle_id,'count':len(payload['predictions']),
                          'training_end':model['trained_until'],'weights_sha256':store.manifest['weights_sha256'],
                          'database_writes':0,'submissions':0}))


if __name__ == '__main__':
    main()
