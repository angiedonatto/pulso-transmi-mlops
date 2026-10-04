from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import candidate_l_runtime as runtime
from candidate_l_spectral import MODEL_TYPE, REGIME_START
from cycle_predictions import CyclePrediction, generate_predictions_for_cycle, prediction_trace_payloads
from forecast_cycle_sync import CycleTarget, ForecastCycle
from official_submission import StoredPrediction, build_submission_payload


class Store:
    def __init__(self, observations):
        self.observations = observations
        self.saved = []
        self.manifest = None

    def observations_until(self, cutoff):
        return self.observations

    def persist_l_refit(self, model, manifest):
        self.manifest = manifest

    def persist_cycle_targets(self, cycle):
        pass

    def persist_predictions(self, cycle, model, predictions):
        self.saved.append((dict(model), predictions))
        return len(predictions)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv('CANDIDATE_L_REFIT_DIR', str(tmp_path))
    times = pd.date_range(pd.Timestamp(REGIME_START)+pd.Timedelta(minutes=15), periods=64, freq='15min')
    stations = [f'{i:05d}' for i in range(12)]
    obs = pd.concat([pd.DataFrame({'station_id': s, 'observed_at': times,
                                  'demand': 300+100*np.sin(np.arange(64)*2*np.pi/24)}) for s in stations])
    cutoff = times[-1].isoformat().replace('+00:00', 'Z')
    targets = [CycleTarget(s, (times[-1]+pd.Timedelta(minutes=h)).isoformat().replace('+00:00','Z'), h, i*4+j, {})
               for i,s in enumerate(stations) for j,h in enumerate([15,30,45,60])]
    cycle = ForecastCycle('test-l-cycle', cutoff, '2026-10-05T00:00:00Z', 48, targets, {})
    model = {'model_id': 'l-id', 'run_id': 'l-run', 'model_type': MODEL_TYPE, 'is_active': True}
    store = Store(obs)
    def unavailable(*args):
        raise RuntimeError('K comparator unavailable')
    monkeypatch.setattr(runtime, 'build_k', unavailable)
    return cycle, model, store


def test_end_to_end_exact_targets_trace_and_training_provenance(setup):
    cycle, model, store = setup
    result = generate_predictions_for_cycle(cycle, store, model=model)
    assert result.predictions_generated == 48
    assert result.model_id == 'l-id'
    assert store.manifest['cycle_id'] == cycle.cycle_id
    assert model['trained_until'] == pd.Timestamp(cycle.data_cutoff).isoformat()
    predictions = store.saved[-1][1]
    assert len(prediction_trace_payloads(predictions)) == 48
    payload = build_submission_payload(cycle, model, [StoredPrediction(p.station_id,p.target_at,p.prediction) for p in predictions])
    assert len(payload['predictions']) == 48
    assert payload['model']['training_data_end'] == cycle.data_cutoff
    assert payload['model']['version'] == 'l-id'


def test_failed_l_uses_k_identity_in_submission(setup, monkeypatch):
    cycle, model, store = setup
    store.observations = store.observations.iloc[:1]
    k = {'model_id':'k-id','run_id':'k-run','model_type':'shadow_deep_candidate_k',
         'trained_until':cycle.data_cutoff, 'is_active':False}
    predictions = [CyclePrediction(cycle.cycle_id,'k-id',t.station_id,t.target_at,t.horizon_minutes,100) for t in cycle.targets]
    monkeypatch.setattr(runtime,'build_k',lambda *args:(k,predictions,[]))
    result = generate_predictions_for_cycle(cycle,store,model=model)
    assert result.model_id == 'k-id'
    assert model['model_id'] == 'k-id'
    assert store.saved[-1][0]['model_id'] == 'k-id'
    assert store.manifest is None


def test_double_failure_does_not_persist_predictions(setup):
    cycle,model,store = setup
    store.observations = store.observations.iloc[:1]
    with pytest.raises(RuntimeError,match='comparator unavailable'):
        generate_predictions_for_cycle(cycle,store,model=model)
    assert store.saved == []


def test_inactive_l_rejected(setup):
    cycle,model,store = setup
    model['is_active'] = False
    with pytest.raises(ValueError,match='active registered'):
        generate_predictions_for_cycle(cycle,store,model=model)
    assert not store.saved


def test_k_comparator_is_persisted_under_separate_identity(setup,monkeypatch):
    cycle,model,store = setup
    k = {'model_id':'k-id','run_id':'k-run'}
    predictions = [CyclePrediction(cycle.cycle_id,'k-id',t.station_id,t.target_at,t.horizon_minutes,100) for t in cycle.targets]
    monkeypatch.setattr(runtime,'build_k',lambda *args:(k,predictions,[]))
    result = generate_predictions_for_cycle(cycle,store,model=model)
    assert result.model_id == 'l-id'
    assert [m['model_id'] for m,p in store.saved] == ['k-id','l-id']
