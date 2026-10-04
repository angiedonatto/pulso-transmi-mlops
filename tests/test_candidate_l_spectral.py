import numpy as np
import pandas as pd
import pytest
from candidate_l_spectral import predict, REGIME_START


def fixture():
    times = pd.date_range(pd.Timestamp(REGIME_START) + pd.Timedelta(minutes=15), periods=80, freq="15min")
    x = np.arange(80)
    obs = pd.DataFrame({"station_id": "02300", "observed_at": times,
                        "demand": 300 + 120*np.sin(2*np.pi*x/24) + 35*np.cos(4*np.pi*x/24),
                        "ingested_at": times + pd.Timedelta(days=13)})
    cutoff = times[63]
    targets = pd.DataFrame({"station_id": "02300", "target_at": times[64:68], "horizon_minutes": [15,30,45,60]})
    return obs, cutoff, targets


def test_periodic_signal_and_training_provenance():
    obs, cutoff, targets = fixture()
    frame, info = predict(obs, cutoff, targets)
    assert np.mean(abs(frame.prediction.to_numpy()-obs.demand.iloc[64:68].to_numpy())) < 10
    assert info["stations"]["02300"]["rows"] == 40
    assert pd.Timestamp(info["stations"]["02300"]["training_end"]) <= cutoff
    assert len(info["weights_sha256"]) == 64


def test_future_values_cannot_change_forecast():
    obs, cutoff, targets = fixture()
    a, ia = predict(obs, cutoff, targets)
    obs.loc[obs.observed_at > cutoff, "demand"] = 1000000
    b, ib = predict(obs, cutoff, targets)
    pd.testing.assert_frame_equal(a,b)
    assert ia == ib


def test_missing_values_not_zero_or_training_labels():
    obs, cutoff, targets = fixture()
    obs.loc[[30,40,50],"demand"] = np.nan
    a, ia = predict(obs,cutoff,targets)
    b, ib = predict(obs.dropna(),cutoff,targets)
    pd.testing.assert_frame_equal(a,b)
    assert ia == ib


def test_real_release_time_is_respected():
    obs, cutoff, targets = fixture()
    now = cutoff + pd.Timedelta(days=13)
    obs.loc[60:63,"ingested_at"] = now + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match="stale"):
        predict(obs,cutoff,targets,available_at=now)


def test_requires_real_availability_for_replay():
    obs, cutoff, targets = fixture()
    with pytest.raises(ValueError,match="availability"):
        predict(obs.drop(columns="ingested_at"),cutoff,targets,available_at=cutoff)


def test_insufficient_history_fails_closed():
    obs, cutoff, targets = fixture()
    with pytest.raises(ValueError,match="insufficient"):
        predict(obs[(obs.observed_at<=cutoff)].tail(20),cutoff,targets)


@pytest.mark.parametrize("bad",[-1,float("inf")])
def test_invalid_labels_rejected(bad):
    obs, cutoff, targets = fixture()
    obs.loc[50,"demand"]=bad
    with pytest.raises(ValueError,match="invalid"):
        predict(obs,cutoff,targets)


def test_duplicate_targets_rejected():
    obs, cutoff, targets = fixture()
    with pytest.raises(ValueError,match="duplicate"):
        predict(obs,cutoff,pd.concat([targets,targets.iloc[:1]]))


def test_inconsistent_horizons_rejected():
    obs, cutoff, targets = fixture()
    targets.loc[0,"horizon_minutes"]=30
    with pytest.raises(ValueError,match="horizons"):
        predict(obs,cutoff,targets)


def test_deterministic_and_nonnegative():
    obs, cutoff, targets = fixture()
    a, ia=predict(obs,cutoff,targets)
    b, ib=predict(obs,cutoff,targets)
    pd.testing.assert_frame_equal(a,b)
    assert ia == ib
    assert (a.prediction>=0).all()
