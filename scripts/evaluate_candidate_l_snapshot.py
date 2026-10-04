"""Read-only paired replay of frozen L against saved K predictions.

Input files are explicitly supplied; this script has no network or database writes.
The development snapshot is not an independent holdout or official leaderboard.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from candidate_l_spectral import predict
from drift_monitor import accuracy_summary


def evaluate(observations, rows):
    rows = rows.copy()
    for col in ['created_at', 'data_cutoff', 'target_at']:
        rows[col] = pd.to_datetime(rows[col], utc=True)
    outputs = []
    manifests = {}
    for cycle, group in rows.groupby('cycle_id'):
        forecast, manifest = predict(
            observations, group.data_cutoff.iloc[0], group,
            available_at=group.created_at.min(),
        )
        paired = group.merge(forecast.rename(columns={'prediction': 'candidate_l'}),
                             on=['station_id', 'target_at', 'horizon_minutes'],
                             validate='one_to_one')
        outputs.append(paired)
        manifests[cycle] = manifest
    paired = pd.concat(outputs, ignore_index=True)
    scored = paired[paired.actual.notna()]

    def score(frame, column):
        return accuracy_summary(frame.assign(abs_error=(frame[column]-frame.actual).abs()))

    summary = {'scope': 'development replay, not independent validation or official score',
               'expected': len(paired), 'revealed': len(scored),
               'K': score(scored, 'prediction'), 'L': score(scored, 'candidate_l'),
               'cycles': {}}
    for cycle, group in paired.groupby('cycle_id'):
        actual = group[group.actual.notna()]
        summary['cycles'][cycle] = {'expected': len(group), 'revealed': len(actual),
                                    'K': score(actual, 'prediction'), 'L': score(actual, 'candidate_l')}
    return summary, manifests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--created-since', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    obs = pd.DataFrame(json.loads(Path(args.observations).read_text()),
                       columns=['station_id', 'observed_at', 'demand', 'ingested_at'])
    rows = pd.DataFrame(json.loads(Path(args.predictions).read_text()))
    rows = rows[pd.to_datetime(rows.created_at, utc=True) >= pd.Timestamp(args.created_since)]
    summary, manifests = evaluate(obs, rows)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
    (destination / 'training_manifests.json').write_text(json.dumps(manifests, indent=2))
    print(json.dumps(summary, indent=2))
