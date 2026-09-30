# Automated Retraining / Drift Remediation

## What Is Automated

- `Predict` still runs every 10 minutes, submits the active champion, and writes the monitoring snapshot.
- `Retrain Autopilot` runs hourly and can also be launched manually from GitHub Actions.
- When recent accuracy falls below `82`, the autopilot:
  - rebuilds the monitoring snapshot;
  - diagnoses drift type: performance drop, data distribution drift, residual bias, station-localized drift, or horizon-localized drift;
  - runs safe Candidate I shadow remediation when enabled;
  - compares complete shadow cycles against the champion;
  - writes `reports/retrain_autopilot/latest.md` and uploads it as a workflow artifact.

## What Is Not Fully Automatic

Promotion is intentionally manual. The autopilot never changes `model_versions.is_active` and never submits a challenger to the competition API. This prevents a bad automatic promotion when the recent window is noisy or the challenger has not completed enough resolved shadow cycles.

## Manual Setup In GitHub Actions

1. Confirm repository secrets exist:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `PULSO_API_KEY` only for `Predict`, not for autopilot

2. Enable these workflows in the Actions tab:
   - `Predict`
   - `Shadow Candidate I`
   - `Retrain Autopilot`

3. Run `Retrain Autopilot` manually once:
   - Open GitHub Actions.
   - Select `Retrain Autopilot`.
   - Click `Run workflow`.
   - Keep `threshold = 82`.
   - Keep `run_shadow = true`.

4. Open the uploaded artifact `retrain-autopilot-report`.
   Review `latest.md`.

5. If the recommendation is `manual_promotion_review`, promote only after checking:
   - candidate recent accuracy is above `82`;
   - candidate beats champion on recent complete cycles;
   - no station-critical regression appears in the report;
   - leakage/drift diagnosis is understood.

## Operational Rule

Use this as the gate:

```text
if recent_accuracy < 82:
  diagnose drift
  run shadow remediation
  compare against champion
  promote only after manual approval and only if recent evidence improves
```

## Notes

- The current implementation uses Candidate I as the safe remediation candidate because it is already isolated as shadow and cannot affect official submissions.
- Full research replays such as `candidate_i_lab.py bases/final` are still too heavy for the fast prediction workflow. Run them manually or in a longer dedicated workflow before changing the production champion.
- Candidate G remains the active conservative champion unless a challenger clears the review gate.
