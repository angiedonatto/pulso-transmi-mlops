# 2026-10-04: observation schema migration broke fresh inference

## Evidence

The stream began publishing schema_version=2 at virtual time 2026-09-20T12:15Z. Demand is now measurement.value (unit=passengers, quality=observed). The former runtime validator required top-level demand and quarantined every v2 record. observations stopped at virtual 2026-09-20T12:00Z despite successful Predict workflows.

Official K predictions for virtual cutoffs 2026-09-20T13:00Z through 2026-09-21T00:00Z matched J on all 48 targets of each cycle. K traces had feedback_samples=0 and calibration_factor=null, confirming fallback J. This is an ingestion failure; it does not establish that K itself failed on fresh inputs.

The monitor's historical rolling accuracy remained around 86.68 because it could not resolve newer cycles without actuals. It therefore did not represent the user's reported leaderboard rolling-6 of 19.9. Missing quality values prevent reconstructing that complete official score solely from the stored stream.

## Applied recovery

- Commit 3576c4550aad071c39c600cc35f6afc773551f49 accepts observed v2 passenger measurements while preserving raw evidence. It rejects missing, invalid, ambiguous and unsupported measurements. Legacy records remain supported.
- Restored 589 unique, observed, nonnegative integer passenger values directly from audited stream raw_payloads, from virtual 2026-09-20T12:15Z through 2026-09-21T00:30Z. All stations and target time slots already existed; there were no conflicting values. Existing observations were preserved.
- Left 11 quality=missing values absent; no fabricated actuals or changed scoring.
- Marked 28,602 matching obsolete missing_required_field audit events resolved after verifying their values matched restored observations; raw evidence remains stored.
- Predict on the parser fix completed successfully, with no_open_cycle. Standard CI and Candidate K CI both succeeded. Local ingestion regression suite passed 64 tests before the watermark follow-up.
- Follow-up makes validated identical/backfilled observations advance the stream's observation watermark even when they require no write. Cursor token semantics remain unchanged.

K remains the champion. Its native panel already forward-fills missing inputs causally and excludes missing target samples from training. Upcoming forecasts must be verified through receipt and K traces; recovery of future accuracy is not guaranteed until new targets resolve. Do not rewrite historical official predictions.

## Follow-up

Five hourly checks were scheduled during the user's trip. Compare stream watermark to virtual forecast data_cutoff, not wall clock. Verify 48 exact-target submissions and nonzero K train_rows/nonnull seasonal_weight, then report measured accuracy with window and coverage. Promotion/rollback requires comparable causal evidence; ingestion should be repaired first.
