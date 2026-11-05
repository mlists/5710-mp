# Generated pinch benchmark results

Generated: 2026-11-05T04:11:35+00:00

Only independently hand-reviewed clips marked COMPLETE are included.
Event tolerance: ±0.50 seconds. Uncertain labels are excluded.
Undefined precision/recall/F1 are BLANK, not perfect scores.

**Gesture detection delay** uses actual capture timestamps and human-labelled contact/separation.
**Frame processing time** starts after camera read and ends after gesture decision.
Neither video playback FPS nor camera-read time is substituted for gesture delay.

## Confirmation ablation (pooled over trials)

| Confirm frames | Positive F1 | Median absolute delay (ms) | Negative false starts/min |
|---:|---:|---:|---:|
| 1 | 1.000 | 36.1 | 5.985 |
| 3 | 1.000 | 112.7 | 3.591 |
| 5 | 1.000 | 191.1 | 3.591 |

## Data files

- per_clip_metrics.csv: event-type and ALL metrics by clip.
- per_condition_metrics.csv: pooled TP/FP/FN and landmark detection rates by condition.
- event_matching_audit.csv: matched/unmatched individual events, delays, excluded predictions.
- replay_predictions.csv: every frame replayed under each confirmation setting.
- confirmation_ablation.csv: pooled summary of the exact same landmark streams.
- table_conditions.tex / table_negative.tex / table_ablation.tex: editable LaTeX tables.
- baseline_replay_consistency.csv: confirms frame-by-frame identity with live 3-frame decision logging.
- summary.json: machine-readable metrics and protocol parameters.

## Limitations

- Event contact timing is limited by frame sampling and manual judgement.
- Capture time stamps are recorded at camera-read return, not at the sensor exposure instant.
- Only the same camera/participant/environment used here are assessed; no generalisation claim is implied.
- Ratios/frequencies and F1 are not physical accuracy in centimetres.
