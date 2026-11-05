# AR chess pinch-gesture benchmark: figure package

Generated from the consolidated gesture benchmark handover (6 November 2026).
Figures are publication-sized. Use PDF for LaTeX and PNG for preview.

## Suggested main-report materials

1. `main/Fig01_F1_by_condition.pdf`: session 2 event-level F1 by condition.
2. `main/Fig02_False_START_rates.pdf`: negative controls, corrected static-fist category.
3. `tables/Tab01_Confirmation_ablation.tex`: small 1/3/5-frame table.

If space is particularly tight, combine Figs 01 and 02 or use only Fig 01 + the ablation table.

## Supplemental diagnostics

- `supplementary/Fig03_Confirmation_ablation.pdf`: graph version of the ablation table.
- `supplementary/Fig04_Hand_tracking_vs_F1.pdf`: high hand presence can coexist with low event recognition.
- `supplementary/Fig05_Manually_reviewed_errors.pdf`: three manually reviewed clips, not all failure cases.
- `supplementary/Fig06_Virtual_chess_event_timing.pdf`: three root episodes for six audit entries.
- `supplementary/Fig07_Pooled_positive_metrics.pdf`: precision/recall/F1 summary.
- Figs 08 and 09 are created only when the relevant raw evaluator CSVs are readable.
- `tables/`: editable CSVs and ready-to-include LaTeX tables.
- `video_contact_sheets/`: created only when `--videos` points to the real videos.

## Tested findings (3-frame baseline)

- Session 1 (six controlled positive conditions): F1 1.000 in each; negative 3.591 START/min (approximate from earlier plot).
- Session 2: 268 TP, 14 FP, 27 FN across 295 annotations; precision 0.950, recall 0.908, F1 0.929.
- Session 2 negative controls: 51 false STARTs over 131.88 s = 23.20/min.
- Session 2 median absolute matched-event delay: 95.5 ms; based on capture timestamps, not full end-to-end display latency.

## Critical interpretation safeguards

- Sessions 1 and 2 use different difficulty levels and negative-action compositions: do **not** treat them as a controlled before/after comparison.
- The positive F1 is an **event** metric, not the percentage of full chess moves completed successfully.
- The negative START rate is per minute, not per fist closure; faster repetitions increase the number of opportunities per minute.
- A genuine physical fingertip contact during a fist action can still be an unintended chess command; intention and contact are different labels.
- The hand-detection percentages rely on hand-visibility annotations; review boundary visibility before claiming precise tracking error rates.
- Correct `negative_fist_static_01` to `fist_static` in `session02/manifest.csv` before using evaluator-generated category charts.
- Virtual chess movement has 4 FP + 2 FN audit entries, traced to three premature START root episodes, not six independent causes.
- No reliable ground-truth frame image can be generated from the numerical handover alone. Use video contact sheets for manual selection.
- All timestamps and condition diagnoses here are based on the supplied handover. Verify against source data before final submission.

## Re-run

```bash
pip install numpy pandas matplotlib opencv-python
python generate_pinch_benchmark_results.py --out pinch_results
python generate_pinch_benchmark_results.py --session2-dir path/to/session02 --out pinch_results
python generate_pinch_benchmark_results.py --videos path/to/videos --out pinch_results
python generate_pinch_benchmark_results.py --data-json pinch_benchmark_data.json --out pinch_results
```

For custom data, edit a copy of `pinch_benchmark_data.json` and provide it using `--data-json`.
The script uses the handover values unless a recognisable local CSV or unambiguous summary.json explicitly updates a metric.
CSV parsing depends on your evaluator column names. Consult `data_provenance.txt` to see exactly what was read.
Use `--strict-csv` to reject incomplete primary CSV parsing instead of retaining handover values.

## Source and parsing log

- Base numerical source: Pinch_Gesture_Report_Handover_REDOWNLOAD(1).docx, 6 November 2026
