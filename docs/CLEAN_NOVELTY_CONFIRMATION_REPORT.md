# Independent clean-novelty confirmation

## Scope and integrity

All 90 frozen cells completed: 40 unseen-seed UNSW matched-size cells, 40 matched-size CICIDS transfer cells, and 10 CICIDS scale cells. All integrity gates passed. Isolation Forest is primary; kNN distance is secondary. No outcome-based detector selection or retuning was performed.

## Findings at the frozen 5% clean-removal operating point

| Population | Detector | Features | Capture | Clean removal | Clean removed | Poison removed | Precision | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unsw_matched | isolation_forest | 60 | 13.99% | 5.19% | 51.6 | 70.0 | 57.63% | 0.787 | 0.570 |
| unsw_matched | isolation_forest | 540 | 14.68% | 4.72% | 46.9 | 73.4 | 61.14% | 0.810 | 0.585 |
| unsw_matched | knn_distance | 60 | 12.70% | 4.73% | 47.0 | 63.5 | 57.63% | 0.815 | 0.591 |
| unsw_matched | knn_distance | 540 | 14.73% | 4.79% | 47.6 | 73.7 | 61.54% | 0.839 | 0.620 |
| cicids_matched | isolation_forest | 60 | 8.02% | 5.98% | 49.8 | 40.1 | 44.40% | 0.640 | 0.468 |
| cicids_matched | isolation_forest | 540 | 7.01% | 5.04% | 42.2 | 35.0 | 45.96% | 0.558 | 0.409 |
| cicids_matched | knn_distance | 60 | 8.85% | 5.52% | 46.0 | 44.2 | 49.01% | 0.738 | 0.539 |
| cicids_matched | knn_distance | 540 | 7.32% | 5.62% | 46.4 | 36.6 | 44.08% | 0.708 | 0.501 |
| cicids_scale | isolation_forest | 60 | 9.36% | 6.86% | 564.0 | 468.0 | 45.40% | 0.644 | 0.461 |
| cicids_scale | isolation_forest | 540 | 9.86% | 6.98% | 574.2 | 493.2 | 46.11% | 0.569 | 0.404 |
| cicids_scale | knn_distance | 60 | 12.13% | 6.60% | 548.4 | 606.4 | 51.98% | 0.824 | 0.599 |
| cicids_scale | knn_distance | 540 | 9.92% | 6.87% | 564.0 | 496.0 | 47.10% | 0.822 | 0.582 |

## Paired 540−60 poison-capture effects

- unsw_matched, isolation_forest: 0.69 pp (95% CI -0.66 to 1.98).
- unsw_matched, knn_distance: 2.03 pp (95% CI 0.99 to 3.07).
- cicids_matched, isolation_forest: -1.01 pp (95% CI -1.68 to -0.35).
- cicids_matched, knn_distance: -1.53 pp (95% CI -2.46 to -0.59).
- cicids_scale, isolation_forest: 0.50 pp (95% CI 0.07 to 0.89).
- cicids_scale, knn_distance: -2.21 pp (95% CI -3.62 to -0.80).

Per-family consistency (positive/negative/zero seed-level paired effects):

- unsw_matched, isolation_forest, transpositions: 4/1/0; mean 1.40 pp (95% CI -1.32 to 3.52).
- unsw_matched, knn_distance, transpositions: 5/0/0; mean 2.60 pp (95% CI 0.88 to 4.32).
- unsw_matched, isolation_forest, block_reversal: 3/2/0; mean 0.60 pp (95% CI -2.20 to 3.20).
- unsw_matched, knn_distance, block_reversal: 3/2/0; mean 2.20 pp (95% CI 0.04 to 4.36).
- unsw_matched, isolation_forest, block_swap: 3/2/0; mean 0.24 pp (95% CI -2.20 to 2.80).
- unsw_matched, knn_distance, block_swap: 3/2/0; mean 1.80 pp (95% CI -0.60 to 4.20).
- unsw_matched, isolation_forest, cyclic_shift: 3/2/0; mean 0.52 pp (95% CI -2.48 to 3.00).
- unsw_matched, knn_distance, cyclic_shift: 3/2/0; mean 1.52 pp (95% CI -0.48 to 3.52).
- cicids_matched, isolation_forest, transpositions: 1/4/0; mean -1.00 pp (95% CI -1.68 to -0.36).
- cicids_matched, knn_distance, transpositions: 1/4/0; mean -1.96 pp (95% CI -3.96 to 0.48).
- cicids_matched, isolation_forest, block_reversal: 2/3/0; mean -0.68 pp (95% CI -2.12 to 0.76).
- cicids_matched, knn_distance, block_reversal: 2/3/0; mean -0.64 pp (95% CI -2.76 to 1.36).
- cicids_matched, isolation_forest, block_swap: 0/5/0; mean -1.84 pp (95% CI -3.28 to -0.92).
- cicids_matched, knn_distance, block_swap: 0/5/0; mean -2.00 pp (95% CI -2.72 to -1.20).
- cicids_matched, isolation_forest, cyclic_shift: 2/3/0; mean -0.52 pp (95% CI -1.96 to 0.92).
- cicids_matched, knn_distance, cyclic_shift: 1/4/0; mean -1.52 pp (95% CI -3.60 to 0.08).
- cicids_scale, isolation_forest, transpositions: 4/1/0; mean 0.50 pp (95% CI 0.07 to 0.89).
- cicids_scale, knn_distance, transpositions: 0/5/0; mean -2.21 pp (95% CI -3.62 to -0.80).

## Frozen decision

The primary representation-improvement claim **fails confirmation**. On unseen-seed UNSW, IF-540 met the clean-cost band and exceeded 12% capture, but its paired improvement over IF-60 did not have a 95% CI excluding zero. Matched-size CICIDS transfer was worse with 540 features on average. CICIDS scale behavior was mixed on capture and failed the frozen clean-cost requirement because mean held-out clean removal exceeded 6%.

The secondary kNN result is mixed: 540 features improve UNSW mean capture, but degrade matched-size CICIDS and CICIDS-scale mean capture at the 5% operating point. Under the preregistration, the secondary detector cannot rescue the failed primary claim.

Figures: `figures/clean_novelty_confirmation_capture.png` and `figures/clean_novelty_confirmation_paired_effects.png`. All values and confidence intervals come from `results/clean_novelty_confirmation.json`.
