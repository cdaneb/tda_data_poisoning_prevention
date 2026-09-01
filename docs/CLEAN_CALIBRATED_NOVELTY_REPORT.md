# Clean-calibrated label-free detector benchmark

Status: complete. All 40 representation cells (four Frame B families, five seeds, 60/540 features) passed zero-raw-noop, cache-hash, raw-group split, finite-score, and all-detector gates. Detector fitting, standardization, and threshold calibration used trusted clean observations only.

Across all 20 family/seed realizations, mean poison capture at calibration-clean budgets 0.1%, 1%, and 5% was:

| Representation / detector | 0.1% | 1% | 5% | realized clean at 5% |
|---|---:|---:|---:|---:|
| 60 kNN / empirical kNN | 0.49% | 3.34% | 14.02% | 4.83% |
| 60 LOF novelty | 0.44% | 1.01% | 3.73% | 6.01% |
| 60 Isolation Forest | 0.43% | 3.27% | 16.34% | 5.09% |
| 60 One-Class SVM | 0.40% | 3.42% | 8.13% | **21.74%** |
| 540 kNN / empirical kNN | 0.36% | 3.06% | 17.79% | 5.23% |
| 540 LOF novelty | 0.41% | 1.78% | 3.06% | 6.84% |
| 540 Isolation Forest | 0.29% | 2.90% | **18.95%** | 5.35% |
| 540 One-Class SVM | 0.39% | 2.17% | 15.51% | 6.97% |

The ordinary and empirically calibrated kNN arms share the same distance score and clean-quantile rule, so their budget results are mathematically identical; both names are retained to make the preregistered calibration interpretation explicit. At 5%, stack kNN capture was 18.68%, 17.08%, 17.80%, and 17.60% for transpositions, reversal, swap, and cyclic shift, versus 13.52%, 14.16%, 14.08%, and 14.32% in the 60-feature control. At stricter budgets the stack does not improve kNN: mean capture is lower at 0.1% and 1%.

Isolation Forest is the best aggregate 5%-budget arm (18.95% stack), but detector selection was not performed and every detector is reported. One-Class SVM's 60-feature 5% calibration threshold transfers badly to held-out clean (21.74% removal), so its apparent capture is not an operational fixed-cost result. LOF is weak throughout.

Exact poison-vector sharing with clean falls from 3.59% (60) to 0.20% (540), confirming the representation repair. Existing OPTICS nevertheless labels 86.54%/37.16% poison/clean as noise in the control and 99.08%/54.53% in the stack. Clean-calibrated kNN and Isolation Forest therefore improve absolute poison capture at usable 5% clean cost, establishing that fixed OPTICS is a binding limitation there. They do not solve strict-budget detection: capture remains about 0.3--3.4% at 0.1--1% budgets.

All thresholds, realized clean removal, absolute removals, precision, AUROC/AUPRC, score distributions, family/seed values, population SDs, confidence intervals, preprocessing states, exact splits, parent indices, feature hashes, fitted representation states, and runtimes are retained in the JSON/CSV/cell artifacts. No Monkam 126/280 arm was added: the missing 1,000 poison population and lack of a valid full evaluated shared-fit transformation make that comparison invalid.
