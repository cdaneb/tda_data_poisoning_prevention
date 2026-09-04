# Downstream classifier extension: preregistration record

## Status and scope

This is a prospective extension of the completed clean-novelty confirmation, not an independent replication. The earlier anomaly-detection outcomes are known. No downstream-classifier outcome may be computed until the exact prepared manifest and the corresponding Git commit have been registered externally and the matching receipt has been added to the repository.

The machine-readable design of record is `results/downstream_classifier_design.json`. The WIRE preparation step will create `results/downstream_classifier_preregistration.json`, which freezes literal training and test rows, corrected attack realizations, hashes, software versions, and the parent-confirmation hash.

## Protocol correction discovered before registration

The shared binary-label helper previously recognized UNSW-NB15 `normal` but did not recognize CICIDS2017 `BENIGN`. Consequently, the earlier CICIDS attack construction did not guarantee malicious parents. This extension fixes the mapping before any downstream outcomes:

- `normal` and `BENIGN`, after whitespace and case normalization, are benign (`0`).
- Every other dataset label is malicious (`1`).
- Every new poison parent must be malicious under that mapping.
- The corrected UNSW attack hashes must exactly reproduce the earlier confirmation.
- Corrected CICIDS attacks must differ from the earlier realizations and are recomputed from scratch; prior CICIDS detection outcomes are not reused as evidence about malicious-only poisoning.

This is a disclosed protocol correction, not an outcome-driven adjustment.

After registration, an existing UNSW TDA feature cache may be reused only if its input, raw matrix, poison mask, 60-feature matrix, and 540-feature matrix hashes all reproduce. This reuses a deterministic representation, not a downstream outcome. Prior CICIDS feature caches are forbidden because their attack populations used the incorrect mapping.

## Locked experiment

The study has 45 population-family-seed cells:

- UNSW matched: 5,000 unmodified rows plus 500 poisons, four attack families, seeds 2026–2030.
- CICIDS matched: 5,000 unmodified rows plus 500 corrected poisons, four attack families, seeds 2026–2030.
- CICIDS scale: 50,000 unmodified rows plus 5,000 corrected transposition poisons, seeds 2026–2030.

Each cell compares six conceptual training arms:

1. Clean: all sampled unmodified rows with correct binary labels.
2. Poisoned: clean rows plus malicious poison copies whose observed training label is benign.
3. Random clean-cost matched removal.
4. Isolation Forest removal using the 60-feature control representation.
5. Isolation Forest removal using the 540-feature multithreshold stack.
6. Oracle: all poisons removed and all unmodified rows retained; by construction this is identical to clean.

The trusted detector core is the original 60% detector-training plus 20% calibration split. The suspect batch is the 20% held-out unmodified split plus appended poison. Filtering acts only on the suspect batch. Random removal uses a fixed ordering and the shortest prefix that removes exactly the same number of unmodified suspect examples as the paired detector; poisons encountered in that prefix are also removed.

The downstream model is a raw-byte `RandomForestClassifier` with 100 trees, the cell seed as `random_state`, eight workers, no preprocessing, and all other scikit-learn defaults. Poisoning is malicious-to-benign dirty-label injection. The byte transformations preserve byte multisets, but packet, protocol, or application functionality is not claimed.

## Test data and endpoints

For each population and seed, WIRE preparation selects a new test set equal in size to the unmodified training population. Selection is deterministic and excludes every candidate whose exact raw-payload SHA-256 occurs in any clean or poisoned training arm. Transformed test copies are also checked against their paired training arm and the unmodified test set. Test rows remain untouched by detector fitting, threshold calibration, sanitizer decisions, and classifier training.

The primary endpoint is malicious recall on the unmodified, identity-safe test set. Benign false-positive rate is the guardrail. Balanced accuracy, macro F1, accuracy, AUROC, AUPRC, training composition, poison retention, per-class unmodified removal, and performance on independently transformed malicious test copies are secondary.

The sole primary analysis is UNSW, the 60-feature Isolation Forest arm, and the 5% calibration budget. An operational-benefit claim requires all four gates:

1. Clean minus poisoned malicious recall is at least 2 percentage points and its 95% hierarchical bootstrap confidence interval excludes zero.
2. IF60 minus poisoned malicious recall is at least 1 percentage point and its interval excludes zero.
3. IF60 minus random-cost-matched malicious recall is positive and its interval excludes zero.
4. The upper confidence bound for IF60 minus clean benign false-positive rate is at most 1 percentage point.

Inference uses 100,000 paired hierarchical bootstrap draws with seed 20260904, resampling the five seeds and then attack families within each sampled seed. IF540, the 1% budget, CICIDS, and every secondary endpoint are non-rescuing analyses. If poisoning does not reduce recall, the recovery fraction is undefined and no recovery claim is made.

## Venue plan

The primary paper target is the [IEEE BigData 2026 Undergraduate and REU Consortium](https://bigdataieee.org/BigData2026/undergraduate-reu-consortium/), whose paper deadline is September 20, 2026, 11:59 p.m. AoE and whose limit is six pages including references. The contingency venue is [IEEE SaTML 2027](https://satml.org/), with abstract registration September 22 and full papers September 29, 2026.

SaTML does not allow simultaneous full-paper submission. Registering a SaTML abstract can preserve the option, but a SaTML full paper should be submitted only if the BigData submission is not active by September 29. A venue decision does not change the frozen experiment or permit retuning.

## Deviations

All failures, incomplete cells, and deviations must be reported. Cells are atomic and resumable, but a completed cell may not be selectively rerun. Any protocol change after registration requires a versioned amendment before affected outcomes are computed.
