# Phase Q4: exact-payload clean-frame deduplication mechanism test

**Status: complete. Primary disposition -- H-Q4 NOT SUPPORTED.**  Run locally
on August 2, 2026 with `venv312`, the packet-level UNSW-NB15 Payload-Byte CSV,
and seeds `42, 123, 456, 789, 1024`.

Artifact: `results/phase_q4_dedup_mechanism.json`.

## Question

Is the standing frame's raw-duplicate composition causal for strict-purity
poison capture in the controlled reconstruction?

Q3 found that 50.80% +/- 0.50% of final repeated-member collision mass was
already identical at the raw 1,500-byte payload stage. Q4 tests that mechanism
directly by removing exact duplicate clean payloads before poison generation.
It is not a preprocessing recommendation and is not reported as a reproduction
improvement.

## Preregistered single-variable design

For each seed, Q4 rebuilds the exact standing 5,000-row sample, deduplicates on
all 1,500 payload bytes, retains the first row in seeded sample order, preserves
that order, and performs no backfill. Labels do not participate in the
deduplication key or retention rule. The 10% poison rate is applied to the
reduced frame, so rates rather than raw counts are compared.

Held fixed:

- UNSW-NB15 Payload-Byte as the only dataset;
- `malicious_random_attack`, 60 disjoint transpositions, and the same seed;
- poison rate 0.10;
- threshold 0.4, 30-by-50 raster, two Height and three Radial filtrations;
- cubical persistence, Scaler, persistence entropy, and five amplitude
  summaries, producing the same 60-feature map;
- `OPTICS(min_samples=5, max_eps=2.0)` with the standing `xi` extraction;
- all five confirmation seeds.

The completed Q3 artifact is the control. Q4 replays each control realization
only far enough to prove that its input and poison-mask hashes match Q3, then
runs the expensive TDA and OPTICS stages only for the deduplicated arm.

The two primary mechanism endpoints were frozen before Q4 results:

1. poison rows whose earliest merger occurs at the raw-payload stage, divided
   by all poison rows; and
2. poison coordinate-class obstruction at the final 60-vector.

Strict-100%-purity capture and clean/poison unclustered fractions are secondary.
Raw no-op poison was explicitly expected to remain identical to its retained
source, so the manipulation check was zero repeated exact payloads among clean
rows -- not zero raw collisions in the combined frame.

## Manipulation check

The intervention operated as specified on every seed:

| Seed | Clean before | Clean after | Rows removed | Appended poison |
|---:|---:|---:|---:|---:|
| 42 | 5,000 | 3,942 | 1,058 | 394 |
| 123 | 5,000 | 3,974 | 1,026 | 397 |
| 456 | 5,000 | 4,019 | 981 | 401 |
| 789 | 5,000 | 3,991 | 1,009 | 399 |
| 1024 | 5,000 | 3,944 | 1,056 | 394 |

Mean clean-frame size was 3,974 +/- 29 rows; 1,026 +/- 29 rows were removed.
Every deduplicated clean frame had exactly zero repeated-member and redundant
raw-payload rows. The standing replay matched Q3's input and poison-mask hashes
on all five seeds, and every instrumented Q4 feature matrix was bitwise equal to
the production 60-vector.

There were 117.2 +/- 5.2 repeated payload classes per seed containing more than
one dataset label before deduplication. The first-occurrence rule remained
label-free, but this conflict is a threat to interpreting deduplication as a
neutral data-cleaning operation: exact payload equality does not imply label
equality in this packet-level derivative.

## Results

Five-seed means +/- population SD (`ddof=0`):

| Metric | Standing control | Deduplicated frame | Paired delta |
|---|---:|---:|---:|
| Raw first-merger poison rate | 0.1152 +/- 0.0136 | 0.1306 +/- 0.0171 | **+0.0154 +/- 0.0288** |
| Raw first-merger share of collision mass | 0.5080 +/- 0.0050 | 0.0549 +/- 0.0067 | **-0.4531 +/- 0.0075** |
| Final-vector obstruction | 0.4228 +/- 0.0164 | 0.3694 +/- 0.0203 | **-0.0534 +/- 0.0281** |
| Final repeated-member fraction | 0.5633 +/- 0.0068 | 0.4317 +/- 0.0047 | **-0.1316 +/- 0.0030** |
| Poison unclustered fraction | 0.5624 +/- 0.0165 | 0.6458 +/- 0.0129 | **+0.0834 +/- 0.0231** |
| Clean unclustered fraction | 0.3756 +/- 0.0053 | 0.4541 +/- 0.0087 | **+0.0785 +/- 0.0047** |
| Exact-purity poison capture (%) | 1.8000 +/- 0.5060 | 0.3553 +/- 0.7107 | **-1.4447 +/- 1.0224** |

Deduplication removes the clean-clean mass that made the raw stage visually
dominant: its share of final repeated-member collision mass falls from 50.8%
to 5.5%. That denominator-dependent change does **not** remove raw-no-op poison.
The raw first-merger poison rate equals the measured attack raw-no-op rate in
each Q4 arm and rises slightly on average because deduplication changes the
malicious target pool while retaining the same attack generator and seed.

The final-vector obstruction does fall by 5.34 percentage points, so some
clean/poison coordinate sharing is frame-composition dependent. That reduction
does not translate into capture. Four deduplicated seeds capture 0% at strict
purity; seed 1024 captures 1.78%. Instead, OPTICS assigns more of both classes
to label `-1`: poison noise rises 8.34 points and clean noise 7.85 points.

The exhaustive Q3-D decomposition remains complete in every seed. The share of
poison that is non-noise but shares an exact final vector with clean falls from
39.48% to 32.50%, while the unclustered share rises from 56.24% to 64.58%.
Mixed neighborhoods among distinct vectors are essentially unchanged
(2.48% to 2.57%). The mechanism displaced failure from exact coordinate
sharing toward OPTICS noise rather than recovering poisoned samples.

## Disposition

**H-Q4 is not supported as a causal explanation for low capture.** Raw duplicate
clean payloads account for a large fraction of collision mass, but that mass is
not load-bearing in proportion to its size for strict-purity detection. Removing
it reduces total degeneracy and modestly reduces poison obstruction, yet capture
worsens and clustering fragmentation increases.

This sharpens Q3 rather than overturning it:

- raw duplicates are a major source of population-level feature degeneracy;
- single-threshold collapse remains a separate source of clean/poison confusion;
- legacy permutation no-ops survive clean-frame deduplication;
- OPTICS noise assignment, not raw clean-clean duplication, is the dominant
  remaining failure mode after the intervention.

No claim is made that deduplication inherently harms TDA or OPTICS, that the
result generalizes beyond the fixed reconstruction and five seeds, or that it
describes the source study's unrecovered frame. The Q4 result is a mechanism
test on this controlled reconstruction only.

## Reproduction and validation

```powershell
venv312\Scripts\python.exe programs\run_phase_q4_dedup_mechanism.py
venv312\Scripts\python.exe -m unittest tools.test_phase_q4 -v
venv312\Scripts\python.exe tools\phase_q4_summarize.py
```

The runner writes after every completed seed and resumes only when the stored
preregistration matches exactly. Validation checks the five seed keys, Q3
control hashes, no-backfill counts, zero repeated clean payloads, unchanged
attack and OPTICS parameters, bitwise instrumented/production feature equality,
exhaustive failure categories, exact paired deltas, finite values, and
population-standard-deviation summaries.
