# Phase Q3: collision-provenance audit

**Status: complete. Primary disposition — (5) MIXED CAUSE, with the topological
feature map effectively exonerated.**
Run 2026-07-29 on WIRE, `venv312`, full UNSW-NB15 Payload-Byte CSV.
Artifact: `results/phase_q3_collision_audit.json` (five seeds, 451 s).

**One-paragraph summary.** Phase Q2's "51.8% exact duplicate rows" is the
**redundancy fraction** `2849/5500`; the repeated-member fraction on the same
frame is **57.02%**. Both reproduce exactly on the frozen R60/seed-42 gate
frame, as does the 1043-row block. Tracing every repeated final-vector class
back to its earliest merger stage: **50.80% ± 0.50 of the collision mass is
already identical in the raw 1500-byte payload**, **46.68% ± 0.46 is first
created by threshold-0.4 binarization**, and **2.52% ± 0.31 originates in
cubical persistence**. Every other stage — the support record, the five
filtration images, the Scaler, the persistence summaries, and the final
concatenation — merges **exactly zero rows on every seed**. So ~97.5% of the
problem is upstream of any topology, and the empty-payload hypothesis is
refuted outright (**0 all-zero payload rows**, all seeds). Collisions are
load-bearing but not dominant: **42.28% ± 1.64 of poisoned rows share an exact
60-vector with a clean row**, while **56.24% ± 1.65 are assigned label `-1`**.

---

## 0. Scope and what was not done

Diagnostic phase only. No row was filtered, no data deduplicated, no metadata
added, no binarization or filtration changed, no OPTICS parameter tuned, no
attack family added, nothing committed or sent. The legacy single-threshold-0.4
pipeline was used throughout, on the exact R60 realization of the standing
regression gate.

The audit is **additive**: `tda_pipeline.py` was not modified. The instrumented
path calls the production `extract_tda_features()` unchanged and then replays
the *already fitted* sub-pipelines with `.transform()` to recover intermediates.
The rebuilt 60-vector is **bitwise equal** to the production output on all five
seeds — asserted, never relaxed to `allclose`. That equality is what licenses
treating the recovered intermediates as the real ones.

**Provenance.** Seed 42 reproduces the Q2 frame exactly:
`input_hash = fce036c2424196ef`, `poison_mask_hash = 8f8ee5534151e4fe`,
`feature_hash(round 9) = ffce38eb0cd462df` — all three match
`results/phase_q2_geometry.json → arms.legacy_30x50_t04`. Exact (unrounded)
feature hash `9e6650402ee5dcdece6a598afdb73b64`. Binarizer `max_value_ = 255.0`
in all five sub-pipelines, effective byte cut **102.0**.

---

## 1. Definitions (the reason Q2's sentence was ambiguous)

Two different statistics can both be called "duplicate fraction". Q3 never uses
that phrase.

| Statistic | Definition | Seed 42, final 60-vector |
|---|---|---:|
| **repeated-member fraction** | rows in a class of size ≥ 2, ÷ all rows | **3136/5500 = 57.02%** |
| **redundancy fraction** | `(n_rows − n_unique_classes) / n_rows` | **2849/5500 = 51.80%** |

They differ by exactly the number of repeated *classes* (287): deleting
redundant copies leaves one representative per class. Redundancy ≤
repeated-member always, and equality holds only when every repeated class has
size 2.

**Answer to Q2's ambiguity: the 51.8% was the redundancy fraction.** It matches
`2849/5500` to machine precision and does not match the repeated-member fraction
(57.02%). Q2 computed it as `len(X) − len(np.unique(X, axis=0))` in
`tools/phase_q2_common.py::feature_diagnostics`, which is the redundancy count.

Other definitions used below:

- **coordinate-class obstruction** — poison rows sharing an exact final
  60-vector with ≥ 1 clean row, ÷ all poison rows.
- **earliest merger stage** — the first stage at which a row's exact
  equivalence class stops being a singleton. Well-defined and mutually
  exclusive because each stage is a deterministic function of the previous one,
  so equality can only be created, never destroyed. Monotonicity is **asserted**:
  `monotonicity_violations == 0` on every seed.
- **diagram canonicalisation** — valid points are those with `death > birth`;
  sorted by `(homology_dimension, birth, death)`. giotto-tda's diagonal padding
  is excluded and array point order carries no information, because a
  persistence diagram is a multiset. A genuine zero-persistence point would also
  be dropped; it is indistinguishable from padding by any consumer of the
  diagram, and this is stated rather than hidden.

Signed zero is normalised and NaN canonicalised before any float hash, so
numerically identical arrays cannot receive different hashes. **Nothing is
rounded before an exact collision is declared.**

---

## 2. Q3-A — the Q2 observation reproduces exactly

| Quantity | Q2 | Q3 | Match |
|---|---:|---:|:--:|
| redundant rows | 2849 | 2849 | ✅ |
| redundancy fraction | 0.5180 | 0.5180 | ✅ |
| largest final class | 1043 | 1043 | ✅ |
| largest class share | 0.190 | 0.1896 | ✅ |
| repeated-member rows | *(not reported)* | 3136 (57.02%) | — |
| repeated classes | *(not reported)* | 287 | — |

Repeated-class size quantiles (seed 42): 5% = 2, 25% = 2, **median = 2**,
75% = 3, 95% = 13.7, max = 1043. The distribution is overwhelmingly pairs, with
one enormous outlier — the 1043 block alone is a third of all repeated-member
mass.

### OPTICS behaviour inside exact coordinate classes

| Seed | Repeated classes | Split across labels | ...across two *real* clusters | ...one real cluster + `-1` |
|---|---:|---:|---:|---:|
| 42 | 287 | 4 | **0** | 4 |
| 123 | 278 | 2 | **0** | 2 |
| 456 | 273 | 6 | **0** | 6 |
| 789 | 271 | 2 | **0** | 2 |
| 1024 | 279 | 0 | **0** | 0 |

**No exact coordinate class is ever split across two distinct real clusters, on
any seed.** The only splits observed are a class where one twin was left at
`-1`. This is the precise scope at which the obstruction can be called a
constraint, and it is why §5 declines to call `1 − obstruction` an
algorithm-independent ceiling.

---

## 3. Q3-B — the 1043-row block, traced upstream

**Verdict: the 1043-row block was NOT already identical at raw input as a whole,
but a majority of it was. Its earliest merger stage is `raw_payload`.**

| | Seed 42 | 123 | 456 | 789 | 1024 |
|---|---:|---:|---:|---:|---:|
| class size | 1043 | 1094 | 1001 | 1090 | 1035 |
| clean members | 950 | 1001 | 906 | 974 | 948 |
| poison members | 93 | 93 | 95 | 116 | 87 |
| poison whose clean source is also in the class | 93 | 93 | 95 | 116 | 87 |
| distinct **raw** signatures | 406 | 422 | 379 | 440 | 391 |
| distinct **binary mask** signatures | 34 | 41 | 32 | 39 | 36 |
| distinct **diagram** signatures | 1 | 1 | 1 | 1 | 1 |
| all-zero payloads | **0** | 0 | 0 | 0 | 0 |

The funnel is **406 → 34 → 1**. Within the class, the earliest-merger histogram
at seed 42 is: 792 rows already raw-identical, 238 first merged by binarization,
13 first merged in cubical persistence.

**The poison members are the interesting part.** Of the 93 poisoned rows in the
class, only **6** are raw-identical to their own clean source packet — the
60-swap permutation did change 87 of them at the byte level. But **all 93 are
binary-identical to their own clean source**, and remain identical through every
downstream stage. The same pattern holds on all five seeds (6/14/14/13/9 raw
versus 93/93/94/114/87 binary).

That is Claim 1 appearing as a measured fact rather than an inference: the
permutation moved bytes, but not across the threshold-102 cut, so the binary
image — and therefore everything downstream — is unchanged. All 1043 members sit
in a single OPTICS cluster (label 116 at seed 42), which is **Yellow**: 950 clean
rows make it unremovable at any purity threshold.

Composition is ordinary traffic, not a degenerate corner: UNSW labels at seed 42
are fuzzers 225, normal 201, exploits 179, generic 124, dos 105, reconnaissance
76, analysis 71, backdoor 62. Median `support_end` is 80 bytes, mean 323, min 1,
max 1500.

*(No payload bytes appear in this report or in the JSON artifact. Row indices in
the artifact are ordinary 0-based positions into the 5500-row combined matrix:
0–4999 subsampled clean, 5000–5499 appended poison.)*

---

## 4. Q3-C — population-wide earliest-merger attribution

**Mutually exclusive by earliest merger stage; `monotonicity_violations = 0` on
every seed; rows sum exactly to the final repeated-member mass.**

Seed 42 (3136 repeated-member rows):

| Earliest equality/merger stage | Repeated classes | Member rows | Clean rows | Poison rows | Mixed-class poison rows | Share of final repeated-member mass |
|---|---:|---:|---:|---:|---:|---:|
| raw padded payload | 489 | 1595 | 1547 | 48 | 48 | **0.5086** |
| supported-payload / padding record | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| **binarization (threshold 0.4)** | 168 | 1468 | 1313 | 155 | 155 | **0.4681** |
| filtration images (all five) | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| unscaled persistence diagrams | 34 | 73 | 73 | 0 | 0 | **0.0233** |
| scaled persistence diagrams | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| 12-feature filtration summaries | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| final concatenation only | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| **TOTAL** | | **3136** | 2933 | 203 | 203 | 1.0000 |

Five-seed means ± population SD:

| Stage | Share of collision mass | Poison rows |
|---|---:|---:|
| raw padded payload | **0.5080 ± 0.0050** | 57.6 ± 6.8 |
| binarization | **0.4668 ± 0.0046** | 153.2 ± 3.2 |
| unscaled diagrams | **0.0252 ± 0.0031** | 0.6 ± 0.8 |
| every other stage | **0.0000** | 0 |

Stage-by-stage class counts (seed 42; five-seed values in the artifact):

| Stage | Unique classes | Repeated classes | Repeated-member frac. | Redundancy frac. | Largest class |
|---|---:|---:|---:|---:|---:|
| raw payload | 4394 | 489 | 0.2900 | 0.2011 | 63 |
| supported-payload record | 4394 | 489 | 0.2900 | 0.2011 | 63 |
| binary mask | 2754 | 317 | 0.5569 | 0.4993 | 898 |
| filtration images | 2754 | 317 | 0.5569 | 0.4993 | 898 |
| unscaled diagrams | 2651 | 287 | 0.5702 | 0.5180 | 1043 |
| scaled diagrams | 2651 | 287 | 0.5702 | 0.5180 | 1043 |
| feature blocks | 2651 | 287 | 0.5702 | 0.5180 | 1043 |
| final 60-vector | 2651 | 287 | 0.5702 | 0.5180 | 1043 |

Only two transitions do any work, on every seed:

- **support record → binary mask:** 174 downstream classes absorb 324 upstream
  classes, 2679 rows, **29 clean/poison mixed classes created here**, largest
  merged class 898.
- **filtration images → unscaled diagrams:** 48 downstream classes absorb
  multiple upstream classes, 1539 rows, **0 new mixed classes**, largest merged
  class 1043.

Every other transition is exactly zero. Note the second one **creates no new
clean/poison confusion at all** — it merges clean rows with clean rows.

### The support record and the padding hypothesis

The supported-payload record merges **exactly 0 rows**, as it must: because
padding is zero by construction, `(support_end, row[:support_end])` determines
the whole row, so the record is a *bijection* with the raw payload. Its row in
the table is a self-check, and it passed. "Padding artifact" is therefore a
possible *description* of the raw-collision class, not a separate merger stage.

**That description is refuted.** There are **0 all-zero payload rows** on every
seed, so the collisions are not empty packets. Raw-repeated rows are *shorter*
than singletons (mean `support_end` 156.6 vs 427.6 at seed 42; the same gap on
all five seeds), but they carry real bytes. The right reading is short,
repeated, protocol-stereotyped payloads — not padding.

### Per-filtration ablation (diagnostic only)

Five-seed means ± population SD. **Q3 does not select, reweight, or drop any
filtration on the basis of this table.**

| 12-feature block | Unique classes | Repeated-member frac. | Largest class | Rows repeated in this block but singleton at final |
|---|---:|---:|---:|---:|
| `height_0_1` | 1761 ± 16 | **0.7060 ± 0.0035** | 1054 ± 35 | **785 ± 34** |
| `height_1_0` | 2561 ± 30 | 0.5868 ± 0.0057 | 1059 ± 36 | 129 ± 12 |
| `radial_0_50` | 2559 ± 34 | 0.5876 ± 0.0072 | 1057 ± 35 | 134 ± 12 |
| `radial_0_25` | 2562 ± 33 | 0.5868 ± 0.0068 | 1058 ± 35 | 129 ± 12 |
| `radial_30_0` | 2545 ± 31 | 0.5884 ± 0.0057 | 1059 ± 36 | 138 ± 7 |

`height_0_1` is the most collision-prone block on all five seeds — it is the
direction that sweeps along the 50-pixel axis, and it alone loses ~800 rows'
worth of distinctions that the other four recover. But **no filtration dominates
the final equality pattern**: every block already contains a ~1055-row class, so
the big block is common to all five and concatenation cannot break it.
Concatenating all five resolves only 129–785 rows per block relative to that
block alone.

---

## 5. Q3-D — strict-100%-purity capture failure, decomposed

Every poisoned row lands in exactly one category (priority order as listed).
Categories sum to 500 on every seed; the residual category is **empty** on every
seed, as predicted — a non-noise poisoned row is in a cluster that is either
100% poison or mixed.

| Category | Seed 42 | Five-seed mean ± SD |
|---|---:|---:|
| 1. captured in a 100%-poison non-noise cluster | 11 (2.20%) | **1.80 ± 0.51%** |
| 2. label `-1` (unclustered) | 292 (58.40%) | **56.24 ± 1.65%** |
| 3. non-noise, shares an exact 60-vector with a clean row | 186 (37.20%) | **39.48 ± 1.73%** |
| 4. non-noise, distinct vector, but in a mixed cluster | 11 (2.20%) | 2.48 ± 0.90% |
| 5. residual, unexplained | **0** | **0.00 ± 0.00%** |

Category 1 reproduces the standing gate: 2.2000% at seed 42, and the five-seed
mean **1.80 ± 0.51** with per-seed `[2.2, 2.2, 2.2, 1.0, 1.4]` matches
`CLAUDE.md` §6's Test B transpositions row exactly.

### Coordinate-class obstruction

| | Seed 42 | Five-seed |
|---|---:|---:|
| poison sharing an exact final vector with ≥1 clean row | 203/500 | — |
| **obstruction fraction** | **0.4060** | **0.4228 ± 0.0164** |
| captured rows *also* in a mixed exact class | **0** | **0 on all seeds** |

The obstruction exceeds category 3 (42.28% vs 39.48%) because ~2.8% of
obstructed rows are *also* at `-1` and were counted there by priority.

**The obstruction binds under this fit**, verified empirically: no captured row
is in a mixed exact-vector class, on any seed. But `1 − obstruction` is
**not** claimed as an algorithm-independent capture ceiling. It is a ceiling
exactly to the extent that the clustering rule cannot split identical
coordinates, and §2 shows OPTICS *can* separate identical twins by leaving one
at `-1`. The claim is therefore made at the scope of the standing
`OPTICS(min_samples=5, max_eps=2.0)` fit and no wider.

### Comparison of mechanisms

| Mechanism | Poison mass affected |
|---|---:|
| `-1` assignment | **56.24 ± 1.65%** |
| exact clean/poison vector sharing | **42.28 ± 1.64%** |
| mixed neighbourhood among *distinct* vectors | 2.48 ± 0.90% |
| actually captured | 1.80 ± 0.51% |

Clean rows are unclustered at 37.56 ± 0.53%, poison at 56.24 ± 1.65%.

---

## 6. Q3-E — five-seed stability

Every mechanism is stable; nothing here is seed-42-driven.

| Quantity | Mean ± population SD | Per seed |
|---|---:|---|
| raw repeated-member fraction | 0.2862 ± 0.0060 | 0.290, 0.287, 0.276, 0.284, 0.294 |
| raw redundancy fraction | 0.1970 ± 0.0051 | 0.201, 0.198, 0.188, 0.196, 0.202 |
| binary repeated-member fraction | 0.5491 ± 0.0076 | 0.557, 0.551, 0.537, 0.544, 0.557 |
| binary redundancy fraction | 0.4934 ± 0.0062 | 0.499, 0.494, 0.483, 0.491, 0.500 |
| final repeated-member fraction | 0.5633 ± 0.0068 | 0.570, 0.564, 0.552, 0.561, 0.570 |
| final redundancy fraction | 0.5128 ± 0.0061 | 0.518, 0.513, 0.502, 0.512, 0.519 |
| largest final class size | 1052.6 ± 35.1 | 1043, 1094, 1001, 1090, 1035 |
| largest final class share | 0.1914 ± 0.0064 | 0.190, 0.199, 0.182, 0.198, 0.188 |
| obstruction fraction | 0.4228 ± 0.0164 | 0.406, 0.436, 0.402, 0.444, 0.426 |
| poison unclustered fraction | 0.5624 ± 0.0165 | 0.584, 0.556, 0.580, 0.548, 0.544 |
| clean unclustered fraction | 0.3756 ± 0.0053 | 0.370, 0.374, 0.383, 0.380, 0.371 |
| exact-purity capture % | 1.80 ± 0.51 | 2.2, 2.2, 2.2, 1.0, 1.4 |
| raw first-merger share | 0.5080 ± 0.0050 | 0.509, 0.508, 0.500, 0.507, 0.516 |

The full intermediate-stage extraction was affordable (~90 s/seed) and was run on
**all five seeds**, exceeding the preregistered minimum of seed-42 localization
plus five-seed raw/binary/final confirmation.

---

## 7. Disposition

**Primary: (5) MIXED CAUSE.** No single stage accounts for a clear majority of
final collision mass. Two co-dominant contributors, stable to ±0.5 pp across
seeds:

1. **(1) Raw representation collapse — 50.80% ± 0.50.** These rows are already
   byte-identical in the 1500-byte input. No payload-only detector of any kind,
   topological or otherwise, can separate them. This is a property of the data,
   not of TDA.
2. **(3) Single-threshold collapse — 46.68% ± 0.46.** Raw-distinct rows made
   identical by the threshold-0.4 cut at byte 102. This transition also creates
   **all** of the new clean/poison mixed classes (29 of them at seed 42); the
   later merger creates none.

Secondary and minor:

3. **(4) Topological feature collapse — 2.52% ± 0.31**, entirely at the cubical
   persistence step, and it merges **zero** poison rows into mixed classes. The
   filtration images, the Scaler, the six persistence summaries, and the final
   concatenation merge **exactly nothing** on any seed. **The topological feature
   map is effectively exonerated as a source of clean/poison confusion.**

Refuted:

4. **(2) Padding/support artifact — refuted.** Zero all-zero payload rows on
   every seed; the support record is a bijection and merges nothing.

On **(6) collision not load-bearing**: partially true, and the honest answer is
that both mechanisms are real. Exact collision obstructs 42.28 ± 1.64% of
poison and does bind under this fit, but `-1` assignment accounts for a larger
56.24 ± 1.65%. Collisions are load-bearing without being the dominant mechanism.
Only 2.48 ± 0.90% of poison fails for the "mixed neighbourhood among distinct
vectors" reason.

### What this does and does not say about the source study

It says nothing about how Monkam, De Lucia and Bastian preprocessed their data,
and no such inference is drawn. It does sharpen the open question: if roughly
half of this reconstruction's collision mass is raw-duplicate payload rows, then
whether the source frame contained them is a first-order difference, and Q2's
`Red share = poison_rate × capture` identity constrains only the poisoning
*proportion*, not the frame composition. That is a question for the authors, not
a licence to adopt whichever filter improves capture.

---

## 8. Next single-variable hypothesis (not implemented)

**H-Q4: raw-duplicate frame composition is causal for capture.**

*Single variable:* deduplicate the 5000-row clean subsample on the exact
1500-byte payload **before** the attack is generated, holding the attack family,
poison rate, threshold, raster, feature map, and OPTICS parameters fixed. Compare
capture and the Q3-D decomposition against the standing frame.

*Why it is the right next test:* it is the only one-variable change that
directly attacks the larger of the two co-dominant contributors, and Q3 has
already measured exactly what it should move — the raw first-merger share
(50.80%) should go to ~0, the binarization share should stay, and the
obstruction should fall by roughly the raw-attributed poison mass.

*Pre-commitment required before running it:* deduplication changes the frame, so
capture on a deduplicated frame is **not** comparable to the 2.2000% gate and
must never be reported as a reproduction improvement. It is a mechanism test.
If capture rises, the finding is "the reconstruction's frame contains
irreducible duplicate mass," not "the detector works better." State that before
running, per §7 of `CLAUDE.md`.

*Explicitly not proposed:* threshold selection, block weighting, feature
selection, OPTICS tuning, dropping `height_0_1`, or combining Q2's geometry +
threshold + `min_samples` factors. Each would be its own preregistration.

### Drafted author question — NOT SENT

> In the Payload-Byte UNSW-NB15 preprocessing for the cubical-persistence
> pipeline, were packets filtered or deduplicated before feature extraction —
> specifically, (a) packets with no transport-layer payload, (b) rows whose
> 1500-byte payload vector is byte-identical to another row's, or (c) any
> sampling by traffic class? We ask because in our reconstruction roughly a
> fifth of the 5500-sample feature matrix collapses to a single 60-dimensional
> point, and about half of that collapse is already present in the raw
> 1500-byte payload rows themselves rather than being introduced by the
> topological pipeline.

Christian decides whether and when to send this.

---

## 9. Files

Created:

- `phase_q3_collisions.py` — canonical hashing, equivalence classes,
  earliest-merger attribution, transition reports, label-freedom guardrail
- `phase_q3_stage_pipeline.py` — additive instrumented extraction of all eight
  stages, with the bitwise-equality assertion against the production pipeline
- `run_phase_q3_collision_audit.py` — Q3-A…Q3-E driver
- `tools/test_phase_q3.py` — 39 tests (29 unit + 10 artifact-structure)
- `tools/phase_q3_summarize.py` — read-only validator and summarizer
- `results/phase_q3_collision_audit.json` — the artifact
- `docs/PHASE_Q3_COLLISION_REPORT.md` — this report

Modified: `.gitignore` (`!results/phase_q3_*.json`),
`docs/PHASE_Q2_RECONCILIATION_REPORT.md` (wording corrections only — no Q2
result was rerun or changed), `CLAUDE.md`, `README.md`.

## 10. Verification performed

```
python verify_env.py                            PASSED
python -m unittest tools.test_phase_q            6 tests OK
python -m unittest tools.test_phase_q2          32 tests OK
python -m unittest tools.test_phase_q3          39 tests OK
python tools/repro_check.py --expect 2.2000     PASS, 2.2000%, (5500, 60)
python tools/phase_q3_summarize.py              all structural checks passed
```

Structural assertions enforced in `tools/test_phase_q3.py` and re-checked
independently by `tools/phase_q3_summarize.py`: `n_clean == 5000`,
`n_poison == 500`, shape `(5500, 60)`, instrumented ≡ production bitwise, every
stage covers all 5500 rows, both collision statistics obey their definitions and
repeated-member ≥ redundancy, attribution mutually exclusive with shares summing
to 1 and zero monotonicity violations, failure categories exhaustive with an
empty residual, largest-class trace internally consistent with monotone
non-increasing upstream signature counts ending at 1, five-seed keys exactly
`[42, 123, 456, 789, 1024]`, population SD (`ddof=0`) throughout, no non-finite
numeric field, and seed-42 capture exactly 2.2000%.
