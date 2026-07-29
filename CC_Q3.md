# Claude Code handoff: Phase Q3 collision-provenance audit

## Objective

Investigate the strongest new lead from Phase Q2: approximately 51.8% of the
legacy TDA feature matrix consists of members of exact duplicate-vector groups,
including one block of 1,043 identical 60-dimensional vectors.

Trace those collisions backward through the full pipeline and determine the
earliest stage at which distinct samples become indistinguishable:

1. raw 1,500-byte payload;
2. supported payload and padding profile;
3. threshold-0.4 binary image;
4. each of the five filtration images;
5. each unscaled persistence diagram;
6. each scaled persistence diagram;
7. each 12-feature filtration block; and
8. the final 60-feature vector.

Then quantify how exact mixed clean/poison collision classes constrain strict
100%-purity poison capture under the current OPTICS fit.

This is a diagnostic phase, not a repair. Do not filter rows, deduplicate the
dataset, add metadata, change binarization, change a filtration, or tune
OPTICS. Do not commit, push, email authors, or edit the poster. Leave the
working tree ready for Christian to review and commit.

## Central question

Assign the observed final-vector collisions to one or more causal categories:

- **Raw-data collision:** the 1,500-byte payload rows were already identical.
- **Padding/support collision:** rows differ only in information not represented
  by the zero-padded payload array or its conservative supported prefix.
- **Binarization collision:** distinct raw payloads produce the same threshold
  mask.
- **Filtration collision:** distinct masks produce the same tuple of filtration
  images or persistence diagrams.
- **Persistence-summary collision:** distinct diagrams produce identical
  12-feature blocks or final 60-vectors.
- **No exact-collision explanation:** the rows are distinct at 60 features and
  the remaining failure is neighborhood geometry rather than equality.

Do not force one global culprit. The largest 1,043-row block may have a
different origin from smaller duplicate groups. Report both the largest block
and population-weighted attribution across all repeated final-vector groups.

## Where to work

- Actual writable WIRE clone:
  `~/beels_tda/tda_data_poisoning_prevention`
- UNSW Payload-Byte CSV:
  `~/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv`
- Existing Python environment: `venv312`

The WIRE tree contains completed, uncommitted Phase Q and Q2 work. Preserve it.
Begin with `git status --short`. Do not pull, reset, clean, or discard any
existing changes.

The user has stopped the two previous shell processes. Verify that no Q2 writer
remains before starting Q3:

```bash
pgrep -af 'run_phase_q2|phase_q2|tee' || true
```

An idle terminal is harmless. If an actual Q2 writer remains, report its exact
PID and command before taking any action.

## Results and guardrails entering Q3

Preserve these findings:

- Legacy regression gate: UNSW-NB15, R60, seed 42, threshold 0.4,
  `(5500, 60)`, OPTICS capture exactly `2.2000%`.
- Phase M8: relaxing cluster purity does not close the source gap.
- Phase Q: the fixed multithreshold stack reduces exact representation
  collisions but is falsified as a matched-clean-cost detector.
- Q2-A: omitting label `-1` and renormalizing can make displayed cluster shares
  sum to 100%; this does not establish whether the source fit contained noise.
- Q2-B: the literal `1 x 1500` geometry raises capture modestly but has
  structurally trivial H1 and cannot support the claimed nontrivial
  H1-dependent feature map.
- Q2-C: `min_samples=2` creates approximately 599 clusters with median size 2.
  This is genuine fragmentation/micro-clustering, not a source-like
  reconciliation.
- Q2's primary disposition is **not reconciled**.

Before adding Q3 results, make these wording corrections in the Q2 report if
the current text is stronger:

1. Use "can be reproduced by dropping `-1` and renormalizing," not "fully
   explained." The source's actual noise handling remains unknown.
2. The `6.03% = 0.10 x 60.3%` identity supports a 10% poisoning proportion and
   the paper's stated all-sample Red-share denominator. It does not prove that
   packet selection, class composition, or deduplication matched this project.
3. Say the printed `1 x 1500` geometry cannot support the **claimed nontrivial
   H1-dependent map**, not that it cannot compute any feature vector.

These are documentation corrections only. Do not rerun Q2 or change its
artifacts.

## Scientific rules

- Use the legacy single-threshold 0.4 pipeline for the primary audit.
- Use the exact R60/seed-42 input realization from the standing regression gate.
- Matched stage comparisons must use the same row order, combined-data hash,
  attack realization, and poison mask.
- Keep label `-1` separate and never count it as captured.
- Ground-truth labels may be used only for retrospective group composition and
  capture attribution, never to alter a representation or clustering fit.
- Exact equality means exact equality. Do not round floats before declaring an
  exact collision.
- A tolerance-based comparison may be reported separately, with the tolerance
  fixed before results, but it must never replace the exact result.
- Persistence diagrams are multisets. Canonicalize valid points by homology
  dimension, birth, and death before hashing; do not treat array point order or
  giotto-tda padding as topology.
- Normalize signed zero before hashing float arrays. Handle NaN and infinite
  diagram entries explicitly and deterministically.
- Do not use an `O(N^2)` pairwise collision matrix. Work with stable hashes and
  equivalence classes.
- Do not call all repeated rows "duplicates" without defining the statistic.
  Report both repeated-member fraction and redundancy fraction as defined
  below.
- Population SD is `ddof=0`.
- Preserve all historical artifacts and regression behavior.

## Required collision statistics

At every stage report:

- `n_rows`;
- `n_unique_classes`;
- **repeated-member fraction**:
  rows belonging to a class of size at least 2 divided by all rows;
- **redundancy fraction**:
  `(n_rows - n_unique_classes) / n_rows`;
- number of repeated classes;
- largest class size and share;
- repeated-class size quantiles;
- clean-only, poison-only, and mixed clean/poison class counts and member mass.

Never use the bare phrase "duplicate fraction" in the report. The Q2 value
near 51.8% must be mapped to one of these exact definitions before comparison.

For each downstream stage transition, report:

- how many downstream classes contain more than one upstream class;
- how many rows participate in those newly merged classes;
- how many clean/poison mixed classes are first created at that transition;
- the largest newly merged class;
- a deterministic example with row indices and hashes, without dumping packet
  payload contents into the report.

## 1. Preflight and frozen gates

Run:

```bash
cd ~/beels_tda/tda_data_poisoning_prevention
git status --short

source venv312/bin/activate
export TDA_DATA_DIR="$HOME/wire/DataSets/PayloadByte_UNSW"
export TDA_RESULTS_DIR="$PWD/results"
export LOKY_MAX_CPU_COUNT="$(nproc)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -f "$TDA_DATA_DIR/Payload_data_UNSW.csv"
python verify_env.py
python -m unittest tools.test_phase_q -v
python -m unittest tools.test_phase_q2 -v
python tools/repro_check.py --expect 2.2000

for f in results/phase_q2_*.json; do
  python -m json.tool "$f" >/dev/null
done
```

Also enumerate and read all Q2 files rather than relying only on the handoff:

```bash
rg --files | rg 'phase_q2|PHASE_Q2'
```

Required outcomes:

- Phase Q tests: 6 passing.
- Phase Q2 tests: 32 passing, unless Q2 added an explicitly documented test
  after its handoff.
- Regression gate: exact `2.2000%`, shape `(5500, 60)`.
- Every Q2 JSON artifact parses.

If a gate fails, stop and diagnose it. Do not change legacy behavior to make Q3
run.

## 2. Build an additive stage-inspection path

Read at minimum:

- `data_loader.py`
- `tda_pipeline.py`
- `phase_q_pipeline.py`
- `clustering.py`
- `tools/repro_check.py`
- the Q2 accounting and geometry code;
- `run_test_b_capture.py` and its exact-duplicate utilities;
- `phase_q_metrics.py`;
- the Q2 report and all Q2 JSON artifacts.

Implement the audit additively. Do not edit the production transformers merely
to expose intermediates. Prefer an instrumented pipeline that mirrors the
legacy steps and regression-tests its final output against
`extract_tda_features()` exactly.

Suggested files:

- `phase_q3_collisions.py` - canonical hashing, equivalence classes, transition
  attribution, and mixed-class metrics;
- `phase_q3_stage_pipeline.py` - matched extraction of binary masks,
  filtration images, diagrams, scaled diagrams, and feature blocks;
- `run_phase_q3_collision_audit.py` - seed-42 discovery and five-seed
  confirmation driver;
- `tools/test_phase_q3.py` - deterministic unit and regression tests;
- `tools/phase_q3_summarize.py` - read-only validator/summarizer;
- `docs/PHASE_Q3_COLLISION_REPORT.md`;
- `results/phase_q3_collision_audit.json`.

The names may be adjusted to fit existing Q2 conventions, but keep all durable
Q3 outputs under a clear `phase_q3_` prefix.

### Required stage representations

Capture stable signatures for:

1. **Raw padded payload:** exact `uint8[1500]` row.
2. **Supported-payload record:** conservative support end, supported prefix,
   nonzero count, and padding count. The support end is one past the last
   nonzero byte. State explicitly that legitimate trailing zero bytes cannot be
   recovered and that packet `total_len` is not a verified transport-payload
   length.
3. **Binary mask:** exact threshold-0.4 mask. Record fitted `Binarizer.max_value_`
   and the effective byte cut. Use packed bits only as a lossless storage/hash
   representation.
4. **Five filtration images:** two Height and three Radial outputs, each kept
   separate and also represented as a five-part tuple.
5. **Five unscaled cubical diagrams:** canonicalized by valid diagram points
   and homology dimension.
6. **Five scaled diagrams:** capture the fitted scaler state and canonicalized
   scaled diagrams.
7. **Five 12-feature blocks:** entropy plus five amplitudes over the observed
   homology dimensions.
8. **Final 60-vector:** exact concatenation in the same order as the legacy
   pipeline.

The stage-inspection final 60-vector must be bitwise equal to the legacy output
for the full seed-42 frame. If bitwise equality is impossible solely because of
documented execution-order floating arithmetic, require exact numerical array
equality first, then investigate; do not silently relax to `allclose`.

## 3. Unit tests before the full audit

Add deterministic tests covering at least:

- exact hashing distinguishes different raw rows;
- signed zero normalizes consistently;
- exact float differences are not erased by rounding;
- packed binary masks are lossless;
- persistence-diagram point ordering does not change a canonical signature;
- diagram padding is excluded without dropping valid points;
- homology dimension remains part of the diagram signature;
- repeated-member and redundancy fractions differ on a known fixture;
- transition attribution identifies the first merger stage correctly;
- clean-only, poison-only, and mixed class composition is exact;
- candidate code cannot use ground-truth labels to construct stage signatures;
- identical final coordinate classes receive one OPTICS label under the
  standing seed-42 fit, or any exception is explicitly reported rather than
  assumed;
- the instrumented final 60-vector equals the legacy 60-vector.

Run the Q3 tests before the full extraction. Do not weaken a failing equality
test to make the audit proceed.

## 4. Q3-A: reproduce and define the Q2 collision observation

On the exact R60/seed-42 regression frame:

1. Recompute the legacy feature matrix once.
2. Record its deterministic hash and compare it with any Q2 feature/input hash.
3. Compute both repeated-member and redundancy fractions.
4. Identify the largest exact final-vector class.
5. Verify whether its size is exactly 1,043 and its share approximately 0.190.
6. Verify whether the Q2 "51.8%" value meant repeated-member fraction,
   redundancy fraction, or a different statistic.

If the values do not reproduce, stop and determine whether Q2 used a different
frame, arm, row ordering, or definition. Do not continue using an assumed
1,043-row target.

Record the current OPTICS label distribution within every repeated final-vector
class. Explicitly test whether any exact coordinate class is split across
multiple cluster labels.

## 5. Q3-B: trace the largest final-vector class upstream

For the largest class, report at every stage:

- number of distinct upstream signatures represented;
- clean and poison members;
- source-row versus appended-poison members;
- class-label composition from the UNSW labels, used only descriptively;
- whether each poisoned member is identical to its own clean source at that
  stage;
- support-end, nonzero-count, and all-zero-payload distributions;
- whether rows are raw-identical, differ only after the conservative support
  end, or are genuinely different within support;
- which filtration/diagram/summary components first merge distinct inputs.

Do not print payload bytes, full row indices, or other large/sensitive records
to the console or committed JSON. Store aggregate counts and a small set of
stable hashes. If example indices are needed for reproducibility, limit them to
a few rows and confirm they are ordinary dataset row numbers rather than
identifiers.

Conclude with one of these statements for the 1,043-row block:

- already identical in raw payload space;
- first merged by threshold-0.4 binarization;
- first merged by named filtration(s);
- first merged in cubical persistence;
- first merged by named persistence summary block(s);
- not reproduced on the frozen frame.

## 6. Q3-C: population-wide collision attribution

Repeat the first-merger attribution for every repeated final-vector class, not
only the largest block.

Produce a population-weighted table:

| Earliest equality/merger stage | Repeated classes | Member rows | Clean rows | Poison rows | Mixed-class poison rows | Share of final repeated-member mass |
|---|---:|---:|---:|---:|---:|---:|

Rows must include:

- raw padded payload;
- supported-payload/padding representation;
- binarization;
- each filtration or combined filtration tuple;
- unscaled persistence diagrams;
- scaled persistence diagrams;
- 12-feature filtration summaries;
- final concatenation only.

If equality already exists upstream, do not attribute the same rows again to a
later deterministic stage. The attribution must be mutually exclusive by
earliest merger stage.

Also report per-filtration discriminating power:

- unique-class count for each 12-feature block;
- repeated-member fraction for each block;
- number of raw- or binary-distinct classes that each block merges;
- number of individual-block collisions resolved by concatenating all five
  blocks;
- whether one filtration dominates the final equality pattern.

This is an ablation diagnostic, not permission to select or reweight features.

## 7. Q3-D: decompose strict-purity capture failure

For the current OPTICS labels, place every poisoned row into exactly one
mutually exclusive category:

1. captured in a 100%-poison non-noise cluster;
2. label `-1`;
3. non-noise and in a final-vector class containing at least one clean row;
4. non-noise, not in an exact mixed vector class, but assigned to a mixed
   cluster;
5. any residual category, which must be explained.

Report counts and percentages. The categories must sum to all poisoned rows.

Compute a **coordinate-class obstruction diagnostic**:

```text
poison rows sharing an exact final vector with at least one clean row
--------------------------------------------------------------------
                    all poison rows
```

Do not call `1 - obstruction` an algorithm-independent capture ceiling unless
you prove that the relevant clustering rule cannot split identical
coordinates. For the standing OPTICS fit, verify class co-assignment
empirically and state the result at that scope.

Compare this obstruction with:

- the observed exact-purity capture;
- the poison unclustered fraction;
- the historical exact-100% duplicate/union ceilings.

The purpose is to separate failure caused by exact representation collision
from failure caused by noise assignment or mixed neighborhoods among distinct
vectors.

## 8. Q3-E: five-seed confirmation

After the complete seed-42 audit passes validation, confirm the inexpensive
final-vector and raw/binary collision statistics across seeds
`[42, 123, 456, 789, 1024]` using the same R60 construction as the standing
reconstruction.

At minimum summarize mean plus population SD for:

- raw repeated-member and redundancy fractions;
- binary repeated-member and redundancy fractions;
- final-vector repeated-member and redundancy fractions;
- largest final class size/share;
- mixed final-class poison obstruction;
- poison and clean `-1` fractions;
- exact-purity capture;
- the four-category failure decomposition.

Run the full intermediate-stage extraction on all five seeds only if it can
reuse the same fitted pipeline pass without a major compute increase. Otherwise
the seed-42 stage localization plus five-seed raw/binary/final confirmation is
the preregistered scope. Do not selectively run only favorable seeds.

Do not add the four Phase Q attack families in Q3. If the R60 mechanism is
confirmed, cross-family generalization is a separate preregistered phase.

## 9. Interpretation rules

Assign a primary disposition and any secondary contributors:

1. **Raw representation collapse:** Most final repeated-member mass is already
   identical in the 1,500-byte input. A payload-only TDA detector cannot
   distinguish labels attached to identical rows.
2. **Padding/support artifact:** Collision mass is concentrated in empty or
   heavily padded payloads, indicating a source-data/preprocessing question.
3. **Single-threshold collapse:** Raw-distinct rows first become identical at
   threshold 0.4, strengthening the binarization diagnosis.
4. **Topological feature collapse:** Distinct masks or diagrams merge in the
   filtration/persistence/summary map.
5. **Mixed cause:** No one stage accounts for a majority; report the exact
   population-weighted attribution.
6. **Collision not load-bearing:** Exact mixed-vector obstruction is too small
   to explain low capture; neighborhood density and `-1` assignment remain the
   dominant measured mechanisms.

Do not infer how Monkam et al. preprocessed duplicates unless source evidence
exists. If raw/padding collapse dominates, the next step is to ask whether the
source filtered empty payloads, deduplicated rows, sampled by traffic class, or
included metadata. It is not permission to adopt whichever filter improves
capture.

## 10. Success criteria

Q3 succeeds if it answers all of the following, even if collisions are not the
main cause:

1. What exactly did the Q2 51.8% statistic measure?
2. Does the 1,043-row class reproduce on the frozen frame?
3. Were those rows already identical at raw input?
4. If not, what is their earliest merger stage?
5. What share of all final collision mass is attributable to each stage?
6. How much poisoned mass is obstructed by exact clean/poison vector sharing?
7. How much failure remains attributable to `-1` and mixed neighborhoods among
   distinct vectors?
8. Is the mechanism stable across five seeds?

Success is localization, not improved capture.

## 11. Artifacts and Git visibility

Required durable artifacts:

- all Q3 source, runner, validator, and test files;
- `docs/PHASE_Q3_COLLISION_REPORT.md`;
- compact `results/phase_q3_*.json` artifacts containing hashes, parameters,
  versions, definitions, and aggregate results;
- updated `CLAUDE.md` and `README.md` only after the result is final.

Add the narrow Git rule:

```gitignore
!results/phase_q3_*.json
```

if required, and verify every durable artifact with `git check-ignore`.

Machine-local caches may use `.q3_cache/` and remain ignored. Do not commit raw
or intermediate payload arrays, feature matrices, persistence diagrams, source
PDFs, data files, logs, or virtual environments. The tracked code and compact
JSON must be sufficient to regenerate and audit the result.

Do not overwrite Phase Q, Q2, M, or P artifacts.

## 12. Verification

Run at minimum:

```bash
source venv312/bin/activate
python verify_env.py
python -m unittest tools.test_phase_q -v
python -m unittest tools.test_phase_q2 -v
python -m unittest tools.test_phase_q3 -v
python tools/repro_check.py --expect 2.2000

for f in results/phase_q2_*.json results/phase_q3_*.json; do
  python -m json.tool "$f" >/dev/null
done

for f in results/phase_q3_*.json; do
  if git check-ignore -q "$f"; then
    echo "ERROR: ignored required Q3 artifact: $f"
    exit 1
  fi
done

git diff --check
git status --short
```

Add structural validation assertions for:

- `n_clean == 5000`, `n_poison == 500`, feature shape `(5500, 60)`;
- exact input, poison-mask, and feature hashes;
- every stage contains all 5,500 rows;
- repeated-member and redundancy statistics obey their definitions;
- transition attribution is mutually exclusive and sums correctly;
- the largest-class trace is internally consistent across stages;
- the poisoned failure categories are mutually exclusive and exhaustive;
- five-seed keys are exactly `[42, 123, 456, 789, 1024]`;
- no required numeric field is NaN or infinite except explicitly permitted
  persistence/OPTICS diagnostic fields;
- legacy capture remains exactly `2.2000%` for the standing gate.

## 13. Final handoff to Christian

Report:

- the primary disposition from Section 9;
- the exact definition and reproduced value of the Q2 51.8% statistic;
- whether the 1,043-row block reproduced and its earliest equality stage;
- the population-wide earliest-merger table;
- raw, binary, diagram, block, and final unique-class statistics;
- the composition of the largest class without exposing payload contents;
- the strict-purity failure decomposition;
- five-seed stability;
- any Q2 wording corrections applied;
- all files created or modified;
- all verification commands and results;
- the next single-variable hypothesis justified by Q3, without implementing it.

If raw/padding collapse dominates, draft but do not send an additional author
question asking whether empty payloads, duplicate payload rows, or packets with
no transport payload were filtered before TDA. If a later TDA stage dominates,
name the exact filtration or summary responsible and propose a separately
preregistered one-variable repair.

Do not commit, push, email, or edit the poster. Christian will decide those
actions after reviewing the evidence.
