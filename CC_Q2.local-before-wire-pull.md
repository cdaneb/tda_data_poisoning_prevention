# Claude Code handoff: Phase Q2 source-coverage reconciliation

## Objective

Determine whether the large gap between this reconstruction and Monkam,
De Lucia, and Bastian can be reconciled by an identifiable difference in:

1. OPTICS noise accounting;
2. the TDA image geometry and feature map; or
3. OPTICS parameters or cluster extraction.

The motivating observation is that this reconstruction assigns substantial
mass to OPTICS label `-1`, while the source paper's reported cluster rows sum
to approximately 100%. Treat that as a hypothesis, not a fact: the paper may
have produced almost no noise, or it may simply have omitted/renormalized
noise. Q2 must distinguish those possibilities.

Do not optimize settings to obtain a 40--70% poison-capture number. The goal is
causal attribution and faithful reconstruction. Do not commit or push. Leave
the working tree ready for Christian to review and commit.

## Where to work

- Actual writable WIRE clone:
  `~/beels_tda/tda_data_poisoning_prevention`
- UNSW Payload-Byte data:
  `~/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv`
- Use the existing `venv312` created from `requirements.lock.txt`.
- The source PDF is intentionally excluded by `.gitignore` for copyright
  reasons. Do not force-add it. If a lawful local copy is unavailable on WIRE,
  use DOI `10.1016/j.cose.2024.103929`, the SSRN author manuscript, and the
  page-level facts recorded below. Record which version you inspected.

Do not assume the stale `~/projects/...` path in the original Q handoff.
Do not discard the completed, uncommitted Phase Q R1 work. Begin with
`git status --short`; do not pull across uncommitted changes.

## Results already established

Preserve these results and do not rerun or tune Phase Q R1:

- The standing legacy gate is OPTICS capture `2.2000%` with feature shape
  `(5500, 60)` for UNSW-NB15, R60, seed 42, threshold 0.4.
- Phase M8 showed that relaxing cluster purity does not close the 40--70%
  gap.
- Phase Q R1 showed that the fixed nine-threshold stack reduces exact
  clean-vector duplication by roughly 13--20 times but does not improve
  matched-clean-cost poison removal.
- The stack causes **density collapse/noise inflation**, not fragmentation:
  poison unclustered rises from about 0.85--0.87 to about 0.99, clean
  unclustered rises from about 0.37 to about 0.55, and cluster count falls.
- The multithreshold result localizes the remaining issue downstream of the
  single-cutoff identity mechanism. It does not establish that OPTICS alone is
  at fault.

Use the legacy single-threshold arm for Q2. The multithreshold stack is not a
source-paper reconstruction and must not be mixed into this study.

## Source facts that Q2 must verify and reconcile

Inspect the source paper directly when possible. The following facts were
transcribed from the journal PDF and provide a cross-check:

### TDA construction

- Page 6 says packet payloads are transformed into one-dimensional images of
  size `1 x 1500`.
- Algorithm 1 on page 6 specifies directions `[[0, 1], [1, 0]]`, centers
  `[[0, 1500], [0, 750], [1500, 0]]`, five filtrations, threshold `0.4`,
  `CubicalPersistence`, `Scaler`, `PersistenceEntropy`, and five `Amplitude`
  metrics.
- The captions to Figures 8 and 9 on page 8 say the `30 x 50` format is used
  **for illustration purposes**.
- Section 6.1 on page 9 instead says the binarization threshold is `0.3` and
  claims eight TDA metrics while naming only the five used by Algorithm 1.
- The paper repeatedly claims 72- and 126-feature representations. The printed
  Algorithm 1 arithmetic is five filtrations times six summaries times two
  homology dimensions, which yields 60 rather than 72.

This exposes a major source/reconstruction fork. The current repository treats
`30 x 50` as the operational raster and rescales the radial centers to
`[[0, 50], [0, 25], [30, 0]]`. Q2 must test the printed `1 x 1500` geometry
before searching OPTICS parameters.

### OPTICS result and configuration

- Figure 14 and Section 6.5 report the UNSW OPTICS result.
- The 126-feature poisoned TDA row in Figure 14 contains seven clusters whose
  sample shares sum to 100%. Its two Red clusters contain 56.1% of the poisoned
  population.
- The 72-feature poisoned TDA row contains eight displayed entries whose
  sample shares sum to 100%. Its three Red entries contain approximately 64.1%
  of the poisoned population; one cluster identifier is duplicated in the
  figure.
- Section 6.5 reports an approximately 60.3% poison-capture summary and color
  shares of 47.54%, 45.83%, 6.03%, and 0.59%. Do not assume those prose values
  are arithmetically identical to Figure 14. Recompute the table and document
  any inconsistency.
- Section 7, page 14, gives only broad OPTICS ranges: `min_samples` in
  `[2, 300]`, epsilon in `(0, 2]`, and `min_cluster_size` in `[2, 400]`.
  It does not report the exact fitted configuration, distance metric, Xi
  value, or sklearn extraction rule.
- The paper's OPTICS pseudocode creates clusters directly from epsilon
  neighborhoods. This is not enough to identify whether the implementation
  used sklearn's `cluster_method="xi"` or `cluster_method="dbscan"`.

The rows summing to 100% do **not** by themselves prove that the fitted model
had no noise. Noise may have been omitted and the remaining shares normalized.

## Scientific rules

- Preserve all historical Phase M, Phase P, and Phase Q artifacts.
- Make new code additive. Do not silently change `tda_pipeline.py`,
  `clustering.py`, `tools/repro_check.py`, or R1 drivers.
- Every experimental comparison must change one factor at a time on the same
  data realization and feature matrix where applicable.
- Never use poison labels to fit OPTICS or select a parameter candidate.
  Labels are retrospective evaluation only.
- OPTICS label `-1` is not a captured poison cluster. Report it separately.
- Report sample-weighted cluster shares. Do not substitute counts of clusters
  for percentages of samples.
- Report poison capture together with clean false-removal, removal precision,
  clean/poison unclustered fractions, cluster count, and largest-cluster share.
- Use population SD (`ddof=0`) for five-seed summaries.
- Do not delete seeds, resample attacks, select a favorable attack family, or
  change the evaluation metric after seeing results.
- Do not call lower cluster count plus higher `-1` mass "fragmentation." Use
  "density collapse" or "noise inflation."
- Do not claim full reproduction while the source attack, sampling frame,
  72/126-feature maps, or exact OPTICS configuration remains unrecovered.
- Do not edit the poster during Q2. Update research documentation only after
  the analysis is complete and reviewed.

## 1. Preflight

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
python tools/repro_check.py --expect 2.2000
python -m json.tool results/phase_q_r1_multithreshold_capture.json >/dev/null
```

Required outcomes:

- Environment verification passes.
- Phase Q tests pass.
- The standing gate remains exactly `2.2000%`, shape `(5500, 60)`.
- The completed R1 artifact is valid.

If a gate fails, stop and diagnose it. Do not alter the environment or legacy
pipeline merely to continue Q2.

## 2. Perform a source-to-code audit before writing experiment code

Read at minimum:

- `CLAUDE.md`
- `README.md`
- `tda_pipeline.py`
- `clustering.py`
- `data_loader.py`
- `tools/repro_check.py`
- `run_lens4_baseline.py`
- `run_test_b_capture.py`
- `docs/PHASE_M_A19_A23_REPORT.md`
- `docs/PHASE_Q_MULTITHRESHOLD_REPORT.md`
- `results/lens4_baseline_multiseed.json`
- `results/phase_m_m8_purity_sweep.json`
- `results/phase_q_r1_multithreshold_capture.json`

Create a source-to-code matrix in
`docs/PHASE_Q2_RECONCILIATION_REPORT.md` with these rows:

| Component | Source evidence | Current code | Status | Test needed |
|---|---|---|---|---|
| Input geometry | `1 x 1500`; `30 x 50` illustrative | operational `30 x 50` | mismatch | source-geometry arm |
| Directions/centers | Algorithm 1 values | rescaled centers | mismatch | source-geometry arm |
| Binarizer | 0.4 algorithm / 0.3 prose | 0.4 | ambiguous | separate threshold arm |
| Feature count | claims 72 and 126 | observed 60 | unresolved | arithmetic and runtime audit |
| Feature reduction | text says 126 reduced to 72 | no reduction stage | unresolved | locate source method/code |
| OPTICS parameters | broad ranges only | `min_samples=5`, `max_eps=2.0`, sklearn defaults otherwise | unresolved | provenance-first sensitivity study |
| Noise handling | no displayed noise row | `-1` excluded from capture and reported | unresolved | accounting audit |
| Sample frame | establish from paper/code if possible | 5000 clean + 500 appended poison | likely divergence | provenance audit |
| Poison generator | Chale/Hore attacks | reconstruction attacks | known divergence | do not conflate with clustering audit |

Also search, without guessing, for:

- official supplementary material or author code;
- the 2023 precursor paper, DOI `10.1109/DSC61021.2023.10354143`,
  which the journal article cites for the 126-feature pipeline;
- dependency versions or notebooks that reveal the exact image shape,
  feature-reduction step, or OPTICS call.

Record URLs, access dates, and exact quotations of no more than what is needed.
If no code or exact configuration is recoverable, state that plainly. Do not
reverse-engineer a favorable configuration and present it as the authors' code.

## 3. Q2-A: noise-accounting audit

This stage is read-only with respect to the fitted baseline. Recreate the
standing R60/seed-42 feature matrix and fit the current OPTICS configuration
once. Preserve the exact same combined data realization used by the regression
gate.

Add an isolated diagnostic utility rather than changing `clustering.py`. It
must save:

- `OPTICS().get_params(deep=True)` and package versions;
- `labels_`, `ordering_`, `reachability_`, `core_distances_`, and
  `predecessor_` summaries or hashes sufficient to validate provenance;
- total, clean, and poisoned counts assigned to `-1`;
- number and size of non-noise clusters;
- largest-cluster share;
- finite/infinite reachability and core-distance counts and quantiles;
- sample-weighted Green/Yellow/Pink/Red/Noise shares using the project rules.

From the same labels, compute three accounting views without refitting:

1. **All-sample denominator:** include Noise as its own category.
2. **Clustered-only denominator:** omit `-1` and renormalize the remaining
   sample shares to 100%.
3. **Noise-as-Yellow display:** include `-1` samples in a mixed/unknown display
   category, but never count them as captured.

Compare each view with Figure 14. Answer:

- Can the apparent 100% source coverage be explained solely by denominator
  choice?
- Does any accounting choice change true poison capture when its denominator
  remains all poisoned samples?
- Are the source prose summary and Figure 14 internally consistent?

If a display convention makes percentages look similar but leaves the actual
capture unchanged, label the result **accounting reconciliation only**, not
detector reproduction.

Suggested additive artifacts:

- `tools/phase_q2_accounting_audit.py`
- `tools/test_phase_q2.py`
- `results/phase_q2_accounting_audit.json`

Do not proceed to parameter experiments until this report exists.

## 4. Q2-B: test the source-printed image geometry

This is the highest-priority controlled method fork because it is an explicit
source/code mismatch.

Implement an additive source-geometry pipeline, leaving the legacy pipeline
untouched:

- Input image shape: `(N, 1, 1500)`.
- Directions: `[[0, 1], [1, 0]]`.
- Centers: `[[0, 1500], [0, 750], [1500, 0]]` exactly as printed.
- Threshold: `0.4` for the first comparison.
- Same five filtrations, `CubicalPersistence`, `Scaler`, entropy, five
  amplitude metrics, homology behavior, and dependency versions.
- Same exact R60/seed-42 combined data matrix and poison mask in both arms.
- Same current OPTICS fit for the first comparison.

The only conceptual factor in this arm is **source-printed geometry**. Shape
and centers move together because they define the same coordinate system. Do
not combine this comparison with threshold 0.3 or an OPTICS change.

Before the full run, test a tiny deterministic fixture and record:

- whether giotto-tda accepts the printed shape and centers;
- actual output feature count;
- homology dimensions present;
- NaN/Inf counts;
- zero-variance feature count.

Do not force the result to 60 or 72 columns. The observed output dimension is
part of the diagnosis. If the literal printed configuration fails in the
pinned library, preserve the exact exception and stop that arm. Do not silently
clip or rescale a center.

For both the legacy and source-geometry matrices, record:

- feature shape and a deterministic hash;
- per-feature variance and zero-variance counts;
- exact duplicate-row frequency;
- Euclidean clean-clean, clean-poison, and paired clean-poison distance
  quantiles on a fixed, seeded diagnostic sample;
- nearest-neighbor and fifth-neighbor distance quantiles;
- fraction of points whose core distance is finite and at most `max_eps`;
- OPTICS cluster count, largest-cluster share, and clean/poison `-1` rates;
- the frozen removal metrics already used by Phase Q.

If the source-geometry arm materially lowers noise or changes cluster coverage,
lock that result before testing threshold 0.3. Then run a separate one-factor
threshold comparison, 0.4 versus 0.3, on the same source geometry and data.

Suggested additive artifacts:

- `phase_q2_source_pipeline.py`
- `run_phase_q2_geometry.py`
- `results/phase_q2_geometry.json`

## 5. Q2-C: provenance-first OPTICS sensitivity analysis

Only begin this stage after Q2-A and Q2-B are complete. Reuse cached feature
matrices; do not recompute TDA features for every clustering setting.

The publication does not provide a unique OPTICS configuration. Therefore this
stage is a **sensitivity analysis**, not a source reproduction, unless exact
author code or metadata is recovered.

### Discovery protocol

- Use only the fixed R60/seed-42 discovery matrix.
- Do not expose poison labels to the candidate-selection function.
- Hold the feature map fixed within each sweep.
- Start from the current configuration:
  `min_samples=5`, `max_eps=2.0`, `cluster_method="xi"`, with every other
  parameter at the pinned sklearn default.
- Record the full estimator parameter dictionary for every cell.

Run one-factor-at-a-time, source-anchored sweeps:

1. `max_eps`: `{0.25, 0.5, 1.0, 2.0}` with everything else fixed. A separate
   `inf` cell may be included only as the sklearn-default interpretation and
   must be labeled outside the paper's stated epsilon range.
2. `min_samples`: `{2, 5, 10, 25, 50, 100, 300}` with everything else fixed.
3. `min_cluster_size`: `{2, 5, 10, 25, 50, 100, 300, 400}` with everything
   else fixed. Also retain the existing `None` baseline.

Do not add Xi tuning, alternate metrics, PCA, standardization, or arbitrary
grids in this stage. If direct evidence shows that the authors used sklearn's
DBSCAN extraction, add one separately labeled extraction-method arm; otherwise
record Xi-versus-DBSCAN as unresolved rather than fishing across `eps` values.

For discovery, rank configurations only by label-free quantities:

- unclustered sample fraction;
- number of clusters;
- largest-cluster share;
- cluster-size distribution;
- reachability/core-distance coverage.

Do not rank by Red capture, cluster purity, or any ground-truth label statistic.
Lock at most three candidates, and state the label-free rule used to select
them, before computing poison metrics.

### Confirmation protocol

Evaluate locked candidates unchanged across seeds
`[42, 123, 456, 789, 1024]`. Use the same attack definition and sampling
protocol as the standing R60 reconstruction. Report mean plus population SD.

Only after the candidates are locked, compute:

- exact-purity and relaxed-purity poison removal;
- clean false-removal;
- removal precision;
- clean and poison `-1` fractions;
- sample-weighted color shares;
- cluster count and largest-cluster share.

A configuration that reaches 40--70% capture only on the discovery seed, only
after label-aware selection, or by treating `-1` as captured is not evidence of
reconciliation.

Suggested additive artifacts:

- `run_phase_q2_optics_sensitivity.py`
- `results/phase_q2_optics_sensitivity.json`

## 6. Interpretation categories

Assign exactly one primary disposition and explain the evidence:

1. **Accounting reconciliation only:** Source-like percentages arise by
   omitting or renormalizing noise, but true capture and coverage remain low.
2. **Geometry reconciliation:** The literal source geometry reproducibly
   changes density coverage toward Figure 14 under fixed OPTICS.
3. **Parameter-sensitive configuration:** A publication-supported OPTICS
   setting changes coverage, but missing exact author parameters prevent a
   reproduction claim.
4. **Controlled internal reconciliation:** A locked, source-supported choice
   yields source-like coverage and 40--70% capture across confirmation seeds at
   acceptable clean cost. This is still not full Monkam reproduction unless
   feature count, source attack, and sampling frame also match.
5. **Not reconciled:** None of the source-supported tests closes the gap.

The result may combine a secondary accounting finding with one primary
disposition, but do not blur them.

## 7. Success criteria

The task succeeds even if the gap remains unresolved. Success means:

- every source/code mismatch is documented with page or code evidence;
- the Figure 14 arithmetic and noise denominator are audited;
- the `1 x 1500` source geometry is tested literally against the current
  `30 x 50` geometry on matched data;
- OPTICS internals explain why points become `-1` under the reconstruction;
- any sensitivity experiment is label-free during selection and confirmed on
  held-out seeds;
- the conclusion distinguishes source reproduction, controlled internal
  reconciliation, parameter sensitivity, and accounting artifacts.

Do not define success as finding any setting that produces a desired number.

## 8. Files and Git visibility

Required durable deliverables should include:

- `CC_Q2.md`
- `docs/PHASE_Q2_RECONCILIATION_REPORT.md`
- all new Q2 source, runner, summarizer, and test files;
- the compact JSON results necessary to verify every reported number.

If results are named `results/phase_q2_*.json`, add the narrow rule
`!results/phase_q2_*.json` to `.gitignore` and verify each artifact with
`git check-ignore`. Do not unignore the whole results directory.

Do not commit:

- copyrighted source PDFs;
- UNSW/CICIDS data;
- virtual environments;
- large feature-matrix caches;
- temporary renders or logs unless Christian explicitly requests them.

Large `.npy`/`.npz` feature caches may remain machine-local and regenerable.
The tracked JSON must include data hashes, feature hashes, parameters, seeds,
versions, counts, and timings so the cache is not required to audit the result.

## 9. Verification

At the end, run at minimum:

```bash
source venv312/bin/activate
python verify_env.py
python -m unittest tools.test_phase_q -v
python -m unittest tools.test_phase_q2 -v
python tools/repro_check.py --expect 2.2000
python -m json.tool results/phase_q2_accounting_audit.json >/dev/null
python -m json.tool results/phase_q2_geometry.json >/dev/null

if test -f results/phase_q2_optics_sensitivity.json; then
  python -m json.tool results/phase_q2_optics_sensitivity.json >/dev/null
fi

for f in results/phase_q2_*.json; do
  if git check-ignore -q "$f"; then
    echo "ERROR: ignored required artifact: $f"
    exit 1
  fi
done

git diff --check
git status --short
```

Add structural validators that assert:

- exact dataset, sample counts, seeds, and poison masks are recorded;
- matched arms share the same input and attack hashes;
- no required numeric field is NaN or infinite unless it explicitly summarizes
  OPTICS infinite reachability/core distances;
- discovery candidate selection never receives the poison mask;
- confirmation uses only candidates locked by the label-free discovery rule;
- legacy `(5500, 60)` and `2.2000%` behavior remains unchanged.

## 10. Final handoff to Christian

Report:

- the primary disposition from Section 6;
- whether Figure 14 sums to 100 because of observed coverage or accounting;
- the result of `1 x 1500` versus `30 x 50` under fixed OPTICS;
- the actual feature dimensions and homology dimensions in each geometry;
- which OPTICS settings were source-recovered, source-supported, or merely
  exploratory;
- discovery and confirmation results separately;
- poison removal beside clean cost and `-1` fractions;
- all files created or modified;
- every verification command and its outcome;
- remaining information that only the source authors can supply.

If exact author information remains unavailable, draft but do not send a short
request asking for:

1. the operational image shape and center coordinates;
2. the code or definition producing 72 and 126 features;
3. the exact OPTICS constructor and cluster-extraction parameters;
4. how label `-1` was handled in Figure 14 percentages;
5. the sample size, sampling seed, poison construction, and dependency
   versions.

Do not commit, push, email authors, or edit the poster. Christian will review
the evidence and decide those actions.
