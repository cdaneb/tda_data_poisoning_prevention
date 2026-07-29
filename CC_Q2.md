# CC_Q2 — Phase Q2 handoff to Christian

Completed 2026-07-29 on WIRE in `venv312`. **Nothing was committed, pushed,
emailed, or applied to the poster.** The working tree is left for your review.

Full evidence: `docs/PHASE_Q2_RECONCILIATION_REPORT.md`.

---

## 1. Bottom line

**Primary disposition: (5) Not reconciled.** No source-supported test closes the
40–70% gap.

Three secondary findings are established and replicated:

1. **Accounting reconciliation confirmed — and it settles less than we hoped.**
   Figure 14's rows summing to ~100% is fully explained by dropping label `-1`
   and renormalizing. Our own fit, which puts **38.9%** of samples in `-1`, sums
   to exactly 100.00% under the same convention. So the summation carries **no
   information** about whether the authors' model produced noise. The motivating
   observation cannot decide that question in either direction. True capture is
   **2.2000% in all three display conventions** — unchanged, enforced in code,
   asserted in tests.

2. **The source-printed `1 x 1500` geometry is a real but compromised effect.**
   +3.32 ± 1.29 pp capture, **5/5 seeds positive, sign-stable**, at zero clean
   false-removal, and it moves unclustered mass 0.393 → 0.339 (toward Figure 14).
   But a one-pixel-tall image has no 1-cycles, so **H1 is identically empty** and
   **30 of 60 features are exactly zero-variance**. The printed geometry cannot
   compute the feature map Algorithm 1 asks for. Read it as evidence the
   operational raster was 2-D, not as evidence it was `1 x 1500`.

3. **`min_samples=2` reaches 11.68 ± 1.75% label-free, and is still not enough.**
   Inside the paper's stated `min_samples` range [2, 300]. Selected without any
   label entering the rule, confirmed on all five seeds, zero clean
   false-removal, precision 1.00. But it produces **599 clusters with median size
   2** — a two-poison-packet pair is a 100%-pure Red cluster by definition.
   Figure 14 displays 7–8 clusters.

## 2. Two things worth your attention that I did not go looking for

**§6.5's arithmetic recovers the source's poison rate exactly.** If Red clusters
are 100% poisoned, then `Red share of all samples = poison_rate x capture`. The
paper's printed shares are `47.54, 45.83, 6.03, 0.59` and its capture summary is
60.3%. At a 10% poison rate that identity predicts **6.03%** — an exact hit, no
other tested rate comes close. So: the source's poison rate is 10% (matching
ours), the shares use an all-sample denominator, and the mapping is Green 47.54 /
Yellow 45.83 / **Red 6.03** / Pink 0.59. **Our sampling frame is not the source
of the gap.** Note §6.5's 60.3% matches neither Figure 14 row (56.1% and 64.1%);
treat prose and figure as two separate reported quantities.

**51.8% of the feature matrix is exact duplicate rows**, with one degenerate
block of **1043 identical 60-vectors**. This is why `largest_cluster_share` is
pinned at exactly 0.190 across all 21 swept OPTICS configurations — about a fifth
of the dataset is a single point in feature space and no density method can split
it. This is upstream of every clustering result in the project and is the most
promising unexplored lead Q2 turned up.

## 3. Numbers (5 seeds `[42,123,456,789,1024]`, population SD, zero clean cost throughout)

| Arm | Capture % | Unclustered | Clusters |
|---|---:|---:|---:|
| Legacy `30x50` t=0.4 (standing) | 1.80 ± 0.51 | 0.393 | 137 |
| Source `1x1500` t=0.4 | 5.12 ± 1.53 | 0.339 | 174 |
| Source `1x1500` t=0.3 | 11.12 ± 1.55 | 0.352 | 189 |
| Legacy, `min_samples=2` | 11.68 ± 1.75 | 0.257 | 599 |
| Legacy, `max_eps=inf` (outside range) | 1.84 ± 1.12 | 0.341 | 172 |
| Legacy, `min_cluster_size=100` | 0.00 ± 0.00 | 0.402 | 15 |

The legacy arm reproduced per-seed `[2.2, 2.2, 2.2, 1.0, 1.4]` = 1.80 ± 0.51,
**bit-identical to CLAUDE.md §6's transpositions row** — the internal control
confirming every arm ran on the recorded realization.

## 4. Deliberate non-actions

- **Did not run source-geometry + t=0.3 + `min_samples=2` combined.** Each factor
  was tested alone; stacking three after seeing their results is the search the
  phase was designed to avoid. Preregister it as its own phase if you want it.
- **Did not run a `cluster_method="dbscan"` arm.** No evidence favors it over
  `xi`; running one meant fishing across `eps`. Recorded as unresolved.
- Did not tune Xi, add PCA/standardization, drop seeds, resample attacks, or
  change a metric after seeing results.
- Did not touch `tda_pipeline.py`, `clustering.py`, `phase_q_metrics.py`,
  `tools/repro_check.py`, or any R1 driver. Did not disturb the uncommitted
  Phase Q R1 work.

## 5. Source availability — the hard limit

**No lawful local copy of the source PDF exists on WIRE** (full search of `~`
and `~/wire`). ScienceDirect, the ACM DL mirror, SSRN and IEEE Xplore all
returned 403 or paywalls; USMA Athena served no full text; **no author code,
notebook, or supplementary material exists anywhere I could find.** The
page-level facts in the report are used as *transcribed by you*, not as text I
re-verified against primary source. The image shape, the 72/126-feature
definition, the fitted OPTICS constructor, and the `-1` handling in Figure 14 are
not recoverable from public sources.

**A five-question request to Monkam/Bastian is drafted in §8 of the report. It
was not sent.** Your call.

## 6. Files

Created:
- `CC_Q2.md`, `docs/PHASE_Q2_RECONCILIATION_REPORT.md`
- `phase_q2_source_pipeline.py` — geometry-parameterized feature map (additive)
- `tools/phase_q2_common.py` — realization builder, hashing, OPTICS internals, accounting views
- `tools/phase_q2_accounting_audit.py` — Q2-A
- `run_phase_q2_geometry.py` — Q2-B
- `run_phase_q2_optics_sensitivity.py` — Q2-C
- `tools/test_phase_q2.py` — 28 structural validators
- `results/phase_q2_accounting_audit.json`, `results/phase_q2_geometry.json`,
  `results/phase_q2_optics_sensitivity.json`

Modified: `.gitignore` only — added `!results/phase_q2_*.json` (the existing
`phase_q_*` rule needs a literal underscore after `q`, so it does not match
`phase_q2_`) and `.q2_cache/` for the machine-local feature cache.

Every tracked JSON carries data hash, poison-mask hash, feature hash, full
estimator parameters, seeds, counts, versions and timings, so `.q2_cache/` is
never required to audit a number.

## 7. Verification (all run at completion — see the report §9 for commands)

| Command | Result |
|---|---|
| `python verify_env.py` | PASSED |
| `python -m unittest tools.test_phase_q -v` | 6 tests OK |
| `python -m unittest tools.test_phase_q2 -v` | 28 tests OK |
| `python tools/repro_check.py --expect 2.2000` | **PASS — 2.2000%, `(5500, 60)`** |
| `python -m json.tool` on all three Q2 artifacts | valid |
| `git check-ignore` on all `results/phase_q2_*.json` | none ignored |
| `git diff --check` | clean |
