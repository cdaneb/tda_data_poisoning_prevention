# Phase Q: controlled multithreshold repair

## Question

Can a fixed multithreshold binarization stack remove the single-cutoff
bit-identity blind spot while holding the rest of the reconstructed Monkam
pipeline fixed?

The target is deliberately narrower than universal permutation detection.
No payload-only method can detect a permutation whose action leaves the byte
vector unchanged, and persistence is not injective.  The repair claim is:

> A threshold stack can remove artificial same-bin identity, leaving empirical
> diagram displacement and population-level detection to be measured.

## Controlled design

The only representation change in R1 is:

- Control: threshold `0.4`, 60 features.
- Repair: thresholds `{0.1, 0.2, ..., 0.9}`, nine unchanged 60-feature blocks
  concatenated in threshold order and divided by `sqrt(9)`, 540 features.

Held fixed: the 30x50 raster, two Height and three Radial filtrations,
`CubicalPersistence`, `Scaler`, persistence entropy, five amplitude summaries,
and OPTICS with `min_samples=5, max_eps=2.0`.

The fixed `1/sqrt(9)` dimension compensation ensures that nine identical blocks
would preserve the control's Euclidean distance scale under OPTICS
`max_eps=2`. It gives every threshold equal, non-learned weight. The 0.4 control
is taken directly from the stack extraction, and a regression test requires
exact equality with the legacy pipeline output. There is no feature whitening,
learned weighting, threshold tuning, or clusterer tuning.

## D0: valid attack substrate

The upstream Payload-Byte code defines `total_len` as the IPv4 packet length
but extracts `bytes(f[UDP/TCP/ICMP].payload)` into the payload columns.  Thus
`total_len` is not a payload-length boundary.  Payload arrays are resized to
1500, with shorter arrays zero-filled.

Phase Q defines conservative support as one plus the last nonzero payload-byte
index.  This can omit legitimate trailing zero bytes, but cannot include
appended padding.  All families use the same malicious targets, require at
least 120 conservative-support bytes, retain the original disturbance budgets,
and rejection-sample parameters until the raw byte vector changes.

On the 200-row seed-42 diagnostic frame:

| Family | Legacy raw no-op | Phase Q raw no-op | Phase Q median positions changed |
|---|---:|---:|---:|
| Transpositions | 12.5% | 0.0% | 116 |
| Block reversal | 84.5% | 0.0% | 117 |
| Block swap | 78.5% | 0.0% | 118 |
| Cyclic shift | 0.0% | 0.0% | 311 |

The historical Test B table remains unchanged.  Phase Q is a new baseline, not
a retroactive rewrite.

## D1-D4: where the signal is lost

The full 200-pair, four-family diagnostic used one combined clean+perturbed fit
for every comparison.  At threshold 0.4, only 2.0-4.5% of binary images were
identical.  Exact 60-vector identity was 3.0-5.5%, so most genuinely altered
attacks already survive binarization, diagrams, and vectorization.

The problem is magnitude.  Median attack displacement in the 60-vector was
only 7.08-11.61, compared with about 66.15 for cyclic clean-clean pairings.
Across every family, 0/200 attack displacements exceeded the clean-clean 95th
percentile.

The nine-threshold stack reduced both whole-stack binary identity and exact
540-vector identity to 0/200 for every family.  This confirms the algebraic
repair: the tested attacks are no longer trivially identified with their source
by the representation.  It did **not** improve relative separation:

| Family | Control attack/clean median ratio | Stack ratio |
|---|---:|---:|
| Transpositions | 0.176 | 0.122 |
| Block reversal | 0.107 | 0.070 |
| Block swap | 0.125 | 0.087 |
| Cyclic shift | 0.139 | 0.107 |

Again, 0/200 stack displacements exceeded the clean-clean 95th percentile for
every family.  The stack increases absolute attack displacement, but increases
ordinary population variation more.  Therefore the simple unweighted
concatenation hypothesis is diagnostically unsupported; it must not be called a
detection improvement unless the frozen downstream comparison proves otherwise.

## Frozen downstream metrics

`run_phase_q_experiment.py` reports the original exact-purity capture together
with poison removal, clean false-removal, precision among removed samples,
poison/clean unclustered fractions, and matched-clean-cost comparisons at
purity thresholds `{1.0, >0.95, >0.90, >0.80, >0.50}`.

Cluster purity uses ground truth retrospectively.  It is an evaluation
instrument, not a deployable rule for labeling clusters.

## R1 result: complete five-seed grid (status: COMPLETE)

The full 20-cell grid (four families x seeds `42, 123, 456, 789, 1024`) ran on
WIRE against the full UNSW-NB15 CSV in 102 minutes.  Every cell has
`n_clean=5000`, `n_poison=500`, `raw_noop_count=0`, a 60-feature control block,
a 540-feature repair stack, five frozen removal-curve points per arm, five
matched-clean-cost records, and a provenance block; no interpretation field is
NaN or infinite.  Artifact:
`results/phase_q_r1_multithreshold_capture.json`.  All values below are
five-seed means +/- population SD (`ddof=0`).

### Primary metric: matched-clean-cost poison-removal delta (repair minus control)

At every clean-cost budget where the control removes no clean data (exact 1.0
through `>0.80`), the repair delivers no material gain:

| Family | delta @1.0 | delta @>0.95 | delta @>0.90 | delta @>0.80 | delta @>0.50 |
|---|---:|---:|---:|---:|---:|
| Transpositions | +0.0000 | +0.0000 | +0.0000 | +0.0000 | **-0.0188** |
| Block reversal | +0.0064 | +0.0064 | +0.0064 | +0.0044 | **-0.0060** |
| Block swap | +0.0000 | +0.0000 | +0.0000 | +0.0000 | **-0.0152** |
| Cyclic shift | -0.0020 | -0.0020 | -0.0020 | -0.0020 | **-0.0156** |

The single nonzero positive is block reversal, `+0.0064 +/- 0.0053` at the
strict budgets: on 3/5 seeds the stack removes one 100%-poison cluster of
~5 packets (~1% of poison) that the control does not; on 2/5 seeds it removes
none.  This is at the floor of measurement, not a robust detector gain.  At the
loosest budget `>0.50` — where the control begins to catch a small amount of
poison at nonzero clean cost — every family goes **negative**: the repair's best
feasible point removes *less* poison than the control.

### Supporting metrics (five-seed mean +/- population SD)

| Family | exact-purity removal (ctl / rep) | clean unclustered (ctl / rep) | poison unclustered (ctl / rep) | n_clusters (ctl / rep) | exact-dup-with-clean (ctl / rep) |
|---|---|---|---|---|---|
| Transpositions | 0.000 / 0.000 | 0.371+/-0.006 / 0.545+/-0.009 | 0.852+/-0.022 / 0.991+/-0.005 | 141.8+/-4.5 / 108.2+/-6.5 | 0.035+/-0.004 / 0.003+/-0.002 |
| Block reversal | 0.000 / 0.006+/-0.005 | 0.372+/-0.005 / 0.545+/-0.009 | 0.866+/-0.024 / 0.987+/-0.007 | 140.8+/-5.8 / 108.8+/-6.6 | 0.043+/-0.005 / 0.002+/-0.002 |
| Block swap | 0.000 / 0.000 | 0.371+/-0.005 / 0.545+/-0.009 | 0.869+/-0.019 / 0.993+/-0.005 | 142.4+/-4.7 / 108.0+/-6.4 | 0.039+/-0.004 / 0.002+/-0.002 |
| Cyclic shift | 0.002+/-0.004 / 0.000 | 0.372+/-0.006 / 0.546+/-0.008 | 0.875+/-0.010 / 0.992+/-0.004 | 139.6+/-5.7 / 107.6+/-6.0 | 0.026+/-0.003 / 0.002+/-0.002 |

Clean false-removal is 0.0000 for the repair at every purity point and every
family; the control only reaches nonzero clean false-removal (~0.001) at `>0.50`.

### Reading, along the three-level distinction

1. **Algebraic repair — CONFIRMED.** The stack drives exact-duplicate-with-clean
   frequency down ~13-20x (0.026-0.043 to 0.002-0.003) across all families and
   all five seeds.  The single-cutoff same-bin identity mechanism is closed on
   the tested frame, consistent with the D1-D4 whole-stack-identity result.

2. **Statistical separation — NOT ESTABLISHED.** Consistent with D1-D4, closing
   identity does not translate into usable separation.  The stack instead pushes
   nearly everything to the unclustered bin: poison unclustered rises from
   ~0.85-0.87 to ~0.99, clean unclustered from ~0.37 to ~0.55, and the cluster
   count falls from ~140 to ~108.  The block-reversal `+0.0064` sits inside this
   fragmentation regime and is not usable separation.

3. **Downstream detection (matched-clean-cost OPTICS removal) — FALSIFIED.**
   Three of four families show exactly zero matched-cost delta at operationally
   clean budgets; block reversal shows a negligible `+0.6%` on 3/5 seeds; and all
   four go negative at `>0.50`.  The result is consistent across all five seeds
   (small population SD on every fragmentation metric) — it is not driven by one
   seed — and no family shows a real improvement while another worsens.

**Conclusion.** The simple equally-weighted, unweighted threshold-stack repair
is falsified as a detector: it closes the exact single-cutoff bit-identity
mechanism but does not convert that into matched-clean-cost poison removal, and
the small "exactness" gains are purchased through heavier fragmentation (higher
unclustered fractions, fewer clusters), not through cluster separation.  This is
the D1-D4 "not trivially blind but not more discriminating" outcome, now
confirmed at the R1 capture level rather than the displacement level.  Threshold
selection, block weighting, feature selection, OPTICS tuning, attack resampling,
and seed deletion were **not** performed; any of them would be a new experiment
requiring new preregistration.

## Reproduction

Windows quick checks:

```powershell
.\venv312\Scripts\python.exe -m unittest tools.test_phase_q -v
.\venv312\Scripts\python.exe run_phase_q_support_audit.py
.\venv312\Scripts\python.exe run_phase_q_diagnostics.py --n-diag 200 --family all --include-stack
```

WIRE full R1 grid:

```bash
cd ~/beels_tda/tda_data_poisoning_prevention
TDA_DATA_DIR=~/wire/DataSets/PayloadByte_UNSW \
TDA_RESULTS_DIR=$PWD/results \
  venv312/bin/python run_phase_q_experiment.py --family all --seed all
```

Summarize the completed artifact (read-only, population SD, no tuning):

```bash
venv312/bin/python tools/phase_q_r1_summarize.py
```

The R1 artifact is written after every completed family/seed and skips existing
runs, so the command is safely resumable.  Do not run concurrent writers against
the same JSON file.

Artifacts:

- `results/phase_q_d0_support_audit.json`
- `results/phase_q_d1_d4_diagnostics.json`
- `results/phase_q_r1_multithreshold_capture.json` — complete 20-cell R1 grid
- `results/phase_q_wire_run.log` — R1 runtime log
