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

## Reproduction

Windows quick checks:

```powershell
.\venv312\Scripts\python.exe -m unittest tools.test_phase_q -v
.\venv312\Scripts\python.exe run_phase_q_support_audit.py
.\venv312\Scripts\python.exe run_phase_q_diagnostics.py --n-diag 200 --family all --include-stack
```

WIRE full R1 grid:

```bash
cd ~/projects/tda_data_poisoning_prevention
TDA_DATA_DIR=~/wire/DataSets/PayloadByte_UNSW \
  venv312/bin/python run_phase_q_experiment.py --family all --seed all
```

The R1 artifact is written after every completed family/seed and skips existing
runs, so the command is safely resumable.  Do not run concurrent writers against
the same JSON file.

Artifacts:

- `results/phase_q_d0_support_audit.json`
- `results/phase_q_d1_d4_diagnostics.json`
- `results/phase_q_r1_multithreshold_capture.json` after the full R1 run
