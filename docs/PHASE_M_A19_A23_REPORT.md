# Phase M addendum: A19--A23 disposition

Date: 2026-07-27  
Scope: corrections and read-only analysis of the completed M7 OPTICS records. The
historical M7 artifact was not edited.

## Disposition

| Item | Prior state | Disposition | Result |
|---|---|---|---|
| A19 | Noise Cell N was labelled superseded. | Corrected. | `poison.py` always combines Gaussian noise with 10--50 whole-array swaps, so it cannot generate the noise-only N cell. M7 numerically corroborates N (4.96 +/- 1.64, population SD) but does not establish N's original frame. Git archaeology found string-history hits, but no recoverable deleted N generator. N is now **orphaned, numerically corroborated by M7**, not superseded. |
| A20 | M7 wording extended duplicate/union ceilings to relaxed purity. | Applied in M8. | The M8 sweep treats label `-1` as never captured at every threshold. Duplicate and union ceilings are annotated only at exact 100% purity. The `-1` assertion passed for every family and seed. |
| A21 | A global 40% interpretation was proposed. | Applied in M8. | Each family is scored against its own unclustered ceiling; the results are below. |
| A22 | Source-cluster coverage was an unnamed alternative explanation. | Verified and recorded. | Monkam--De Lucia--Bastian report OPTICS shares 47.54%, 45.83%, 6.03%, and 0.59%, which sum to 99.99% by rounding, with no unclustered category in that accounting. This is a competing reconstruction/source-configuration hypothesis, not a tested cause. |
| A23 | MeanShift and ceiling language needed qualification. | Applied. | MeanShift's approximately 85% coarse cluster makes near-zero strict-purity capture uninformative. The 0.40 union ceiling for block reversal is algorithm-independent only at exact 100%; its 63.64% unclustered ceiling is OPTICS- and threshold-dependent. |

## M8 purity sweep (read-only M7 rescoring)

All values are percent capture, mean +/- population SD across the five recorded
seeds. Thresholds are exact 100%, then strict greater-than 95%, 90%, 80%, and
50% poison purity. See `results/phase_m_m8_purity_sweep.json` for per-seed data.

| Family | 100% | >95% | >90% | >80% | >50% | Unclustered ceiling | 40% reading |
|---|---:|---:|---:|---:|---:|---:|---|
| Block reversal | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 63.64 +/- 1.61 | Decisive |
| Block swap | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 62.20 +/- 1.41 | Decisive |
| Transpositions | 1.80 | 1.80 | 1.80 | 1.80 | 1.80 | 43.76 +/- 1.65 | Nominally reachable; uninformative |
| Cyclic shift | 6.28 | 6.28 | 6.72 | 11.16 | 12.36 | 33.72 +/- 1.70 | Structurally impossible |
| Noise (orphaned N) | 4.96 | 4.96 | 4.96 | 5.80 | 7.52 | 27.72 +/- 2.00 | Structurally impossible |

The strict-purity union-ceiling means at 100% are, respectively: 0.40, 0.32,
4.28, 13.84, and 19.32 percent. They are points for the 100% rule, not
relaxed-purity ceilings.

## Git archaeology (A19)

`git log -S '4.96'` found commits `95157c8`, `51d6594`, `1fb2c0b`, and
`7b9cfa6`; `git log -S 'gaussian'` in Python files found `95157c8` and
`90e918e`. The deleted-path search found no recoverable scratch/noise generator.
This evidence is insufficient to reconstruct N's original script or sampling frame.

## Guardrails

M8 is a rescore, not a new attack, feature-extraction, or clustering experiment.
It does not test why the source study's cluster coverage differs, and it does not
authorize a new whole-array noise run. Do not place the M8 ceiling interpretation
on the MathFest poster without explicitly labelling it as a post-M7 diagnostic.
