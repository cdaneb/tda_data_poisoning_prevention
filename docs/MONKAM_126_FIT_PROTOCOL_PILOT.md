# Monkam 126-Feature Fit-Protocol Pilot

Date: 2026-08-31
Status: single-seed diagnostic complete; multi-seed gate closed

## Question

Does independently fitting the TDA preprocessing pipeline on poisoned and
unmodified observations materially change the downstream DBSCAN result relative
to fitting one pipeline once on the combined population?

Only one methodological variable changes between arms: the fit protocol for the
data-dependent `Binarizer` and `Scaler` transformations.

## Reproducibility artifacts

- Locked preregistration:
  `results/monkam_126_fit_protocol_pilot_preregistration.json`
- Seed-60 result:
  `results/monkam_126_fit_protocol_pilot_seed60.json`
- Runner:
  `programs/run_monkam_126_fit_protocol_pilot.py`
- Locked design hash:
  `546a7472b4af8bae880350997d61346ec85fc0db027692a5c9f5cbd8234f10c1`

The preregistration was written before either feature arm was fitted. The runner
reconstructs the observations and refuses to run if their indices, payload/file
hashes, feature specification, split, or clustering parameters differ from the
locked design.

## Diagnostic design

The complete 1,000-poison corpus used by the saved author notebook was not
delivered. The pilot therefore uses all 18 available poison examples and scales
the notebook's population ratios exactly:

| Population | Count | Construction |
|---|---:|---|
| Unmodified normal | 378 | deterministic sample from the 21,000-row normal block |
| Unmodified attack-category | 180 | deterministic sample, including all 11 unique named poison parents |
| Poison | 18 | every delivered `final_payload_*.csv` |
| Combined | 576 | poison fraction 3.125%, equal to 1,000/32,000 |

The unmodified 378:180 ratio is exactly 21:10, matching the author's 21,000
normal plus 10,000 unmodified attack-category construction.

Fixed analysis choices:

- seed and split: `train_test_split(test_size=0.5, random_state=60)`, no
  stratification;
- clustering partition: 288 training observations, including nine poison;
- input shape: `1 x 1500`;
- threshold: 0.3;
- feature map: the supplied seven-filtration, eight-amplitude-plus-entropy
  configuration, producing 126 coordinates;
- DBSCAN: `eps=200`, `min_samples=9`;
- `min_samples=9` is the size-scaled analogue of the notebook's
  `500/16,000`: `round(500 * 288 / 16000) = 9`.

The preregistered confirmation rule required both:

1. learned-state divergence (effective cut difference at least one byte or a
   branch Scaler difference of at least 5%); and
2. a material clustering change (absolute silhouette change at least 0.10,
   ARI at most 0.90, or poison-removal-rate change at least 0.10 at purity 1.0
   or 0.8).

## Learned preprocessing states

Separate fitting produced two genuinely different representations:

| State | Unmodified fit | Poison fit | Shared fit |
|---|---:|---:|---:|
| Binarizer maximum | 255 | 122 | 255 |
| Effective threshold-0.3 byte cut | 76.5 | 36.6 | 76.5 |
| Largest branch Scaler difference, poison vs unmodified | — | 55.761% | — |

The 55.761% difference occurs in the `radial_1500_0` branch, where the learned
scale is 729.0 for the unmodified/shared fit and 322.5 for the poison-only fit.
The shared-fit learned states equal the unmodified-fit states in this pilot
because the 18 appended poisons do not change the relevant population maxima.

The learned-state divergence changed every poison feature vector and no
unmodified feature vector:

| Rows | Exact feature rows changed | Mean row L2 change | Maximum row L2 change |
|---|---:|---:|---:|
| Unmodified (558) | 0 | 0.000 | 0.000 |
| Poison (18) | 18 | 531.500 | 613.331 |
| All (576) | 18 | 16.609 | 613.331 |

Thus, separate fitting is not a cosmetic implementation detail. It materially
changes the poison coordinates while leaving the comparison population fixed.

## Fixed DBSCAN result

Despite the large poison-coordinate displacement, the two fit protocols produce
the same cluster membership:

| Metric | Separate fit | Shared fit | Difference |
|---|---:|---:|---:|
| Silhouette, noise treated as a label | 0.920197 | 0.879012 | +0.041185 |
| Silhouette excluding noise | 0.923403 | 0.882074 | +0.041329 |
| Non-noise clusters | 2 | 2 | 0 |
| Noise observations | 1 | 1 | 0 |
| ARI between arm assignments | — | — | 1.000000 |

Cluster composition is identical in both arms:

| Cluster | Color | Size | Poison | Unmodified | Poison fraction |
|---:|---|---:|---:|---:|---:|
| -1 | Noise | 1 | 0 | 1 | 0.000% |
| 0 | Yellow/mixed | 249 | 9 | 240 | 3.614% |
| 1 | Green/unmodified-only | 38 | 0 | 38 | 0.000% |

No Red or Pink cluster is formed. At every evaluated cluster-purity threshold
(1.0, 0.95, 0.90, 0.80, and 0.50), both arms remove zero poison and zero
unmodified observations.

The high silhouette score is therefore not evidence of poison separation. It
primarily describes separation between a 38-row unmodified-only cluster and the
249-row mixed cluster containing every poison observation in the clustering
partition.

## Decision

The learned-state divergence criterion is met, but the material-clustering-
effect criterion is not:

- effective byte-cut difference: 39.9;
- maximum relative Scaler difference: 55.761%;
- silhouette difference: 0.041185, below 0.10;
- ARI: 1.0, above 0.90;
- poison-removal-rate difference: 0 at purity 1.0 and 0.8.

Therefore, the preregistered single-seed mechanism is **not confirmed at the
DBSCAN decision level**, and the multi-seed gate remains closed. No additional
seeds were run.

## Interpretation boundary

This pilot supports two simultaneous conclusions:

1. The supplied separate-fit procedure does encode batch membership into the
   feature coordinates and modestly raises silhouette in this diagnostic.
2. Under the locked miniature design and fixed DBSCAN parameters, that encoding
   is insufficient to change cluster assignments or recover any poison.

It would be incorrect to claim from this pilot that separate fitting caused the
published poison separation. It would also be incorrect to treat the fit
protocol as harmless: all poison feature vectors moved substantially. A
definitive test still requires the missing 1,000 poison examples and exact
published observation set.

## Paper implication

The manuscript can now report the fit protocol as a verified representation-
level confound whose downstream effect was **not confirmed in the available
18-poison diagnostic**. This is stronger and more precise than either assuming
the notebook's high silhouette demonstrates poison separation or asserting that
separate fitting manufactured the published result.
