# Phase Q2: source-coverage reconciliation

**Status: complete. Primary disposition — (5) NOT RECONCILED. See §7.**
Run 2026-07-29 on WIRE, `venv312`, full UNSW-NB15 Payload-Byte CSV.

**One-paragraph summary.** Figure 14's ~100% row sum **can be reproduced by
dropping label `-1` and renormalizing** — our own 38.9%-noise fit sums to exactly
100% under that convention, and true capture does not move. That establishes the
convention is sufficient to produce the appearance; it does **not** establish how
the authors' fit actually handled noise, which remains unknown. Two
source-supported changes do move capture, both replicated 5/5 seeds at zero clean
false-removal: the literal `1 x 1500` raster (+3.32 ± 1.29 pp) and
`min_samples=2` (to 11.68 ± 1.75%). Neither closes the 40–70% gap, and the
`1 x 1500` raster achieves its gain while being structurally unable to support
the **claimed nontrivial H1-dependent feature map** (it computes a feature vector
perfectly well; that vector's H1 half is identically trivial). Separately,
§6.5's arithmetic **supports** a 10% poisoning proportion together with the
paper's stated all-sample Red-share denominator, and 51.8% of the feature matrix
is **redundant rows** — see the definitional note below.

> **Definitional note added by Phase Q3.** The "51.8%" here is the
> **redundancy fraction**, `(n_rows − n_unique_classes) / n_rows` =
> `2849/5500`. It is *not* the repeated-member fraction, which on the same
> frame is **57.02%** (3136 of 5500 rows belong to a class of size ≥ 2). The
> bare phrase "exact duplicate rows" is ambiguous between the two and should
> not be used without saying which is meant. See
> `docs/PHASE_Q3_COLLISION_REPORT.md`.

## Question

Can the gap between this reconstruction (single-digit poison capture) and
Monkam, De Lucia and Bastian (40–70%) be attributed to an identifiable
difference in (1) OPTICS noise accounting, (2) the TDA image geometry and
feature map, or (3) OPTICS parameters and cluster extraction?

The motivating observation is that this reconstruction assigns 38.9% of samples
to OPTICS label `-1`, while the source paper's Figure 14 cluster rows sum to
approximately 100%. That was treated as a hypothesis with two live readings —
the authors' fit produced almost no noise, or noise was omitted and the
remaining shares renormalized — and Q2 was designed to separate them.

No stage of Q2 optimized for a 40–70% number. Selection in the parameter stage
is label-free by construction and is tested to be so.

---

## 1. Source availability — what was and was not recoverable

**No lawful local copy of the source PDF exists on WIRE.** A full search of
`~` and `~/wire` (the read-only view of the same share) returned no copy of
either the journal article or the 2023 precursor; the only PDFs present are
this project's own generated poster and figures. The `.gitignore` `*.pdf` rule
excludes source PDFs for copyright reasons and was not overridden.

Open-access retrieval was attempted and failed:

| Route | Accessed 2026-07-29 | Outcome |
|---|---|---|
| ScienceDirect `S0167404824002323` (DOI `10.1016/j.cose.2024.103929`) | yes | HTTP 403, abstract only |
| ACM DL mirror `dl.acm.org/doi/10.1016/j.cose.2024.103929` | yes | HTTP 403 |
| SSRN author manuscript `abstract_id=4640844` (DOI `10.2139/ssrn.4651812`) | yes | HTTP 403, no downloadable manuscript |
| USMA Athena repository, author browse for "Monkam, Galamo F." | yes | no full text served |
| 2023 precursor, IEEE DSC, DOI `10.1109/DSC61021.2023.10354143` | yes | paywalled; abstract only |
| Search for author code, supplementary material, or notebooks | yes | **none found** |

**Consequence.** No author code, dependency manifest, notebook, or supplementary
material is recoverable. The exact image shape, the definition producing 72 and
126 features, the fitted OPTICS constructor, and the handling of label `-1` in
Figure 14 are **not** recoverable from public sources. The page-level facts in
§2 are therefore used as *transcribed and reported* by Christian from the
journal PDF, not as text this phase re-verified against primary source. Every
Q2 conclusion that leans on them is marked accordingly.

This is the single largest limitation of Q2 and it is not closable from this
machine. See §8 for the questions only the authors can answer.

---

## 2. Source-to-code matrix

Source evidence as transcribed from the journal PDF (page references are the
transcriber's). "Current code" is this repository at the Q2 commit.

| Component | Source evidence | Current code | Status | Test run |
|---|---|---|---|---|
| Input geometry | p. 6: payloads become 1-D images of size `1 x 1500`. Figs. 8–9 captions (p. 8): `30 x 50` used "for illustration purposes" | operational `30 x 50` (`tda_pipeline.reshape_for_tda`) | **mismatch** | Q2-B source-geometry arm, §5 |
| Directions | Alg. 1: `[[0,1],[1,0]]` | `[[0,1],[1,0]]` | match | — |
| Radial centers | Alg. 1: `[[0,1500],[0,750],[1500,0]]` | `[[0,50],[0,25],[30,0]]` (rescaled to image units) | **mismatch** | Q2-B source-geometry arm, §5 |
| Binarizer threshold | Alg. 1: `0.4`; §6.1 (p. 9): `0.3` | `0.4` | **ambiguous in source** | Q2-B separate threshold arm, §5.4 |
| Filtration count | Alg. 1: five (2 height + 3 radial) | five | match | — |
| Persistence | Alg. 1: `CubicalPersistence` | `CubicalPersistence`, GUDHI, `coeff=2`, `homology_dimensions=(0,1)` | match | — |
| Summaries | Alg. 1: `Scaler`, `PersistenceEntropy`, five `Amplitude` metrics; §6.1 claims "eight TDA metrics" while naming five | `Scaler`, `PersistenceEntropy`, same five `Amplitude` metrics | **source self-inconsistent** | arithmetic audit, §2.1 |
| Feature count | claims 72 and 126 | observed **60** | **unresolved** | arithmetic + runtime audit, §2.1 |
| Feature reduction | text says 126 reduced to 72 | no reduction stage exists | **unresolved — no source method recoverable** | §1, §8 |
| OPTICS parameters | §7 (p. 14) ranges only: `min_samples` ∈ [2,300], ε ∈ (0,2], `min_cluster_size` ∈ [2,400]; no fitted config, metric, Xi value or extraction rule | `min_samples=5`, `max_eps=2.0`, sklearn defaults otherwise (`clustering.py:54`) | **unresolved** | Q2-C sensitivity, §6 |
| Cluster extraction | pseudocode builds clusters from ε-neighborhoods; does not identify sklearn `xi` vs `dbscan` | `cluster_method="xi"` (sklearn default) | **unresolved — recorded, not fished** | §6.4 |
| Noise handling | no displayed noise row; Fig. 14 rows sum to ~100% | `-1` excluded from capture, reported separately | **resolved by Q2-A** | §4 |
| Sample frame | not recoverable | 5000 clean + 500 appended poison, 10% rate | **partially resolved** — §4.1 recovers the poison rate | §4.1 |
| Poison generator | Chale/Hore attacks | reconstruction permutation attacks | **known divergence** | out of scope; not conflated with the clustering audit |

### 2.1 The 72/126-feature arithmetic

Algorithm 1 as printed is unambiguous about its own arithmetic:

    5 filtrations x 6 summaries (entropy + 5 amplitudes) x 2 homology dims = 60

The runtime audit agrees. Every Q2 extraction, in **both** geometries, at
thresholds 0.4 and 0.3, returned exactly **60** columns
(`results/phase_q2_geometry.json`, `arms.*.feature_diagnostics.shape`). There is
no configuration of the printed algorithm that yields 72. 126 is not reachable
either; the paper attributes it to the 2023 precursor, which is paywalled and
whose feature construction could not be inspected.

`72 = 6 x 12` and `126 = 6 x 21`, so a plausible reading is that the authors
count more than five filtrations or more than two homology dimensions, but this
is speculation and Q2 does not adopt it. **The discrepancy stands unresolved and
is a question for the authors** (§8). It matters: Figure 14's headline rows are
labelled 72- and 126-feature, so the reconstruction is not comparing like with
like at the feature-map level, independent of everything else in this report.

---

## 3. Method

All arms share one data realization per seed, built exactly as the standing
regression gate builds it (`tools/repro_check.py`):

    load_unsw(max_samples=None)                              # 79,881 rows
    RandomState(seed).choice(len(X_all), size=5000)           # subsample
    malicious_random_attack(..., poison_rate=0.10,
                            random_state=seed, n_swaps=60)    # 5000 + 500

Seed 42: `input_hash=fce036c2424196ef`, `poison_mask_hash=8f8ee5534151e4fe`,
`n_total=5500`, `n_clean=5000`, `n_poison=500`, attack validity 100%.

New code is additive. `tda_pipeline.py`, `clustering.py`, `phase_q_metrics.py`,
`tools/repro_check.py` and every R1 driver are unmodified; the geometry module
imports the legacy pipeline and a test asserts bit-for-bit equality of the
legacy arm with it. Removal metrics are the frozen Phase Q ones
(`phase_q_metrics.removal_curve`), unchanged.

Reporting rules held throughout: label `-1` is never counted as captured;
shares are sample-weighted, never counts of clusters; capture is always reported
beside clean false-removal, precision, `-1` fractions, cluster count and
largest-cluster share; five-seed summaries use population SD (`ddof=0`); no seed
was dropped and no attack resampled.

---

## 4. Q2-A — noise-accounting audit

Artifact: `results/phase_q2_accounting_audit.json`.
One OPTICS fit on the standing seed-42 arm; three display conventions computed
from that single labelling with no refit.

### 4.1 The source's prose is internally consistent, and it fixes the poison rate

§6.5 reports a ~60.3% capture summary and color shares
`47.54%, 45.83%, 6.03%, 0.59%`. Those four shares sum to **99.99%** — there is
no noise category in the sum.

If the shares are shares of *all* samples, and Red clusters are exactly 100%
poisoned as the color scheme requires, then

    Red share of all samples = poison_rate x capture_fraction

Testing that identity against the printed numbers:

| Assumed poison rate | Implied Red share | Matches a printed share? |
|---|---|---|
| 5% | 3.02% | no |
| **10%** | **6.03%** | **yes — exactly** |
| 20% | 12.06% | no |
| 30% | 18.09% | no |

This is an exact hit, not a near miss. Three consequences follow — stated as
what the identity *supports*, which is narrower than what it would prove:

1. The identity **supports a 10% poisoning proportion**, matching this
   reconstruction's `POISON_RATE = 0.10`.
2. It **supports the paper's stated all-sample Red-share denominator**, with the
   mapping Green 47.54%, Yellow 45.83%, **Red 6.03%**, Pink 0.59%.
3. §6.5's prose is arithmetically self-consistent.

**What it does not establish.** A matching poisoning *proportion* is not a
matching *sampling frame*. The identity says nothing about how packets were
selected, what the class composition of the selected packets was, or whether
duplicate/empty payload rows were deduplicated or filtered before feature
extraction. Any of those could differ from this reconstruction while the 10%
proportion still holds. The earlier phrasing "the sampling frame is not the
source of the gap" overstated this and is withdrawn; Phase Q3 shows the
composition of the frame is in fact load-bearing (see
`docs/PHASE_Q3_COLLISION_REPORT.md`).

It is **not** consistent with Figure 14. Prose capture 60.3% equals neither
Figure 14 row (56.1% for the 126-feature row, 64.1% for the 72-feature row); it
falls between them. Combined with the duplicated cluster identifier in the
72-feature row, Figure 14 and §6.5 should be treated as two separate reported
quantities, not one.

### 4.2 Denominator choice explains the 100% sum — and nothing else

Our seed-42 fit: 127 clusters, **2141 of 5500 samples unclustered (38.9%)**,
largest cluster 18.98%.

| View | Green | Red | Pink | Yellow | Noise | Sum |
|---|---:|---:|---:|---:|---:|---:|
| All-sample denominator | 15.45 | 0.20 | 0.00 | 45.42 | 38.93 | 100.00 |
| Clustered-only, renormalized | 25.31 | 0.33 | 0.00 | 74.37 | — | 100.00 |
| Noise folded into mixed display | 15.45 | 0.20 | 0.00 | 84.35 | — | 100.00 |

**True poison capture = 2.2000% in all three views.** That invariance is
enforced in code and asserted in `tools/test_phase_q2.py`: the denominator for
capture is the full poisoned population in every view, and `-1` is never
colored Red.

So the answers to Q2-A's three questions:

- **Can the apparent 100% source coverage be reproduced by dropping `-1` and
  renormalizing? Yes.** That convention makes *our* table sum to exactly 100%
  too, from a fit that puts 38.9% of samples in noise. **Figure 14 summing to
  100% is therefore not evidence that the authors' model produced little
  noise.** But reproducing the appearance is not the same as explaining the
  source: the authors' actual noise handling remains **unknown**, and this
  result neither confirms nor refutes it. The summation simply carries no
  information about it.
- **Does any accounting choice change true capture? No.** 2.2000% in all three.
- **Are the source prose and Figure 14 internally consistent?** The prose is
  self-consistent (§4.1). Prose and Figure 14 are **not** consistent with each
  other.

### 4.3 What accounting does not do

Renormalization moves our Red share from 0.2000% to 0.3275% of the displayed
population — a factor of 1.64. The source's Red share is 6.03%. A gap of
**18.4x** remains after the most favorable accounting convention available, and
capture itself does not move at all.

**Q2-A verdict: accounting reconciliation only.** It explains the shape of the
source table. It explains none of the magnitude.

---

## 5. Q2-B — the source-printed image geometry

Artifact: `results/phase_q2_geometry.json`. One conceptual factor moves: the
raster and the radial centers, which are the same coordinate system. Same data
realization, same poison mask, same threshold, same five filtrations, same
persistence and summaries, same OPTICS.

### 5.1 The pinned library accepts the printed configuration

giotto-tda 0.6.2 accepts `(N, 1, 1500)` with centers `[[0,1500], [0,750],
[1500,0]]`. No exception, no NaN, no Inf, output dimension **60**. No center was
clipped or rescaled to make it run. Both rasters are exact factorizations of
1500, so every payload byte survives in both arms.

### 5.2 But the printed geometry cannot support the claimed nontrivial H1-dependent feature map

This is the decisive structural finding, and it is independent of any
clustering result. Per-filtration diagram audit on the deterministic fixture:

| Geometry | Filtration | Distinct filtration values | Non-trivial H0 pts | Non-trivial H1 pts |
|---|---|---:|---:|---:|
| `30 x 50` | height `[0,1]` | 31 | 198 | 1061 |
| `30 x 50` | height `[1,0]` | 51 | 190 | 1061 |
| `30 x 50` | radial `[0,50]` | 850 | 266 | 1061 |
| `30 x 50` | radial `[0,25]` | 765 | 253 | 1061 |
| `30 x 50` | radial `[30,0]` | 408 | 233 | 1061 |
| `1 x 1500` | height `[0,1]` | **2** | 2829 | **0** |
| `1 x 1500` | height `[1,0]` | 1501 | 2829 | **0** |
| `1 x 1500` | radial `[0,1500]` | 1501 | 2829 | **0** |
| `1 x 1500` | radial `[0,750]` | 1501 | 2829 | **0** |
| `1 x 1500` | radial `[1500,0]` | 1501 | 2829 | **0** |

A one-pixel-tall image has no 1-cycles, so **H1 is identically empty in the
printed geometry, across all five filtrations**. Algorithm 1 asks for homology
dimensions (0, 1). On the full 5500-sample matrix this shows up as **30 of 60
features being exactly zero-variance** in the source arm versus **0 of 60** in
the operational arm.

**Stated precisely:** the `1 x 1500` raster computes a 60-column feature vector
without error — it is not unable to produce features. What it cannot support is
the **claimed nontrivial H1-dependent feature map**: the 30 columns that are
supposed to summarize H1 are constant across every sample, so the map is
effectively 30-dimensional wearing a 60-column shape.

Separately, `HeightFiltration(direction=[0,1])` collapses to **2 distinct
values** in the printed geometry, because giotto-tda's direction convention maps
`[0,1]` onto array axis 0, which has extent 1. One of the five filtrations is
degenerate on top of the H1 collapse.

**Reading.** The printed `1 x 1500` geometry is not merely a different choice
from `30 x 50`; it is inconsistent with the paper's own Algorithm 1, which
requests two homology dimensions and claims 72 or 126 features. This is strong
evidence that the operational raster was two-dimensional and that the `30 x 50`
reading of Figures 8–9 is the correct one. It does **not** establish that
`30 x 50` specifically, or the rescaled centers specifically, are what the
authors ran — only that a 2-D raster is required. That remains unrecovered.

### 5.3 Five-seed result under fixed OPTICS

Seed 42 alone showed capture rising 2.20% → 7.80%. This project has twice been
burned by a seed-42 artifact that did not survive replication, so the arm was
re-run on all five seeds before being described as an effect. Seeds
`[42, 123, 456, 789, 1024]`, mean ± population SD, `min_samples=5,
max_eps=2.0` throughout:

| Arm | Capture % | Per-seed capture | Cluster count | Unclustered | Clean unclust. | Poison unclust. | Red share of all samples | Clean false-removal |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `legacy 30x50 t=0.4` | 1.80 ± 0.51 | 2.2, 2.2, 2.2, 1.0, 1.4 | 137.2 | 0.393 ± 0.005 | 0.376 | 0.562 | 0.164% | **0.0000** |
| `source 1x1500 t=0.4` | **5.12 ± 1.53** | 7.8, 4.2, 5.8, 4.2, 3.6 | 174.0 | 0.339 ± 0.012 | 0.320 | 0.535 | 0.465% | **0.0000** |
| `source 1x1500 t=0.3` | **11.12 ± 1.55** | 10.8, 10.0, 14.0, 11.2, 9.6 | 188.8 | 0.352 ± 0.002 | 0.324 | 0.634 | 1.011% | **0.0000** |

Geometry effect (`source t=0.4` − `legacy t=0.4`):
**+3.32 ± 1.29 pp, 5/5 seeds positive, sign-stable.**

**The legacy arm reproduces the standing record exactly.** Per-seed
`[2.2, 2.2, 2.2, 1.0, 1.4]` and 1.80 ± 0.51 are bit-identical to CLAUDE.md §6's
transpositions row. That is the internal control confirming both arms ran on
the recorded realization.

So the geometry effect is real, replicated, and free: clean false-removal is
**0.0000 on every seed in every arm**. Unclustered mass falls 0.393 → 0.339,
i.e. coverage moves *toward* Figure 14, and cluster count rises 137 → 174. The
median poison/clean distance ratio also rises (1.17 → 1.31 at seed 42), so this
is not purely a density artifact — poison does sit further out in the printed
geometry.

**But it does not close the gap, and its mechanism is compromised.** 5.12% is
still an order of magnitude short of 40–70%, and the Red share reaches 0.465%
against the source's 6.03%. More importantly, §5.2 showed the gain is produced
by a feature map in which **half the columns are identically zero** — the
printed geometry improves capture while being structurally unable to compute
the H1 features Algorithm 1 asks for. A capture gain obtained by silently
discarding a homology dimension is not evidence that the authors' geometry was
`1 x 1500`; the more economical reading is that a lower effective dimension
tightens OPTICS density estimates on this data.

### 5.4 Threshold arm

A separate one-factor comparison, source geometry and OPTICS held fixed, `0.4`
vs `0.3` — the two values the source prints in different places (Algorithm 1 vs
§6.1). Never combined with the geometry comparison.

Threshold effect (`t=0.3` − `t=0.4`, source geometry):
**+6.00 ± 1.73 pp, 5/5 seeds positive, sign-stable.**

Taking both source-printed readings together — the printed raster *and* the
prose threshold — gives **11.12 ± 1.55%** at zero clean cost. That is 6.2x the
standing 1.80% and the best result in this project to date, and it is still
**3.6–6.3x short of the source's 40–70%**. Note also that poison unclustered
mass *rises* here (0.535 → 0.634) even as capture rises: the threshold change
is finding a small number of tight all-poison clusters, not improving coverage.

---

## 6. Q2-C — provenance-first OPTICS sensitivity analysis

Artifact: `results/phase_q2_optics_sensitivity.json`.

**This is a sensitivity analysis, not a source reproduction.** The publication
prints only ranges (§7, p. 14) and no author code was recoverable (§1), so no
configuration found here can be attributed to the authors.

Discovery ran on the seed-42 legacy matrix only, one factor at a time from the
baseline `min_samples=5, max_eps=2.0, cluster_method="xi"`, sklearn defaults
otherwise, over the source-anchored ranges. A separate `max_eps=inf` cell is
included and labeled as outside the paper's stated ε range.

### 6.1 The label-free selection rule, fixed in source before any cell was run

    feasible  = {cells : n_clusters >= 2 and largest_cluster_share <= 0.95}
    rank      by unclustered_fraction ascending
    tie-break by |n_clusters - 7.5| ascending      (Fig. 14 displays 7 and 8 entries)
    lock      the top 3

The rule reads exactly three quantities: `n_clusters`,
`largest_cluster_share`, `unclustered_fraction`. It never sees the poison mask,
purity, capture or color shares. This is enforced by a test that hands
`discover_candidates` a record set with inverted capture statistics attached and
asserts the selection is unchanged, and by a test that scans every discovery
record in the artifact for any field whose path contains a poison statistic.

The two guards exist because "lowest noise" alone is trivially gamed by a
configuration that puts everything in one cluster; that is not a detector.

### 6.2 Discovery — label-free structure across the swept ranges

Seed 42, legacy `30 x 50`, threshold 0.4. 21 cells, 3 of them exact duplicates
of the baseline and reused rather than refitted.

| Cell | Clusters | Unclustered | Largest share | Median cluster size |
|---|---:|---:|---:|---:|
| `max_eps=0.25` | 79 | 0.468 | 0.190 | 9.0 |
| `max_eps=0.5` | 96 | 0.443 | 0.190 | 9.0 |
| `max_eps=1.0` | 109 | 0.415 | 0.190 | 9.0 |
| `max_eps=2.0` (baseline) | 127 | 0.389 | 0.190 | 9.0 |
| **`min_samples=2`** | **592** | **0.255** | 0.190 | **2.0** |
| `min_samples=10` | 56 | 0.432 | 0.190 | 17.5 |
| `min_samples=25` | 25 | 0.429 | 0.190 | 51.0 |
| `min_samples=50` | 14 | 0.453 | 0.190 | 124.0 |
| `min_samples=100` | 9 | 0.474 | 0.190 | 264.0 |
| `min_samples=300` | 4 | 0.527 | 0.190 | 615.0 |
| `min_cluster_size=2` | 142 | 0.392 | 0.190 | 7.0 |
| `min_cluster_size=10` | 67 | 0.424 | 0.190 | 15.0 |
| `min_cluster_size=25` | 28 | 0.458 | 0.190 | 36.0 |
| `min_cluster_size=50` | 20 | 0.433 | 0.190 | 66.5 |
| `min_cluster_size=100` | 17 | 0.378 | 0.190 | 110.0 |
| `min_cluster_size=300` | 5 | 0.516 | 0.190 | 383.0 |
| `min_cluster_size=400` | 4 | 0.496 | 0.190 | 621.0 |
| `max_eps=inf` *(outside paper range)* | 162 | 0.329 | 0.190 | 8.5 |

**No configuration anywhere in the paper's stated ranges drops unclustered mass
below 0.255.** The floor across the whole sweep is 25.5%, against a source
figure that displays none. Nothing in the OPTICS parameter space the publication
supports produces a table that would naturally be drawn without a noise row.

**`largest_cluster_share` is 0.190 in every single cell.** That is not a
coincidence of tuning: the feature matrix has a **redundancy fraction of
2849/5500 = 51.8%** (equivalently, a repeated-member fraction of 57.02%; Phase
Q3 defines both), with a single degenerate block of **1043 identical
60-vectors** (`results/phase_q2_accounting_audit.json`,
`feature_diagnostics.max_duplicate_multiplicity`). One cluster of ~1044
identical points survives every parameter choice because no density-based method
can split points at distance zero. This is a property of the data
representation, upstream of clustering, and it is discussed further in §7.3.

The label-free rule locked, in order: `min_samples=2` (unclustered 0.255),
`max_eps=inf` (0.329, labeled outside range), `min_cluster_size=100` (0.378).

### 6.3 Confirmation — five seeds, candidates unchanged

Poison metrics computed here for the first time. Mean ± population SD.

| Candidate | In paper range? | Capture % | Per-seed capture | Clusters | Unclustered | Clean false-removal | Precision | Red share of all samples |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `min_samples=2` | **yes** ([2,300]) | **11.68 ± 1.75** | 9.0, 12.8, 13.4, 10.2, 13.0 | 598.6 | 0.257 | **0.0000** | 1.00 | 1.06% |
| `max_eps=inf` | no | 1.84 ± 1.12 | 2.2, 2.2, 3.4, **0.0**, 1.4 | 171.6 | 0.341 | 0.0000 | 1.00 | 0.17% |
| `min_cluster_size=100` | yes ([2,400]) | **0.00 ± 0.00** | 0, 0, 0, 0, 0 | 15.4 | 0.402 | 0.0000 | — | 0.00% |

Only one candidate moves capture: `min_samples=2`, to **11.68 ± 1.75%**, on all
five seeds, at **zero clean false-removal** and precision 1.00. It was selected
without any label ever entering the rule, and it sits inside the paper's own
stated `min_samples` range. The relaxed >50%-purity budget adds little
(12.84 ± 1.84% at clean cost 0.0006).

The other two candidates are negative results and are reported as such:
`max_eps=inf` is sign-unstable (0.00% on seed 789) and no better than baseline;
`min_cluster_size=100` captures nothing on any seed.

### 6.4 What `min_samples=2` actually does — and why it is not a fix

Median cluster size is **2.0** and cluster count is **598.6 on 5500 samples**.
`min_samples=2` admits two-point clusters, and any two poisoned packets that
land adjacent form a 100%-pure Red cluster by definition. The capture gain is
real and it is free of clean cost, but the mechanism is that the purity
criterion becomes trivially satisfiable at small cluster sizes, not that poison
became more separable.

It should be recorded honestly in both directions. It is **not** a tuning
artifact: it was locked label-free, it replicated 5/5, and it costs nothing in
false removal. It is **also** not a plausible reconstruction of Figure 14, which
displays 7–8 clusters, not 599. The two facts sit together and Q2 does not
resolve them.

### 6.5 Cluster extraction remains unresolved

The paper's pseudocode builds clusters from ε-neighborhoods, which does not
distinguish sklearn's `cluster_method="xi"` from `"dbscan"`. No direct evidence
favors either. **No extraction-method arm was run**, because running one would
have meant fishing across `eps` values with nothing to anchor the choice. This
is recorded as unresolved rather than guessed, and it is question 3 in §8.

---

## 7. Disposition

### 7.1 Primary: **(5) Not reconciled**

No source-supported test closes the 40–70% gap. The best result Q2 produced from
any single source-supported change is **11.68 ± 1.75%** (`min_samples=2`), and
the best from the source-printed feature map is **11.12 ± 1.55%**
(`1 x 1500` raster with the §6.1 threshold 0.3). Both are 3.4–6.0x short of the
source's reported range, and both are 4–6x short of the source's 6.03% Red
sample share.

### 7.2 Established secondary findings

**(1) Accounting reconciliation — confirmed, and it explains only the shape.**
Figure 14's rows summing to ~100% carries **no information** about whether the
authors' fit produced noise: dropping `-1` and renormalizing makes our own
38.9%-noise fit sum to exactly 100% as well. The question the motivating
observation was meant to settle is therefore **not settled by the summation**,
in either direction. What accounting does not do is change capture — 2.2000% in
all three views, enforced in code and tested.

Alongside this, §4.1 recovered something the paper does not state directly: the
identity `Red share = poison_rate x capture` **supports** a **10%** poisoning
proportion for the source, exactly, and **supports** reading §6.5's 6.03% as the
Red share on the paper's stated all-sample denominator. It does **not** show
that packet selection, class composition, or deduplication matched this project;
those remain unrecovered. §6.5 is internally consistent; §6.5 and Figure 14 are
not consistent with each other.

**(2) Geometry — a real, replicated, partial effect with a compromised
mechanism.** The literal `1 x 1500` raster raises capture +3.32 ± 1.29 pp,
5/5 seeds, sign-stable, at zero clean cost, and moves unclustered mass
0.393 → 0.339, i.e. toward Figure 14. But the printed geometry has **identically
empty H1**, leaving 30 of 60 features exactly zero-variance, so while it does
compute a 60-column vector, it cannot support the **claimed nontrivial
H1-dependent feature map** Algorithm 1 specifies. The gain is most economically
read as lower effective dimension tightening density estimates. This is evidence
that the operational raster was 2-D — not evidence that it was `1 x 1500`.

**(3) Parameter sensitivity — real, bounded, and not a reproduction.**
`min_samples=2` is inside the paper's stated range and reaches 11.68 ± 1.75%
label-free across five seeds at zero clean cost. Because the publication prints
no fitted configuration and no author code was recoverable, **this cannot be
attributed to the authors** and is not a reproduction claim.

### 7.3 The finding Q2 did not go looking for

**The feature matrix has a redundancy fraction of 51.8% (2849/5500), and the
largest exact-collision block is 1043 identical 60-vectors.** This is why
`largest_cluster_share` is pinned at 0.190 under every one of the 21 swept
configurations: roughly a fifth of the dataset is a single point in feature
space, and no density-based method can ever split it. Any detector operating on
this representation carries that block as an irreducible floor.

This was not on the Q2 question list and Q2 did not test it. It is the most
promising unexplored lead the phase turned up, and it is upstream of every
clustering result in the project.

> **Followed up in Phase Q3.** `docs/PHASE_Q3_COLLISION_REPORT.md` traces these
> collisions to their earliest pipeline stage. Summary: ~51% of the final
> collision mass is already identical in the raw 1500-byte payload, ~47% is
> first created by threshold-0.4 binarization, and only ~2% originates in the
> topological feature map. The 1043-block is a **mixed** clean/poison class, and
> its poison members are raw-*distinct* from their clean sources but
> binary-identical to them.

### 7.4 What Q2 explicitly does not claim

No full reproduction of Monkam, De Lucia and Bastian is claimed or approached.
The source attack, the sampling frame beyond the recovered 10% rate, the 72- and
126-feature maps, and the exact OPTICS configuration all remain unrecovered.
Q2 performed no threshold selection on results, no block weighting, no feature
selection, no OPTICS tuning beyond the preregistered label-free sweep, no attack
resampling and no seed deletion. **The combination of source geometry +
threshold 0.3 + `min_samples=2` was deliberately not run**: combining three
factors after seeing each one's result is exactly the search this phase was
designed to avoid. If it is wanted, it should be preregistered as its own phase.

## 8. What only the authors can supply

**Drafted, not sent.** Q2 did not email anyone. Christian decides whether, when
and in what form to send this. Monkam and Bastian are at USMA's Army Cyber
Institute — the same institution — so the tone below is collegial and the
framing is "help us reconstruct," which is what it is.

Five questions, in the order they block this work:

1. **Operational image shape and radial centers.** Page 6 gives `1 x 1500` and
   Algorithm 1 gives centers `[[0,1500], [0,750], [1500,0]]`, while the Figures
   8–9 captions describe `30 x 50` as illustrative. Which raster did the
   reported runs use, and were the centers expressed in image coordinates or in
   payload-byte units? We ask because a `1 x 1500` raster has no 1-cycles, so
   `homology_dimensions=(0,1)` yields identically empty H1 there.

2. **The 72- and 126-feature maps.** Algorithm 1 as printed evaluates to
   5 filtrations x 6 summaries x 2 homology dimensions = 60. Could you share the
   code or the explicit definition that produces 72, and the 126-feature
   construction inherited from the 2023 DSC paper, together with the reduction
   step from 126 to 72?

3. **The exact OPTICS call.** Section 7 gives ranges (`min_samples` ∈ [2,300],
   ε ∈ (0,2], `min_cluster_size` ∈ [2,400]) but not the fitted values. Could you
   share the constructor as run — `min_samples`, `max_eps`, `metric`,
   `cluster_method`, `xi`, `min_cluster_size` — and, if scikit-learn was used,
   whether extraction was `xi` or `dbscan`?

4. **Label `-1` in Figure 14.** The displayed cluster shares sum to
   approximately 100% with no noise row. Did the fitted model assign essentially
   no samples to noise, or were noise samples omitted from the figure and the
   remaining shares renormalized? Our reconstruction places 30–40% of samples in
   `-1` under every configuration we have tried.

5. **Sample frame and poison construction.** The number of packets used, the
   sampling seed or selection rule, the poisoning rate, the exact attack that
   generated the poisoned packets, and the dependency versions (giotto-tda,
   scikit-learn) would let us match the frame rather than infer it.

We would also be glad to share our reconstruction code and results; the
reconstruction is public and the intent is to characterize what the pipeline
can and cannot see, not to dispute the reported numbers.

---

## 9. Reproducing Phase Q2

```bash
source venv312/bin/activate
export TDA_DATA_DIR="$HOME/wire/DataSets/PayloadByte_UNSW"
export TDA_RESULTS_DIR="$PWD/results"

python tools/phase_q2_accounting_audit.py                          # Q2-A
python programs/run_phase_q2_geometry.py --threshold-arm --confirm-seeds    # Q2-B
python programs/run_phase_q2_optics_sensitivity.py                          # Q2-C

python -m unittest tools.test_phase_q2 -v
python tools/repro_check.py --expect 2.2000                        # legacy gate
```

Feature matrices are cached in `.q2_cache/` (gitignored, regenerable). Every
tracked JSON carries the data hash, poison-mask hash, feature hash, full
estimator parameters, seeds, counts, library versions and timings, so no cached
matrix is required to audit any number in this report.
