# CLAUDE.md — TDA Poisoning Detection

**Authoritative project context. Last updated 2026-07-23, after Phase P.**

---

## 0. Precedence — read before trusting any other document

This file supersedes every other markdown in this project. Where any other document disagrees with
this one, **this one is correct**.

| Document | Status |
|---|---|
| `CLAUDE.md` (this file) | **Authoritative** |
| `docs/PROJECT_HANDOFF_1.md` | Historical. §4 tables, §5 gitignore claim, and §6 priorities are **stale** — see §9 |
| `docs/SESSION_SUMMARY_2026-07-23.md` | Historical. §2's HDBSCAN claim is now **false** — see §9 |
| `docs/TERMINOLOGY.md` | Definitions valid; **§2 and §5 radial centers are wrong** — see §9 |
| `docs/MATHFEST_POSTER_SKELETON.md` | Superseded by the current `poster_blocks.tex` |
| `cc_summary.md` | Historical narrative. Do not treat as current state |
| `LENS*.md` instruction files | **Dead.** The four-lens program was abandoned at the pivot |
| Any "Phase R/W/P" completion report | Accurate for what it reports; §9 lists its known errors |

If a stale document tells you to do something this file does not list in §8, **do not do it**. Ask.

---

## 1. What this project is

**Question:** What is it about the interaction between an attack's invariance structure and a
filtration's sensitivities that determines whether poisoning is topologically detectable?

**Working title:** *What Can a Persistence Diagram See? Invariance Structure and the Detectability of
Data Poisoning.*

Author: Christian Dane Beels, advised by Dr. Joseph Dorta, Dept. of Mathematical Sciences, USMA West
Point. **Monkam and Bastian (the source paper's authors) are at USMA's Army Cyber Institute — same
institution. Keep every characterization of their work collegial and precise.**

The project reconstructs the pipeline from Monkam, De Lucia & Bastian, *Computers & Security* 144
(2024) 103929, which maps packet payload bytes to images, computes cubical persistence, and clusters
to remove poisoned packets before training. The reconstruction captures single-digit percentages
against their reported 40–70%. **The original hypothesis — that the gap was caused by unrealistically
weak synthetic poison — was tested with a surrogate-guided attack and refuted.** The gap is
structural. The project pivoted to characterizing what the detector can and cannot see.

**Immediate context: MAA MathFest 2026, Boston, Aug 5–8. Abstract already submitted under the old
framing.** Do not start work that cannot land before the conference unless explicitly told it is
paper work.

---

## 2. Claim status (current)

### Claim 1 — PRIMARY, deductive, proof-backed, empirically confirmed
Binarization $b$ is pointwise, hence **equivariant**: $b(x \circ \sigma) = b(x) \circ \sigma$.
Summation is symmetric, so the foreground count $\sum_i b(x_i)$ is **invariant** under every
$\sigma \in S_{1500}$. Count-derived features are therefore blind to permutation attacks; detection
must flow through the position-dependent (Height/Radial) filtrations.

- **Confirmed:** 1,600 count checks (200 packets × 4 families × 2 thresholds), zero exceptions.
- **Scope caveat that must always be stated:** giotto-tda's `Binarizer(threshold=0.4)` cuts at a
  *fraction of the fitted* `max_value_`, i.e. data-dependent. The proof survives because a maximum is
  itself permutation-invariant. `max_value_` invariance is verified in code, not assumed.
- This is the load-bearing result.

### Claim 2 — SECONDARY, UNSW-specific, does NOT replicate
Surrogate guidance raised capture by **+4.68 ± 1.57 on UNSW-NB15, 5/5 seeds** (replicated at
threshold 0.3: +5.24 ± 2.35, 5/5). On CICIDS2017 the effect is **−0.60 ± 3.03, 2/5 seeds positive** —
sign flips per seed, mean crosses zero. Reported as a genuine negative result under pre-commitment.
**Never present Claim 2 without the non-replication in the same visual field.**

### Claim 3 — SECONDARY, empirical, supported on two datasets
Detectability is not ordered by attack realism; it tracks **spatial disruption**. Unguided cyclic
shift (6.28%) ≈ guided search (6.48%). Guidance is an efficient route to disruption, not a distinct
phenomenon.

### The subgroup extension — RESOLVED DOWNWARD by Phase P
Older docs frame same-side-of-threshold permutations ($S_{|B_0|} \times S_{|B_1|} \le S_{1500}$) as
"potentially the strongest result," with bit-identity upgrading the two 0.00% cells to *proven
invisibility*. **That conditional did not fire.** Bit-identity holds for **88.0% (block reversal) and
83.0% (block swap)** of samples — majority, not universal. Any "proven invisibility" pathway is
**deleted**. The 0.00% capture result stands regardless: a sample with a different feature vector can
still fail to land in a 100%-pure cluster.

---

## 3. Environment (WIRE)

| Path | Backing | Mode |
|---|---|---|
| `/home/jovyan` (`~`) | NFS `:/WIREUsers/christian.beels` | **rw**, 37T avail |
| `/home/jovyan/wire` | NFS `:/` (whole WIRE fs) | **ro** |
| `/home/jovyan/local_scratch` | `/dev/sda2` exfat | rw, 3.1T |

- **Repo: `~/projects/tda_data_poisoning_prevention`.** Do not work under `~/wire` — it is the
  read-only view of the same share.
- `local_scratch` is exfat: **no symlinks, no POSIX modes**, may not survive rescheduling.
  Disposable intermediates only.
- 128 cores, 1.0 TiB RAM, Python 3.12.3.
- CUDA is present but **irrelevant** — GUDHI cubical persistence and sklearn OPTICS are CPU-only.
- Pinned in `requirements.lock.txt`: giotto-tda 0.6.2, scikit-learn 1.3.2, numpy 1.26.4,
  scipy 1.17.1, pandas 3.0.5, joblib 1.5.3.

**Data** (both 1505 columns: 1500 `payload_byte_N` + ttl, total_len, protocol, t_delta, label):
- `~/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv` — 79,881 rows
- `~/wire/DataSets/PayloadByte_CICIDS17/Payload_data_CICIDS2017.csv` — 1,410,255 rows
- The raw `UNSW-NB15/` and `CICIDS2017/` directories are **flow-level and unusable** (no payload bytes).

---

## 4. Pipeline mechanics (verified against giotto-tda source)

For one packet: **1500 bytes → 30×50 grayscale image → binarize → 5 filtrations → cubical persistence
→ vectorize → 60 features.**

- `Binarizer(threshold=0.4)`; threshold is a fraction of fitted `max_value_` (≈255 for byte data, so
  effective cutoff ≈102). **"Same side of threshold" ≠ "both zero."**
- Filtrations, verbatim from code: `direction_list = [[0,1],[1,0]]`,
  `center_list = [[0,50],[0,25],[30,0]]`. All five are position-dependent.
- `CubicalPersistence` with the **GUDHI** C++ backend, `coeff=2`, `homology_dimensions=(0,1)`,
  sublevel-set filtration, pixels as top-dimensional cells.
- Vectorization: `Scaler`, then `PersistenceEntropy` + five `Amplitude` metrics (bottleneck,
  Wasserstein, landscape, Betti, heat) → 6 transformers × 2 dims × 5 filtrations = **60 features**.
- **Feature count is 60, empirically confirmed — not the 72 the source paper claims.** Their
  Algorithm 1 as printed yields 60. Treat as a second internal inconsistency in the source paper
  alongside the binarizer 0.4/0.3 ambiguity.
- **Capture %** = poisoned samples landing in **Red** (100%-pure) clusters ÷ total poisoned × 100.
  Cluster colors: Green 0%, Red 100%, Pink >80% but <100%, Yellow mixed. Purity literals are
  hardcoded in `clustering.py:105–112`; there is no parameter.
- **Two distinct persistence computations exist. Never conflate them.** `CubicalPersistence` on
  packet images is the feature extractor and **is what the proof concerns**.
  `VietorisRipsPersistence` in `iterative_filter.py` runs on the 60-dim residual point cloud and is
  descriptive only.

---

## 5. Repo state and key files

**Runnable / current:**
- `paths.py` — path config; replaced hardcoded `Path(r"C:\TDA\...")` across the codebase
- `data_loader.py`, `tda_pipeline.py` — `build_tda_pipeline(threshold=0.4)` and
  `extract_tda_features(X, pipeline=None, threshold=0.4)` are both parameterized as of Phase P
- `adversarial_attack.py` — `malicious_random_attack` (transpositions) plus Phase P's
  `block_reversal_attack(k=120)`, `block_swap_attack(k=60)`, `cyclic_shift_attack`
- `invariance_check.py` — `foreground_count`, `positions_changed`, `crossed_threshold`,
  `max_value_check`. Reuses `gtda.images.Binarizer` directly rather than reimplementing the rule
- `test_b_diagnostics.py`, `run_test_b_capture.py` — Phase P drivers
- `clustering.py`, `iterative_filter.py`, `run_lens4_baseline.py`
- `tools/repro_check.py` — tracked regression test, supports `--expect`
- `models/surrogate_mlp.joblib` (0.9690), `models/surrogate_rf.joblib` (0.9850),
  `surrogate_mlp_cicids.joblib` (97.7%), `surrogate_rf_cicids.joblib` (99.4%)
- `results/test_b_diagnostics.json`, `results/test_b_permutation_families.json` — seed-namespaced,
  with `_reference` and `_summary` blocks
- `poison.py` — **legacy**, always couples noise with an uncontrolled 10–50 swap count on the whole
  dataset. Do not use it to isolate noise as a variable

**Reproduction gate (run this before and after any pipeline change):**
```
python tools/repro_check.py --expect 2.2000
```
Must return **2.2000% exact** with `X_tda.shape == (5500, 60)`. This is cell S / R60: seed 42,
threshold 0.4, UNSW, OPTICS, `n_swaps=60`.

**Git: no identity is configured on this container, and Phase W + Phase P work is staged but
UNCOMMITTED.** The user commits themselves. **Do not run `git config` or invent an identity, and do
not commit.**

---

## 6. Results of record

All OPTICS, population SD, 5 seeds `[42,123,456,789,1024]`, UNSW-NB15, threshold 0.4 unless noted.
Other clustering algorithms ≈0 except where stated.

### Test B — four permutation families (PRIMARY EVIDENCE, poster figure V3)
| Family | Count change | Mean pos. changed | Median | % zero footprint | Capture % |
|---|---|---|---|---|---|
| Transpositions (60) | 0/200 | 22.93 | 8.0 | 12.5% | 1.80 ± 0.51 |
| Block reversal (k=120) | 0/200 | 14.08 | 0.0 | 84.5% | **0.00 ± 0.00** |
| Block swap (2×k=60) | 0/200 | 17.63 | 0.0 | 78.5% | **0.00 ± 0.00** |
| Cyclic shift | 0/200 | 281.09 | 89.0 | 0.0% | 6.28 ± 1.31 |

Per-seed capture: transpositions `[2.20, 2.20, 2.20, 1.00, 1.40]`, cyclic shift
`[6.60, 6.60, 8.40, 5.00, 4.80]`. Transpositions reproduced to full floating-point precision
(`2.1999999999999997`) — the internal control.

**Bit-identity** (one *combined* clean+perturbed pipeline fit): block reversal 88.0% (176/200),
block swap 83.0% (166/200). Binarized-identical and feature-identical match exactly.

### Count invariance / Step 0
- 0/200 at thresholds 0.4 **and** 0.3, all four families. 1,600 checks total, zero exceptions.
- Clean mean foreground count: **62.305/1500 at threshold 0.4**, **114.305/1500 at 0.3**.
- Positive control (Gaussian noise σ=30, same 200 samples): threshold 0.4 → mean Δ +11.5,
  mean |Δ| **14.02**; threshold 0.3 → mean Δ +19.945, mean |Δ| 29.705.

### Main factorial (secondary)
| Cell | Noise | Swaps | Capture % |
|---|---|---|---|
| N | on | — | 4.96 ± 1.64 |
| S | off | 60 random | 1.80 ± 0.51 |
| SG | off | 60 guided (MLP) | 6.48 ± 1.24 |
| SG-RF | off | 60 guided (RF) | 7.92 ± 3.86 ⚠ no backing artifact |
| NS | on | 60 random | 3.52 ± 1.79 |
| NSG | on | 60 guided | 4.88 ± 1.52 |

**7.92% is the study maximum and it was swap-only** — this is why "invisible" is never the right word.

### Test C — CICIDS2017 (does not replicate)
S = 3.28 ± 2.53, SG = 2.68 ± 1.49, effect **−0.60 ± 3.03, 2/5 seeds positive**.
Per-seed Δ: `[−0.40, −2.20, +0.40, +4.20, −5.00]`.

### Provenance status
Test C, the noise cells (N/NS/NSG), SG-RF, and L_unmatched are **orphaned or artifact-incomplete** —
produced by scratchpad scripts that were never committed. Test A (threshold 0.3) is orphaned as
recorded, though the threshold is now parameterized so it is cheap to regenerate. Survivable for a
poster; disqualifying for a paper. Test B and Step 0 were orphans and are now **fully rebuilt and
backed by committed code**.

---

## 7. House conventions — maintain these

- **Phased instructions** as markdown with numbered steps, explicit **stop conditions**, and
  completion reports structured around those steps. **No silent scope creep.**
- **Read-only investigation before any build.** Build phases use **blocking empirical gates** — each
  must pass with reported numbers before the next step runs.
- **Findings are sanity-checked empirically before being trusted**, even when the reasoning sounds
  airtight.
- **Deviations reported honestly, never rounded up.** "Recommended but not confirmed run" is a
  finding, not a failure.
- **Pre-commit to reporting results whichever way they land, before running.** If a result
  contradicts the primary claim: **stop and report. Do not rerun, do not rescope.**
- **No single-seed number ever goes on the poster.** Seed-42 artifacts have burned this project twice
  (the "G60-RF is 2× G60-MLP" claim, sign-unstable across seeds; the "legacy = 6.6%" figure, actually
  2.56 ± 2.21 multi-seed).
- **Population SD throughout.** Every ± must be recomputable from the per-seed tables.
- Update `CLAUDE.md` **only when a phase actually lands**, never from an investigation.
- When calling `extract_tda_features` to compare clean vs. perturbed, **always use one combined fit**
  (concatenate, fit once, split). See §9.

### Language guardrails
- **Never "invisible," "evades," or "defeats."** Say **attenuated on the foreground-count channel**.
- **Equivariance ≠ invariance.** Binarization is equivariant; the *count* is invariant. Both steps
  are needed. Calling the displayed equation "the invariance" is the single easiest error to make.
- **Two senses of "permutation invariance."** Standard TDA usage = a persistence diagram is a
  multiset (unordered points). This project's usage = permuting *source pixels*. Distinguish
  explicitly; a TDA-literate listener hears the standard meaning first.
- **The mathematics is elementary folklore.** Claim novelty for the *observation that it creates a
  concrete blind spot in this security pipeline*, not for the proof. Overclaiming mathematical
  originality is the main risk at a mathematics venue. Preferred hedge: "we find no prior statement
  of it for this pipeline."
- **Prefer "provably insensitive" over "provably closed"** in poster copy — same meaning, less
  semantic debate.
- **Qualify the blind-spot claim:** blind spots *arising from feature-map invariances* can be read
  off the algebra. The algebra does not characterize every failure mode (Test C is a counterexample).
- Distinguish **attacker-side** metrics (83.2% surrogate flip rate) from **detector-side** metrics
  (capture %).
- **Multiset-preserving**, not "functionality-preserving" in general. Whether functionality-preserving
  attacks inherit the attenuation is a conjecture this work motivates, not a result.

---

## 8. Open work, in priority order

**Poster (MathFest is Aug 5–8; deadline governs everything):**
1. Build figures **V2** (binarized clean/permuted/noised with foreground counts beneath) and **V3**
   (four-family comparison). V3 is the best evidence and has no figure yet.
2. Poster copy is at ~600 words in the current `poster_blocks.tex` (two-column footer layout,
   References spanning the left 0.67 only). Apply the §7 language guardrails to any new copy.
3. Fix the 2p(1−p) figure wherever it appears in talk prep — see §9 item 1.

**Post-MathFest, highest value first:**
4. **Per-filtration ablation.** The claim that detection flows through Height/Radial is currently an
   *inference* from the algebra plus the family spread, not a measurement. Re-cluster on feature
   subsets grouped by filtration. The 60-dim features are already computed — this is a feature-slicing
   loop and a re-run, not new pipeline code. **This is the strongest single addition available.**
5. **Phase F:** on-disk `X_tda` cache + purity instrumentation. Unblocks everything downstream.
   `classify_clusters()` currently computes per-cluster `poison_fraction` in memory and **discards
   it**; only per-color counts and pooled percentages are saved.
6. **Tier 0.1 — explain the 40–70% gap.** Cheapest candidate is the zero-tolerance purity criterion.
   Sweep purity at 100/95/90/80/>50%. If the curve crosses 40%, the project was measuring a stricter
   quantity, not failing to reproduce. Requires a re-run (see item 5). **Also: email Monkam/Bastian —
   same institution.**
7. **Tier 0.2 — separate "the diagram didn't change" from "OPTICS didn't notice."** Every number in
   the project is capture %, downstream of clustering. Compute bottleneck/$W_p$ between clean and
   poisoned **cubical** diagrams, per family. **Keep the five filtrations' diagrams separate** — per
   filtration distances answer the Height-vs-Radial question the whole paper rests on. Diagrams are
   not currently bound to a variable inside the sklearn pipeline; new code is required to capture them.
8. Regenerate Test A, Test C, and the noise cells with committed code (paper requirement).
9. Tier 1: characterize and prove the subgroup acting trivially on the full 60-dim feature vector;
   derive the trivially-acting fraction as a function of foreground fraction $p$ and use it to
   **predict** the Test B ordering rather than rationalize it.

**Dead — do not revive without explicit instruction:** Lens 1 (per-point topological influence,
stuck), Lens 3 (game-theoretic defense, minimax degenerates under a static attacker), TVI (never
computed), Hore / Deep PackGen (representation gap, no public code).

---

## 9. Known errors in the record — do not propagate

1. **The 2p(1−p) comparison in the Phase P report is wrong twice over.** It compares a *conditional*
   ratio (crossed ÷ positions-changed) to an *unconditional* probability, and it uses $p \approx
   0.127$ from Test A's recorded 190.6/1500 baseline — a **threshold-0.3** figure applied to a
   threshold-0.4 prediction, which Phase P's own measurement contradicts. **Correct version:** at the
   measured threshold-0.4 count, $p = 62.305/1500 = 0.0415$, so $2p(1-p) = 7.96\%$; the unconditional
   crossed rate is 7.26 per 120 position-slots = **6.05%**. That is a real match and it corroborates
   the new foreground count over the old one. Use ~8%, never 22%.
2. **The "crossed / Δpos" column is a mean of per-sample ratios with 0/0 scored as zero**, not a
   ratio of means. So block reversal's 4.9% mostly re-reports the 84.5% zero-footprint rate. The
   cleaner finding underneath: **conditional on touching anything at all, the crossing rate is ~28–33%
   for all four families.** Families differ only in how often they touch anything.
3. **190.6/1500 (Test A, threshold 0.3) vs 114.305/1500 (Phase P, threshold 0.3) is unreconciled** and
   is missing from Phase P's §P7 reconciliation table. Plausible cause is a different sample frame
   (Phase P draws malicious-only targets). Reconcile or flag explicitly — this number feeds item 1.
4. **`SESSION_SUMMARY_2026-07-23.md` §2 is now false where it says HDBSCAN was "~0% across every UNSW
   variant in the entire investigation."** Phase P found HDBSCAN nonzero on cyclic shift at 3/5 seeds
   (2.00%, 3.20%, 3.80%) — the first nonzero HDBSCAN on UNSW. The CICIDS-is-structurally-different
   argument still stands on the guidance non-replication, but this supporting leg needs restating.
5. **`PROJECT_HANDOFF_1.md` §4's Test B "mean positions value-changed" column is superseded**
   (21.96 / 16.42 / 14.70 / 266.6). The block reversal vs. block swap ranking **flips** in the
   rebuilt data. No seed was recorded for the original draw, so an exact match was never expected.
   Use the §6 table here and report medians alongside means.
6. **`TERMINOLOGY.md` §2 and §5 give the radial centers as `[0,1500]`, `[0,750]`, `[1500,0]`.** The
   code says **`[0,50]`, `[0,25]`, `[30,0]`**. The docs substituted the 1500-byte payload length for
   the 50-pixel image width. A center 30× outside a 30×50 grid would make the filtration nearly
   constant — a TDA-literate reader will notice.
7. **`PROJECT_HANDOFF_1.md` §5's gitignore claim is false.** `CLAUDE.md`, `cc_summary.md`, and all
   LENS instruction files are **tracked and present**. The `results/*.json` exclusion is commented
   out; all JSON files are tracked.
8. **The `Scaler` per-batch artifact.** Fitting `extract_tda_features` separately on a clean batch and
   a perturbed batch produces **spurious** feature differences, because `Scaler` normalizes per batch.
   An earlier bit-identity check reported 16/200 false discrepancies this way. Always fit once on the
   concatenation. **Before any new diagram-distance work, grep for other two-call comparison sites.**
9. **The attribution ladder was originally confounded** — noise was not a controlled variable, voiding
   the early L→R60 comparison. Do not reintroduce swap-count-as-magnitude comparisons across families.

---

## 10. Prior art the project must cite

- **Monkam, De Lucia & Bastian**, *Computers & Security* 144 (2024) 103929 — the source pipeline.
- **Cohen-Steiner, Edelsbrunner & Harer**, "Stability of Persistence Diagrams," SoCG '05, Pisa,
  pp. 263–271. (The project folder holds the 2005 conference version, not the 2007 DCG version — cite
  the one actually read.)
- **Ferrara**, *AIMS Mathematics* 10(7):15457–15475 (2025). Position as a **complementary axis**
  (which points to attack) vs. this work (which perturbation algebra registers). **Never claim to test
  Ferrara's Theorem 2** — its hypotheses (linear classifiers, Vietoris–Rips, Frobenius-ball
  perturbations) all fail here.
- **Zheng, Zhang, Wagner, Goswami & Chen**, "Topological Detection of Trojaned Neural Networks,"
  NeurIPS 2021 — topology detects model-level tampering, a complementary target to poisoned data.
- **GENEO** (Frosini, Bergomi, Quercioli) — exists precisely because persistence is "too invariant."
  **Omitting it is the main way to look uninformed** when using the phrase "invariance structure,"
  which is not standard terminology.
- **Garin & Tauzin**, "A topological reading lesson: Classification of MNIST using TDA"
  (arXiv:1910.08345) — the direct methodological ancestor of the binarize→filtration→cubical pattern,
  cited by giotto-tda's own filtration classes.
- **Sethi & Kantardzic** (2018) — the attack-strength-vs-detectability dilemma; Claim 2 is a
  topological instance. Note theirs is a *diversity*-based framing; arXiv:1802.07295 is the closer
  impact-vs-detectability match.

No paper analyzes the limitations of the Monkam pipeline. This appears to be the first.
