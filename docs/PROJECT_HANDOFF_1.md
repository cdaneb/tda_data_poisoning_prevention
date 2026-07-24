# Project Handoff — TDA Poisoning Detection (v2, post-pivot)

*Supersedes the original four-lens handoff, which is now largely obsolete: the project pivoted away from
"can TVI / game theory / iteration improve on Monkam" to a question about **what a TDA detector can and
cannot see**. Upload this alongside the repo's live `CLAUDE.md` and `cc_summary.md` (primary sources for
codebase facts), `TERMINOLOGY.md`, and `poster_blocks.tex`.*

---

## 1. Immediate context — read this first

**MAA MathFest 2026: August 5–8, Boston. Abstract already submitted under the OLD framing.** Roughly two
weeks out as of this writing. Everything below is oriented toward a poster, not toward finishing the
original research plan. Do not start work that cannot land before the conference.

Author: Christian Dane Beels, advised by Dr. Joseph Dorta, Dept. of Mathematical Sciences, USMA West Point.
*Note: Monkam and Bastian (the source paper's authors) are at USMA's Army Cyber Institute — same
institution. Keep all characterizations of their work collegial and precise.*

---

## 2. What the project is now

**Original question (superseded):** can iteration + Ferrara's TVI / game-theoretic defense improve poison
detection over Monkam et al. (2024)?

**Why it failed:** the founding hypothesis was that the project's low capture rate (single digits vs.
Monkam's reported 40–70%) was caused by unrealistically weak synthetic poisoning. A realistic
surrogate-guided attack was built to test this. **No configuration approached 40–70%** — the gap is
structural, not attack-realism. Separately, Ferrara's minimax framework was found to degenerate to ordinary
optimization under a static attacker, and its defense mechanism $D_\alpha$ has no instantiation in an
unsupervised, non-gradient-trained pipeline.

**Current question:**

> What is it about the interaction between an attack's invariance structure and a filtration's
> sensitivities that determines whether poisoning is topologically detectable?

**Working title:** *What Can a Persistence Diagram See? Invariance Structure and the Detectability of Data
Poisoning.*

---

## 3. The claims, with exact scope

### Claim 1 (PRIMARY) — deductive, proof-backed
A swap attack applies $\sigma \in S_{1500}$ to the byte vector. Binarization $b$ is pointwise, so it is
**equivariant**: $b(x \circ \sigma) = b(x) \circ \sigma$. Summation is symmetric, so the foreground count
$\sum_i b(x_i)$ is **invariant**. Therefore count-dependent features are blind to permutation attacks, and
all detection must flow through the **position-dependent** (Height/Radial) filtrations.

- Confirmed empirically: 0/200 count change under swaps at threshold 0.4 **and** 0.3; 0/200 across four
  distinct permutation families.
- **Scope caveat:** requires a fixed pointwise threshold. giotto-tda's `Binarizer(threshold=0.4)` uses a
  *fraction of the fitted `max_value_`* — data-dependent. The proof survives because a maximum is itself
  permutation-invariant, **but this extra step must be stated explicitly.**
- **Never claim invisibility.** The highest-capture condition in the entire study (7.92%) was swap-only.
  Correct phrasing: *attenuated on the foreground-count channel, not invisible.*

### Claim 2 (SECONDARY) — empirical, robust
Surrogate guidance (optimizing swaps against a classifier's loss, no topological term in the objective)
**increased** topological detectability: **+4.68 ± 1.57, positive in all 5 seeds**. Attacker success and
defender success are positively coupled here. Replicated at threshold 0.3 (+5.24 ± 2.35, 5/5). **Not yet
replicated on a second dataset** (Test C pending).

### Claim 3 (SECONDARY) — empirical
Detectability is not ordered by attack realism. The perturbation *family* used by functionality-preserving
attacks is the quietest tested; guidance more than compensates. Capture tracks **spatial disruption**, not
sophistication.

### An open, unconfirmed extension (potentially the strongest result)
Permutations that only exchange bytes **on the same side of the binarization threshold** leave the binarized
image *bit-identical*, hence all 60 features identical, hence capture provably **exactly 0%**. These form a
subgroup $S_{|B_0|} \times S_{|B_1|} \le S_{1500}$. Test B's block-reversal and block-swap families both
returned **0.00% ± 0.00%** (zero variance across 5 seeds — the signature of a deterministic outcome), which
is consistent with accidental membership in this subgroup but **has not been verified**. See §6.

---

## 4. Results (all OPTICS; other algorithms ≈0 unless noted)

### Main factorial — 5 seeds [42,123,456,789,1024], UNSW-NB15, threshold 0.4, malicious targeting
| Cell | Noise | Swaps | Capture % |
|---|---|---|---|
| N | on | — | 4.96 ± 1.64 |
| S | off | 60 random | 1.80 ± 0.51 |
| SG | off | 60 guided (MLP) | 6.48 ± 1.24 |
| SG-RF | off | 60 guided (RF) | 7.92 ± 3.86 |
| NS | on | 60 random | 3.52 ± 1.79 |
| NSG | on | 60 guided | 4.88 ± 1.52 |
| L_unmatched | on | ~10–50 random | 2.56 ± 2.21 *(outside ladder — confounds targeting)* |

### Main effects
| Effect | Comparison | Δ | Seeds | Verdict |
|---|---|---|---|---|
| Guidance (no noise) | S→SG | +4.68 | 5/5 | **robust** |
| Noise (no guidance) | S→NS | +1.72 | 4/5 | modest |
| Noise (with guidance) | SG→NSG | −1.60 | 4/5 | modest, **sign flip** |
| Swaps (with noise) | N→NS | −1.44 | 4/5 | modest |
| Guidance (with noise) | NS→NSG | +1.36 | 3/5 | inconclusive |
| Surrogate (MLP vs RF) | SG→SG-RF | +1.44 | 3/5 | inconclusive |

**Interaction:** noise *helps* detection when unguided, *hurts* it when guided. Guidance appears to create a
coherent topological signature that added noise scrambles.

### Test A — threshold robustness (COMPLETE)
| | 0.4 | 0.3 |
|---|---|---|
| S | 1.80 ± 0.51 | 1.24 ± 1.02 |
| SG | 6.48 ± 1.24 | 6.48 ± 2.44 |
| Guidance effect | +4.68 ± 1.57 (5/5) | +5.24 ± 2.35 (5/5) |

Count-invariance 0/200 at both. Noise |Δ| = 36.1 at 0.3 (clean mean foreground count **190.6 / 1500**).
**Verdict: threshold-independent.** The source paper's own 0.4-vs-0.3 ambiguity is not load-bearing. Use 0.4
as primary (variance roughly doubles at 0.3).

### Test B — permutation families (COMPLETE; threshold 0.4, no noise, no guidance, 5 seeds)
| Family | Count-invariant | Mean positions value-changed | Capture % |
|---|---|---|---|
| Transpositions (60) | 0/200 | 21.96 ± 30.4 | 1.80 ± 0.51 |
| Block reversal (k=120) | 0/200 | 16.42 ± 39.5 | **0.00 ± 0.00** |
| Block swap (2×k=60) | 0/200 | 14.70 ± 37.2 | **0.00 ± 0.00** |
| Cyclic shift | 0/200 | 266.6 ± 375.6 | 6.28 ± 1.31 |

**This is the best poster figure**: one identical proof across four families, capture spanning 0%–6.3%.
Note cyclic shift (6.28, unguided) ≈ SG (6.48, guided) — guidance is not special, it is an *efficient route
to spatial disruption*; brute-force disruption reaches the same place.

### Test C — CICIDS2017 replication (PENDING at handoff)
Surrogates retrained on CICIDS2017: MLP 97.7%, RF 99.4%, majority baseline 74.16% (valid).
Partial: S seed42=1.40, seed123=3.40; SG seed42=1.00 → **guidance effect at seed 42 = −0.40, negative**
(opposite sign from UNSW). Far too little data to call. **Also notable:** seed-42 SG produced HDBSCAN =
5.0%, the first nonzero HDBSCAN anywhere in the investigation — CICIDS may have different cluster geometry.

### Step 0 mechanism + swap sweep
Step 0 (threshold 0.4): swaps 0/200 count change; noise (σ=30) mean +10.6, mean |Δ| = 15.8.
Swap-count sweep (single seed, no noise/guidance): 10→0.0%, 20→1.0%, 30→1.8%, 60→2.2%, 100→4.2% — monotone.

---

## 5. Repo state

**Committed to GitHub:** `adversarial_attack.py`, `run_lens4_baseline.py`,
`models/surrogate_mlp.joblib`, `models/surrogate_rf.joblib`, updated `CLAUDE.md` / `cc_summary.md`.
CICIDS surrogates saved as `surrogate_mlp_cicids.joblib` / `surrogate_rf_cicids.joblib`.

**Gitignored:** `results/*.json` ("regenerable"), `*.pdf`, `CLAUDE.md`, **and all instruction `.md` files**
— so the lens-instruction reasoning lives only in chat history and in this document. Consider tracking at
least this handoff.

**Phase 2 build gates (all passed):** bit-for-bit equivalence to legacy `poison.py`; MLP 0.9690 / RF 0.9850
vs. 0.738 majority baseline; measured latency 0.215 ms/call; grid fixed empirically at n_swaps=60,
population=50, generations=100, early-stop at benign_proba ≥ 0.5; guided vs. compute-matched random =
98% vs. 10% flip rate. Full run: 500 targets, 257.5 s, **100% validity**, 83.2% surrogate flip rate.

**Pipeline mechanics** (verified against giotto-tda docs/source): `CubicalPersistence` with **GUDHI** C++
backend, `coeff=2`, `homology_dimensions=(0,1)`, sublevel-set filtration on a cubical complex where pixels
are top-dimensional cells. `Binarizer` threshold is a *fraction of fitted `max_value_`*. Feature count
5 filtrations × 6 transformers × 2 homology dims = **60**. See `TERMINOLOGY.md` §5 for detail.

---

## 6. Open items, in priority order

1. **Bit-identity check on block reversal / block swap** *(highest value, ~1 hour).* Assert **exact** array
   equality of (a) binarized images and (b) 60-dim feature vectors, clean vs. poisoned. Also report what
   fraction of the ~15 changed byte positions actually **crossed** the threshold. If bit-identical, the two
   0.00% cells become a *proven invisibility* result rather than "very quiet."
2. **Effective swap fraction** *(one line).* Test A gives clean foreground count 190.6/1500, so
   $p \approx 0.127$ and only $2p(1-p) \approx 22\%$ of random transpositions can change anything —
   ~13 effective swaps out of 60. Measure it directly; it explains the entire Test B ranking and the
   quietness of random swaps.
3. **Test C completion** — report whichever way it lands. If mixed-sign, the honest headline is "replicates
   across configuration, not yet across datasets." Pre-committed: no re-tuning to rescue it.
4. **Verify the 60-vs-72 arithmetic** against `tda_pipeline.py` (see §7).
5. Poster build-out: figures, then Discussion/Conclusion sections.

---

## 7. Corrections and traps (hard-won — do not re-litigate)

- **Equivariance ≠ invariance.** $b(x\circ\sigma) = b(x)\circ\sigma$ is *equivariance*; the *count* is what's
  invariant. Both steps needed.
- **Two senses of "permutation invariance."** Standard TDA usage = persistence diagrams are multisets
  (unordered points). Yours = permuting source pixels. Distinguish explicitly.
- **60 vs 72 features may be the paper's error, not yours.** Monkam's Algorithm 1 as printed yields
  5 filtrations × 6 transformers × 2 dims = **60** — exactly what the project produces. Their claimed 72
  would need six filtrations. Verify, then treat as a second internal inconsistency alongside the binarizer
  0.4/0.3 issue.
- **The attribution ladder was originally confounded** — noise was not a controlled variable, which voided
  the early L→R60 comparison. Fixed by the controlled factorial. Don't reintroduce swap-count-as-magnitude
  comparisons across attack families.
- **Seed-42 artifacts have burned this project twice**: the "G60-RF is 2× G60-MLP" claim (sign-unstable
  multi-seed, range −5.8 to +5.8) and the "legacy = 6.6%" figure (multi-seed 2.56 ± 2.21). **Never put a
  single-seed number on a poster.**
- **Distinguish attacker-side from detector-side metrics.** 83.2% surrogate flip rate ≠ detection result.
- **Two persistence computations exist in the repo.** `CubicalPersistence` on packet images (the feature
  extractor — what the proof concerns) vs. `VietorisRipsPersistence` on the 60-dim residual point cloud
  (Lens 2, descriptive convergence tracking only). The proof says nothing about the latter.

---

## 8. Novelty position (from a full prior-art search)

All three claims appear **novel in the security/TDA-detection literature** and clear the MathFest bar
comfortably. Framing that survives scrutiny:

- Claim 1 is **mathematically elementary but previously unstated in this literature**. Say so. The
  invariance of persistence diagrams under domain homeomorphisms is folklore; the **GENEO** program
  (Frosini, Bergomi, Quercioli — arXiv:2606.21084, arXiv:2010.08823) exists precisely because persistence
  is "too invariant." **Cite it** — omitting it is the main way to look uninformed. Your contribution is
  that this known invariance implies a concrete attack blind spot in *this security pipeline*.
- Claim 2 is a topological instance of the known **attack-strength-vs-detectability dilemma** (Sethi &
  Kantardzic 2018, *Neurocomputing* 289:129–143 — note theirs is a *diversity*-based framing;
  arXiv:1802.07295 is the closer impact-vs-detectability match). Novel for a topological detector.
- **No paper analyzes the limitations of the Monkam pipeline** — this is apparently the first.
- Ferrara's **Theorem 2** claims optimal attacks raise classification error *and* topological distortion
  together. Claim 2 is plausibly the **first empirical test** of that coupling. Position as complementary
  axes (Ferrara: *which points* to attack; this work: *which perturbation algebra* registers) — **never**
  claim to test Theorem 2, whose hypotheses (linear classifiers, Vietoris–Rips, Frobenius-ball
  perturbations) all fail here.
- **Also cite** Garin & Tauzin, "A topological reading lesson: Classification of MNIST using TDA"
  (arXiv:1910.08345) — the direct methodological ancestor of Monkam's binarize→filtration→cubical pattern,
  referenced by giotto-tda's own filtration classes.

---

## 9. Working conventions (house style — maintain)

- **Phased instructions delivered as markdown files** (or paste blocks) with numbered steps, explicit
  **stop conditions**, and completion reports structured around those steps. No silent scope creep.
- **Read-only investigation before any build.** Build phases use **blocking empirical gates** — each must
  pass with reported numbers before the next step runs.
- **Findings sanity-checked empirically before being trusted**, even when the reasoning sounds sound.
- **Deviations reported honestly, never rounded up.** "Recommended but not confirmed run" is a finding.
- **Pre-commit to reporting results whichever way they land**, before running.
- `CLAUDE.md` / `cc_summary.md` updated **only when a lens/phase actually lands**, never from an
  investigation.

---

## 10. Dead or deferred

- **Lens 1 (per-point topological influence).** Stuck. Two sigma designs empirically ruled out; two
  candidates (standardize-then-isotropic per Ferrara Eq 3.3, and per-dimension) never tested. Note Eq 3.3
  uses **isotropic** noise on **standardized** data — the project's per-dimension hunt was reconstructing
  per-cluster what Ferrara does once globally.
- **Lens 3 (game-theoretic defense).** Never built. Minimax degenerates to plain optimization under a static
  attacker; Ferrara's $D_\alpha$ has no instantiation here. Shippable version is just threshold relaxation.
- **Iteration.** Already implemented in the repo; **never tested against the new attack substrate.**
- **TVI** never computed. Machinery exists (`compute_whole_residual_diagram` + real `PairwiseDistance`
  Wasserstein) — computing $W_p$ per attack family would quantify distortion in Ferrara's own currency, and
  remains the cheapest high-value addition.
- **Hore / Deep PackGen** deferred (raw-PCAP representation gap, no public code).
- 5-seed Yellow/Pink cluster-size probe never run.
