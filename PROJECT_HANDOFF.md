# Project Handoff — TDA Data Poisoning Detection (Lenses 1–4)

*Written to migrate context to a new chat when this one runs low on tokens.
Paste or upload this at the start of the new chat, along with the current
`CLAUDE.md` and `cc_summary.md` from the repo (or have the new chat read
them fresh via Claude Code — they are the primary source of truth for
codebase facts; this document covers the strategic/conversational layer
those files don't capture).*

---

## 1. What this project is

A reproduction-plus-extension of Monkam, De Lucia, & Bastian (2024,
*Computers & Security* 144, 103929) — a TDA + unsupervised-clustering
data-poisoning detector for network intrusion detection systems (NIDS).
The repo (`C:\TDA`, Python 3.12, `venv312`) reproduces that paper's
pipeline and adds an **iterative filtration loop** (the project's original
novel contribution, predating this conversation's work) that repeatedly
re-extracts TDA features from unresolved ("Yellow") residual data and
re-clusters it.

This conversation's work has been extending that further: testing whether
concepts from **Ferrara (2025, AIMS Mathematics 10(7):15457–15475)** — a
Topological Vulnerability Index (TVI), per-point topological influence,
and a game-theoretic minimax defense framing — can improve on Monkam's
binary purity-threshold clustering, organized as **four "lenses"**
(below).

**Two other papers came up but are not methodology sources — related work only:**
- Monkam, De Lucia, Bastian (2023, IEEE DSC) — the conference precursor to the 2024 journal paper. Same lineage, not separately actionable.
- Monkam, Yan, Bastian (2025 preprint) — a *different* problem: forensic detection of poisoning in an *already-trained* model (via model inversion + TDA + SHAP + Benford's Law), on CIFAR-10/CNN/ResNet/YOLOv8. Not applicable to this project's pre-training data-sanitization approach. Cite only as related work / generalizability evidence if useful.

## 2. The research question (current framing)

> Can a Topological Vulnerability Index and per-point topological
> influence — used to explain and target which points in a NIDS training
> set are most susceptible to poisoning — combined with a game-theoretic
> (minimax) defense strategy, improve poisoned-data detection and removal
> over Monkam et al.'s static purity-threshold clustering baseline,
> **including under an attacker that specifically targets topologically
> vulnerable points**?

Restated more concretely, per the most recent framing: **can incorporating
(a) iteration [already done, pre-existing] and (b) Ferrara's TVI/per-point
influence/game-theoretic concepts [Lenses 1–4] reduce the number of
unresolved Yellow clusters and improve poison capture over Monkam's
reported results?**

Important distinction: **iteration is not one of the four lenses** — it's
the project's pre-existing contribution. The four lenses are Ferrara-derived
additions on top of it.

**A live tension worth keeping in view:** the project's current
reproduction captures poison at low single-digit percentages (best case
~2.6–3.6% via OPTICS), while Monkam's original paper reports 40–70%
capture across *all four* algorithms. This gap is currently attributed to
the project's synthetic poisoning (Gaussian noise + byte swaps) being
topologically weaker than the paper's real attacks (Chale et al.
genetic-algorithm / Hore et al. deep-RL) — which is squarely **Lens 4's**
job to address, and Lens 4 hasn't started. Until that gap closes, "improves
over Monkam" is aspirational, not yet demonstrated.

## 3. Codebase quick-reference (verify against live `CLAUDE.md`/`cc_summary.md` — this may drift)

- `data_loader.py`: `load_unsw(max_samples=None)`, `load_cicids(max_samples=None)` → `(X: (N,1500) uint8, y: (N,) str)`.
- `poison.py`: `poison_dataset(X, y, poison_rate=0.10, noise_scale=30, random_state=42)` → `(X_combined, y_combined, is_poisoned)`. **Appends** poisoned samples (doesn't replace) — output is `N + n_poison` rows.
- `tda_pipeline.py`: `extract_tda_features(X)` → 60-dim features (5 filtrations × 6 extractors × 2 homology dims H0/H1). Paper claims 72; documented, understood gap.
- `clustering.py`: `run_all_clustering()` → DBSCAN/HDBSCAN/OPTICS/MeanShift labels. `classify_clusters(cluster_labels, is_poisoned)` → Green (`poison_fraction==0`), Red (`==1.0`), Pink (`>0.80,<1.0` — vanishingly rare, only 3 ever observed), Yellow (else), Noise (`label==-1`). **Zero tolerance**, no partial credit — this is the mechanism Lens 1/3 target.
- `iterative_filter.py` — **[Lens 2 rebuilt this file, done/verified]**: `compute_whole_residual_diagram(X_tda_residual)` (one `VietorisRipsPersistence` diagram over the whole residual as a point cloud) and `wasserstein_distance_between_diagrams(diag1, diag2)` (real `PairwiseDistance`, p=1). Descriptive-only — logged/plotted, not load-bearing in the stopping condition. `iterative_filter(X_raw, is_poisoned, algorithm=..., max_iterations=10)` → `sanitized_indices` / `poisoned_pool_indices` / `residual_indices` / `iteration_log`.
- `run_baseline.py` / `run_iterative.py` / `run_multi_seed.py` — the three experiment drivers. `SEEDS=[42,123,456,789,1024]`, `MAX_SAMPLES=5000`, `POISON_RATE=0.10`, `MAX_ITERATIONS=10`.
- `results_io.py` — new (post-cleanup) shared JSON serialization helper, replacing two duplicated implementations.

## 4. Four-lens framework: rationale + status

| Lens | Ferrara concept | Fix targets | Status |
|---|---|---|---|
| **1** | Per-point topological influence, $TI(x_i)$ (Eq 3.3) | Partially resolve Yellow/Pink clusters instead of discarding them whole | **Blocked** — see §5 |
| **2** | Wasserstein stability (Eq 3.1), detection formula (Eq 5.1) | Replace approximate per-sample "distance" with a real diagram-to-diagram distance | **Done, verified** — see §6 |
| **3** | Minimax defense (Eq 4.3), Algorithm 3 | Replace the flat 100%-purity threshold with a game-theoretic threshold-selection process | **Not started.** Does **not** depend on Lens 1 — only needs Lens 2's real distance + existing threshold params. Candidate for pivot, see §7 |
| **4** | Optimal poisoning strategy (Theorem 2), Appendix A.4 attack recipes | Replace weak synthetic poisoning with a topologically-motivated attack (high-TI points, optionally near a decision boundary); also where TVI itself gets computed as a standalone metric (Ferrara's Table 1 replication) | **Not started.** Likely the actual fix for the Monkam-capture-rate gap (§2) |

## 5. Lens 1 — exactly where it's stuck

**No repository code has been changed for Lens 1.** All work has been
read-only investigation via external monkeypatching of `classify_clusters`
(never touching the actual files), per a 3-phase plan (investigate →
diagnostic validation → decision-rule implementation) — currently stuck
mid-Phase-1, on the sigma question, before diagnostic validation can even
start.

**Confirmed findings:**
- Real Yellow/Pink cluster sizes (seed 42, 5000 samples, both datasets, all 4 algorithms): 790 Yellow, only 3 Pink (essentially never fires). Yellow sizes: min 3, median 14, mean 95.4, max 5019, ~75,385 total points; excluding Mean Shift, max drops to 1128, total 47,970.
- **Mean Shift is excluded from Lens 1's scope** — not primarily for compute reasons, but because it repeatedly produces a single "Yellow cluster" covering 86–91% of the whole residual (it can't produce noise/reject points), meaning there's no local structure for a local-influence measure to find. This is a methodological exclusion, worth stating as such in any writeup.
- TI cost: ~0.4s/diagram × 5 draws ≈ 2s/point, roughly independent of cluster size in the range tested (10 vs. 971 points).

**Two sigma designs tried, both failed, empirically (not just in theory):**
1. Per-cluster-anchored: $\sigma_{dim} = f \cdot \text{std}(X_{cluster}, \text{axis}=0)$, floored at $0.1 \cdot \text{std}(X_{residual})$. **Failed**: real Yellow clusters have near-zero internal variance in 55–59 of 60 dimensions (density-based clusters are, by construction, near-identical along most axes), so the floor dominates everywhere and collapses into a single global constant — reintroducing the exact "one constant across heterogeneous dims" problem per-cluster scaling was meant to avoid.
2. Whole-residual-anchored (no per-cluster component): $\sigma_{dim} = f \cdot \text{std}(X_{residual}, \text{axis}=0)$, tested at $f \in \{0.01, 0.03, 0.05, 0.1\}$. **Failed**: even the smallest tested $f=0.01$ produced perturbations ~7× the typical cluster's own median inter-point distance; $f=0.1$ was ~70× oversized. Swamps local structure rather than measuring local sensitivity.

**Two untested candidate redesigns, not yet evaluated:**
- (A) Per-dimension: use the cluster's own std where it's meaningfully non-zero (a cluster-relative epsilon test, not a hard `>0` check, to avoid floating-point noise counting as "has variance"); **zero perturbation** in genuinely degenerate dimensions — no global fallback at all.
- (B) Fit PCA on the cluster's own points, perturb along local principal directions. Deprioritized as a general default — median cluster size 14 in 60 dimensions is a bad regime for covariance estimation — but potentially worth revisiting specifically for the large clusters (max 971–1128).

**Recommended next test** (proposed, not yet run): test option (A) on the same two reference clusters already used (size-10, size-971), checking the resulting perturbation-to-local-spacing ratio — same empirical discipline that caught the first two failures.

**Unconfirmed / needs verification when picking this back up:**
- A request was made to extend the cluster-size probe from 1 seed to a full 5-seed sweep (DBSCAN/HDBSCAN/OPTICS only, both datasets) to get an honest Phase-2 cost estimate — **it's unclear whether this was ever completed and reported.** Check before assuming it's done.
- A VM/remote-compute option was offered and discussed but **never resolved** — local investigation continued in the interim, but no explicit decision was made to stay local vs. migrate. Revisit if Lens 1 (or Lens 4, if it turns out to be compute-heavy) needs it.

## 6. Lens 2 — what actually landed (for reference, since it's the one completed piece)

Replaced `compute_persistence_diagrams()` (per-sample cubical-persistence
diagrams + an approximate amplitude-difference "Wasserstein") with
`compute_whole_residual_diagram()` (one `VietorisRipsPersistence` diagram
over the whole residual as a point cloud in the 60-dim TDA feature space)
+ a rebuilt `wasserstein_distance_between_diagrams()` (real
`gtda.diagrams.PairwiseDistance`, p=1). This is now a faithful
instantiation of Ferrara's Eq 3.1/5.1 (both require whole-dataset diagrams;
the old per-sample approach wasn't actually the object those results are
about). Verified: poison capture unchanged before/after (8.2% OPTICS/UNSW)
— proof the change was purely a metric-correctness fix with zero effect on
classification behavior. Value remains descriptive-only; not wired into
the stopping condition or `classify_clusters()`'s thresholds — that's an
explicitly deferred Lens 3 decision.

**Unconfirmed: whether this was ever committed to git.** A commit message
was drafted and recommended; no explicit confirmation of `git commit`
having been run exists in the conversation record. Verify before assuming
it's safely checkpointed.

## 7. Open strategic decision, right now

Given Lens 1 is genuinely stuck (not just slow — two real designs ruled
out, no third one tested yet) and Lens 3 has **no dependency on Lens 1**
(it only needs Lens 2's real distance, already done, plus the existing
threshold parameters in `classify_clusters()`), there's a live proposal on
the table: **pivot to starting Lens 3 now**, since it's the more direct
lever on the actual headline metric (Yellow cluster count / capture rate),
and let Lens 1 resume later once/if the sigma question unblocks. This was
proposed but **not yet decided** when this handoff was written — it's the
first thing to resolve in the new chat.

## 8. Working conventions established (the "house style" — maintain this)

- **Two-or-three-phase task structure for anything touching methodology**: investigate/confirm assumptions (read-only, explicit report, stop) → implement (only after explicit go-ahead) → (for higher-uncertainty work like Lens 1) a diagnostic-validation phase between the two, before any decision logic is built on an unconfirmed premise.
- **Claude Code instructions are delivered as markdown files placed in the repo root** (e.g. `LENS2_WASSERSTEIN_INSTRUCTIONS.md`), not just inline chat text, for anything substantial — inline go-ahead messages are used for confirming/adjusting an already-drafted plan.
- **Every phase ends with an explicit stop condition** and a completion report structured around the original numbered steps — no silent scope creep into the next phase or next lens.
- **Findings get sanity-checked empirically before being trusted**, even when the reasoning sounds sound (the sigma redesign's own recommended fix was itself shown to fail when tested) — a design isn't accepted until numbers from the real data confirm it.
- **Deviations and approximations get reported honestly, not rounded up** — e.g. "resolved via deletion, not refactor" was deliberately not simplified to "refactored"; the sleep/resume timing anomaly was diagnosed via `Get-Process` rather than reported at face value.
- **`CLAUDE.md` and `cc_summary.md` are kept accurate as living docs**, updated after each lens lands (or partially, honestly reflecting blocked/in-progress status) — not just at project milestones.
- Anthropic/Ferrara citations in code comments are held to a real-applicability standard — a citation is only added where the code genuinely implements what the cited result is about (this exact issue is *why* Lens 2 needed a redesign, not just a function swap).

## 9. Immediate next actions for the new chat

1. Confirm whether Lens 2's changes were actually committed to git (§6).
2. Confirm whether the 5-seed cluster-size probe extension (§5) was completed.
3. Resolve the Lens 1 vs. Lens 3 pivot decision (§7) with the user.
4. Depending on that decision: either draft the sigma-option-(A) test instructions (Lens 1) or draft Phase 1 investigation instructions for Lens 3 (minimax threshold framing), following the house style in §8.
5. Re-read the live `CLAUDE.md` and `cc_summary.md` from the repo before writing any new instructions — this document is a snapshot at hand-off time, not a substitute for the current source.

*If finer-grained detail on any specific decision is needed beyond what's here, the new chat can use its conversation-search capability with keywords like "Lens 1 sigma," "Lens 2 Wasserstein," or "wasserstein_convergence" to pull the original exchanges — but treat that as a supplementary fallback, not the primary reference; this document is meant to stand on its own.*
