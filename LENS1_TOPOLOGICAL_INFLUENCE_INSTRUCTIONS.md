# Lens 1 — Per-Point Topological Influence for Partial Yellow/Pink Resolution

## Purpose

Implement Ferrara's per-point topological influence, $TI(x_i)$ (Eq 3.3), to
partially resolve the Yellow and Pink residual — currently the majority of
the data, discarded with zero partial credit under `classify_clusters()`'s
binary purity rule — using the whole-residual-diagram machinery Lens 2 just
built.

This is the most novel and least empirically certain of the four lenses.
Ferrara's Theorem 2 — the claim motivating this whole lens, that poisoning
attacks concentrate on high-TI points — has a thin proof sketch in the
source paper, more assertion than derivation. So unlike Lens 2, this is
staged in **three phases**: investigate feasibility → diagnostic validation
(does TI actually separate poisoned from clean points on this project's
real data) → decision-rule implementation, and the third phase only happens
if the second one confirms the premise is worth acting on.

**Do not proceed to the next phase without my explicit go-ahead**, after
reviewing your report at the end of each phase.

---

## Background — reusing Lens 2's infrastructure, not rebuilding it

$$TI(x_i) = \mathbb{E}_\delta\left[W_p\left(PD_p(X), PD_p(X \setminus \{x_i\} \cup \{x_i + \delta\})\right)\right]$$

— the expected persistence-diagram distance caused by perturbing a single
point by small random noise. This builds directly on what Lens 2 already
produced: `compute_whole_residual_diagram()` generalizes to any point
cloud (it doesn't have to be the whole residual — a single cluster's
points work identically), and `wasserstein_distance_between_diagrams()`
is already the real distance between two such diagrams. For a given
cluster's point set $X$: compute one baseline diagram; for each point
$x_i$, draw several small Gaussian perturbations, recompute the diagram on
the perturbed point set, and average the resulting distances to baseline.

Do not modify `compute_whole_residual_diagram()` or
`wasserstein_distance_between_diagrams()` — Lens 1 calls them as-is.

---

## Phase 1 — Investigate feasibility (read-only, no code changes)

Confirm the following before any code is written:

1. **What's currently logged per point/per cluster.** Does
   `iterative_filter.py`'s current logging retain, per iteration, which
   specific points ended up in which Yellow/Pink cluster, along with their
   ground-truth poison labels (from `poison_dataset()`)? Or only aggregate
   per-cluster summary stats (`dpdc`, sizes)? This determines whether the
   Phase 2 diagnostic can be built from existing saved results or needs
   fresh instrumentation added to the loop.
2. **Real Yellow/Pink cluster sizes.** From existing completed runs (or a
   minimal fresh instrumented pass if item 1 shows this isn't currently
   retained), report the distribution of Yellow and Pink cluster sizes
   across iterations, for both datasets, across all four algorithms —
   min/median/max, and total points affected across a full run. This is
   the key input for judging TI's compute cost.
3. **Empirical TI cost at representative scale.** Time *one point's* TI
   computation (5 noise draws, a proposed default `sigma` — see item 4) at
   a representative Yellow-cluster size found in item 2. Extrapolate: what
   would the added cost be for (a) one full `run_iterative.py` run, and
   (b) a full multi-seed sweep, if TI were computed for every point in
   every Yellow/Pink cluster at every iteration? Report this explicitly —
   don't assume it's cheap just because individual clusters are smaller
   than the whole residual Lens 2 timed.
4. **Sigma selection.** Propose a default perturbation scale proportional
   to the natural spread of the TDA feature vectors *within each cluster*
   (e.g. a small fraction of the per-cluster per-dimension standard
   deviation), rather than one fixed absolute constant across all clusters
   — the 60-dim feature space doesn't have an otherwise-established scale.
   Report what you propose and why. Do not implement it yet.

**Report back** on items 1–4 and stop. If item 3's extrapolated cost looks
impractical at full scale, propose (but do not apply) mitigations — e.g. a
cap on cluster size before subsampling, fewer draws, or restricting Phase 2
to a subset of iterations/algorithms/datasets — and let me decide before
proceeding.

---

## Phase 2 — Diagnostic validation (implement TI computation and logging only — no decision-rule changes)

Only after I've reviewed and approved Phase 1's findings.

**A.** Add `compute_topological_influence(cluster_points, sigma, n_draws=5, seed=None)`,
reusing `compute_whole_residual_diagram()` and
`wasserstein_distance_between_diagrams()` exactly as described above.
Homology dimensions should match Lens 2's convention `(0, 1)` unless Phase 1
surfaced a reason to differ.

**B.** Wire this into `iterative_filter.py`'s main loop: for every Yellow
and Pink cluster identified at each iteration, compute TI for every point
(or a subsample, per whatever Phase 1 concluded is tractable — state
clearly which it is). Log each point's TI score alongside its ground-truth
poison label and which cluster/iteration/algorithm/dataset it came from.
**Do not change which pool any point is assigned to in this phase** —
Yellow/Pink points still flow to the residual exactly as they do today.
This is purely additive logging; the existing classification and stopping
behavior must be provably unchanged (confirm via a diff of the existing
poison-capture/purity numbers before vs. after this change — they should
be identical, since nothing about the actual decision logic moved).

**C.** Produce a correlation/separability analysis, with Yellow and Pink
reported *separately* (they start from very different poison-fraction
priors and may behave differently): for each dataset and algorithm where
enough Yellow/Pink data exists to be meaningful, report at minimum —
mean/median TI for truly-poisoned vs. truly-clean points, and a simple
separability measure (e.g. AUC treating TI as a classifier score against
the ground-truth label).

**D.** Save this analysis to a new results file (e.g.
`results/ti_diagnostic_<dataset>.json`) and a simple plot if
straightforward (TI distribution split by true label, per dataset). Run
`py_compile` on every file touched.

**Stop here after reporting the diagnostic results.** Do not proceed to
Phase 3 — the actual decision-rule/reclassification mechanism — until I've
reviewed whether the correlation found in C is strong enough to be worth
acting on. It's entirely possible the answer is "no" — that's a legitimate
and useful finding, not a failure to fix.

---

## Phase 3 — Decision-rule implementation (contingent; do not start without explicit go-ahead)

This section is intentionally left provisional. I will fully specify it
based on what Phase 2 finds — the right threshold shape, or whether this
is worth doing at all, depends entirely on the diagnostic results. The
general shape, if Phase 2 supports proceeding:

- Within Yellow/Pink clusters, use a percentile-based threshold on TI
  (e.g. top-X% by TI → move to poisoned pool, bottom-Y% → move to
  sanitized pool, middle band stays in the residual for the next
  iteration) — clearly marked in code as a provisional rule, not a
  claimed-optimal one.
- Mark (comment only, no implementation) that adaptive/optimal threshold
  selection is deferred to Lens 3 — the same deferral pattern used for
  Lens 2's Wasserstein value becoming load-bearing.
- Track and report the effect on capture rate **and** on false-positive
  rate / sanitized-pool purity, separately. This mechanism is expected to
  trade away some of the current zero-false-positive property — that
  tradeoff must be measured and reported honestly, not minimized or
  hidden.

---

## Hard constraints

- No Lens 3 or Lens 4 work in this task.
- Do not modify `classify_clusters()`'s existing purity-based logic in
  Phase 1 or Phase 2 — Phase 2 is diagnostic-only, additive logging, with
  provably unchanged classification behavior.
- Do not touch the stopping condition.
- Do not modify `compute_whole_residual_diagram()` or
  `wasserstein_distance_between_diagrams()` — Lens 1 reuses them as-is.
- Each phase stops for review before the next begins. Do not chain phases
  together without an explicit go-ahead between them, even if a phase
  seems to finish cleanly.

## Stop conditions

- **After Phase 1:** report findings and stop.
- **After Phase 2:** report the diagnostic results (correlation/separability
  analysis, Yellow and Pink reported separately) and stop. Do not proceed
  to Phase 3 without explicit go-ahead.
- **After Phase 3 (if authorized):** full completion report — steps taken,
  `py_compile` results, and the measured capture-rate vs. false-positive-rate
  tradeoff — and stop.
