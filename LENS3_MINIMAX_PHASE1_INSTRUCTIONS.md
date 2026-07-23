# LENS3_MINIMAX — Phase 1 Feasibility Investigation (Game-Theoretic Defense)

**Lens 3 concept:** Ferrara (2025) minimax defense (Eq 4.3, Algorithm 3). Bring a game-theoretic
threshold/defense-selection process to the project's poison-detection step, in place of the flat
**100%-purity** rule hardcoded in `classify_clusters()`.

**Read this first — do not pre-assume what "the defender's move" is.** In Ferrara, the minimax objective is
`D_{α*} = arg max_α min_β u(D_α, Δ_β)` (Eq 4.3), with utility `u(D_α, Δ_β) = −W_p(PD_p(X), PD_p(X ⊕ Δ_β ⊕ D_α))`.
That is: the defender's action `D_α` is a **defense *mechanism*** (a data/feature transformation — Theorem 3
instantiates it as *topological regularization*), the attacker's action `Δ_β` is a poisoning strategy, and
the payoff is **topological distortion** (Wasserstein distance between the clean and attacked-then-defended
persistence diagrams). It is **not**, on its face, a purity threshold, and the payoff is **not** poison-capture
rate. The project's natural instinct — "the defender picks the Green/Red purity threshold" — is a *different*
object from Ferrara's `D_α`, and mapping Eq 4.3 onto it without checking is exactly the citation-faithfulness
error that forced the Lens 2 rebuild (a result must be cited only where the code implements what the result
is actually about). **So the mapping itself is the first thing Phase 1 must settle, not assume.**

Two candidate readings, both to be weighed in Phase 1 (see step 2):
- **Reading T (threshold selection):** defender's action = the purity threshold pair in `classify_clusters()`;
  payoff = the capture-vs-purity tradeoff. Borrows the minimax *vocabulary* but changes the objective from
  Ferrara's topological distortion to the project's poison capture. Closest to cc_summary §10.A; least
  faithful to Eq 4.3.
- **Reading D (defense mechanism, Ferrara-faithful):** defender's action = a data/feature transformation
  `D_α`; payoff = `−W_p` topological distortion — which the project **already computes** via Lens 2's
  `wasserstein_distance_between_diagrams(...)`. Most faithful to Eq 4.3/Alg 3, but a larger build and further
  from the project's "threshold relaxation" plan.

**Why Lens 3 is still a good pivot target:** either reading has **no dependency on Lens 1**. Both need only
(a) Lens 2's real Wasserstein distance — already in place — and (b) the existing `classify_clusters()`
machinery. Reading T is the more direct lever on the headline metric (capture rate, capped by the
100%-purity rule, Known Limitation #2); Reading D is the more direct instantiation of the cited theory.

**This file is Phase 1 only: a read-only feasibility investigation.** No repository code is modified. The
goal is to settle which reading (if either) Lens 3 should pursue, whether a minimax solution is even
non-trivial yet, and what Phase 2 would need to build — then **stop** for an explicit go-ahead.

---

## Essential prior context to reconcile before starting

Re-read the live `CLAUDE.md` / `cc_summary.md`, then note these two things, because Lens 3 sits directly on
top of them:

1. **cc_summary §10.A ("Threshold Relaxation", flagged highest priority)** already proposes the *exact*
   parameterization **Reading T** would need: `classify_clusters(cluster_labels, is_poisoned,
   green_threshold=0.0, red_threshold=1.0)` with Green iff `poison_fraction <= green_threshold`, Red iff
   `poison_fraction >= red_threshold`, then sweeping `red_threshold` 1.0→0.5 and `green_threshold` 0.0→0.5.
   Be precise about what this is: that brute-force sweep is a **plain optimization over a threshold**, and
   under Reading T with a *static* attacker it is *all* the "minimax" reduces to (see step 3). It is a
   useful baseline, but calling it game-theoretic would overstate it.

2. **A marked Lens-3 hook already exists in `iterative_filter.py`:** Lens 2's
   `wasserstein_distance_between_diagrams(...)` value is currently *descriptive-only* (logged/plotted, not
   read by the stopping condition or `classify_clusters()`), explicitly annotated as a possible Lens 3 use.
   Note that this quantity **is** Ferrara's payoff object — the `W_p` in `u(D_α, Δ_β)` (Eq 4.3) and the
   detection statistic in Eq 5.1 (`d_poison = W_p(PD_p(X_ref), PD_p(X_test))`). That makes it the natural
   payoff for **Reading D** and is a strong reason to take Reading D seriously rather than defaulting to the
   threshold reading. Phase 1 should determine whether the minimax payoff should be this topological-distortion
   distance (Ferrara-faithful) or the project's purity/capture tradeoff (Reading T's re-purposing).

---

## Hard constraints (house style)

- **No repository code may be modified in Phase 1.** Locate, read, quote, and reason — do not edit
  `clustering.py`, `iterative_filter.py`, or anything else. Any parameterization is *proposed* in the
  report, not implemented.
- Environment: `cd C:\TDA; .\venv312\Scripts\Activate.ps1`. Reference regime for any cost estimates:
  **seed 42 / `MAX_SAMPLES=5000` / `POISON_RATE=0.10`**, OPTICS (the only algorithm that reliably produces
  Red clusters — DBSCAN/HDBSCAN/MeanShift are ~0% capture and are not the point of this lens).
- Report deviations and inconvenient conclusions honestly — in particular the static-attacker tension in
  step 3, which is the single most important thing this investigation has to get right.

---

## Numbered steps

### 1. Locate and document the current threshold mechanism
- Open `clustering.py`, find `classify_clusters(...)`, and quote its actual color-assignment lines
  (`poison_fraction == 0` → Green, `== 1.0` → Red, `> 0.80` → Pink, else Yellow) and the `def` signature
  verbatim. Confirm these are still as documented; report any drift.
- Trace how Green/Red masks flow into `iterative_filter.py` (which unions Green→sanitized, Red→poisoned,
  and passes Yellow/Pink/Noise to the next iteration). Confirm the thresholds are the *only* place the
  100%-purity rule lives — i.e. that relaxing them is genuinely the single lever.

### 2. Decide which reading Ferrara's Eq 4.3 / Algorithm 3 actually licenses (do not pre-commit)
- Read **Eq 4.3**, **Algorithm 3**, and the surrounding §4 definitions (`S_D`, `S_A`, `u(D_α, Δ_β)`) directly
  from the Ferrara PDF (now available — confirm it is checked into the repo per the kickoff file; if it
  somehow is not, stop and flag rather than guessing the formalism). Note the exact objects: `D_α` is a
  defense *mechanism*, `Δ_β` a poisoning strategy, `u = −W_p(PD_p(X), PD_p(X ⊕ Δ_β ⊕ D_α))` a topological
  distortion. Quote `u` and Eq 4.3 verbatim in the report.
- **Now evaluate the two readings against that definition, side by side — this is the core deliverable:**
  - **Reading D (Ferrara-faithful):** defender's action `D_α` = a data/feature transformation of the
    residual (Theorem 3's topological regularization is the canonical instance); payoff = `−W_p`, i.e. Lens
    2's `wasserstein_distance_between_diagrams(...)` between the clean-reference and the attacked-then-defended
    residual diagram. Assess: what would a concrete `D_α` be for this pipeline (a feature-space regularizer? a
    residual denoiser?), and does the payoff line up 1:1 with Eq 4.3? This reading *implements the cited
    result*; say how closely.
  - **Reading T (threshold selection):** defender's action = `(green_threshold, red_threshold) ∈ [0,1]²`;
    payoff = the capture-vs-purity tradeoff (`capture_pct` up, `sanitized_purity`/`poisoned_pool_precision`
    down as thresholds relax). Assess honestly: this reuses the *minimax vocabulary* but swaps Ferrara's
    payoff (topological distortion) for a different one (poison capture), and the defender's action is a
    downstream classification decision, not a `D_α` transformation. State plainly how far this departs from
    Eq 4.3 — it may be the more *useful* lever for the headline metric while being the less *faithful* one.
- **Attacker's action (both readings):** the poisoning strategy `Δ_β`. **Currently fixed** (Gaussian noise +
  byte swaps in `poison.py`); making it adaptive is **Lens 4's** job (see step 3 for why this is decisive).
- For whichever reading(s) you judge viable, write down the concrete minimax objective and the cost of a
  single evaluation: for Reading T, one parameterized `classify_clusters` + one `iterative_filter` run per
  threshold setting; for Reading D, one `D_α` application + one whole-residual diagram + one `W_p` per
  (α, β) pair.

### 3. Assess whether the game is non-trivial yet (the load-bearing question)
State this plainly, because it determines whether Lens 3 can stand alone — **and it applies to both readings**:
- The inner `min_β` in Eq 4.3 ranges over the attacker's strategies `Δ_β`. With a **static** attacker
  (today's `poison.py`), the attacker has no move — `min_β` is over a single point — so the minimax
  **degenerates to ordinary optimization over the defender's action alone**. Under Reading T that is "pick
  the best threshold on the §10.A tradeoff curve"; under Reading D it is "pick the `D_α` that best preserves
  topology against the one fixed attack." Either way it is **not a game**, and calling the result
  game-theoretic would overstate it.
- The minimax framing only becomes substantive once an **adaptive attacker** (Lens 4: poisoning that targets
  topologically vulnerable / high-`TI` points near the decision boundary — Ferrara Theorem 2 and the four
  Appendix-A.4 strategies: random / boundary / topological / combined) can *respond* to the defender. So
  Lens 3's "game" is **coupled to Lens 4** regardless of reading.
- Report a clear recommendation on sequencing: can Lens 3 ship a meaningful standalone deliverable now — the
  principled *selection procedure* plus the degenerate-attacker instance (§10.A tradeoff curve for Reading T,
  or a single-attacker distortion-minimizing `D_α` for Reading D) — deferring the true minimax until Lens 4
  supplies an attacker to minimax against? Or should Lens 3 wait on Lens 4? Give your reasoned answer — do
  not hedge it away.

### 4. Feasibility & cost of a Phase-2 implementation
Enumerate what Phase 2 would need for the reading(s) you judged viable in step 2, and which pieces already
exist vs. must be built.
- **Reading T:** parameterized `classify_clusters(..., green_threshold, red_threshold)` — **to build** (per
  §10.A); a threshold-sweep + iterative re-run harness — **to build** (thin driver; reuse `run_iterative` /
  `run_multi_seed` scaffolding); a payoff over `(capture, purity, precision)` — **to define**.
- **Reading D:** a concrete `D_α` transformation family — **to design**; the payoff already exists (Lens 2's
  `wasserstein_distance_between_diagrams`); an (α, β) evaluation loop — **to build**.
- **Both:** a parameterized attacker for a real minimax — **not available (Lens 4)**; note the
  degenerate-attacker fallback for now.
- **Compute estimate:** using Section 8 timings (~200 s per OPTICS iterative run on this machine, ~12.5 s per
  whole-residual `VietorisRipsPersistence` diagram on a real ~5.5k residual), estimate the wall-clock for a
  representative grid × seeds — e.g. Reading T: `red_threshold ∈ {1.0,0.9,0.8,0.7,0.6,0.5}` ×
  `green_threshold ∈ {0.0,0.1,0.2}` × 5 seeds — and flag whether this reopens the **local-vs-remote (VM)**
  compute question for the user.

### 5. Draft the Phase-2 proposal (as a proposal, not an implementation)
Sketch — in the report only — the Phase-2 plan following house style: what gets built, in what order, with
its own investigate→implement stop conditions. Make a recommendation between (and possibly across) the two
readings, weighing **faithfulness to Eq 4.3 (favors Reading D)** against **directness on the headline capture
metric and reuse of §10.A (favors Reading T)**. If recommending Reading T, state explicitly that the §10.A
threshold-relaxation sweep should ship **as an honest, non-game-theoretic baseline** first — labelled as
such, not as a minimax result — with any game-theoretic layer deferred until Lens 4 supplies an adaptive
attacker.

---

## Stop condition & completion report

After step 5, **stop**. No repository code is changed; no threshold is swept; no `classify_clusters`
parameterization is written; no `D_α` is implemented. Produce a report structured around the five numbered
steps, with `clustering.py` quoted verbatim in step 1 and Ferrara's `u(D_α, Δ_β)` and Eq 4.3 quoted verbatim
in step 2.

Close the report with two unambiguous recommendations to the user:

> **(a) Which reading?** Reading D (Ferrara-faithful, payoff = Lens 2's `W_p`), Reading T (threshold
> relaxation, more direct on capture but a re-purposing of the cited result), or a sequenced combination —
> with the faithfulness-vs-usefulness tradeoff stated plainly rather than split down the middle.
>
> **(b) Independent or sequenced with Lens 4?** Whether Lens 3 can ship a standalone degenerate-attacker
> deliverable now, or must wait for Lens 4's adaptive attacker for the minimax to be non-vacuous — and, if
> coupled, exactly what Lens 4 must supply first.

Ferrara (2025) is now available (both source papers were provided); the kickoff file's step should confirm
it is checked into the repo so step 2 reads the real formalism rather than a summary. Do not update
`CLAUDE.md` / `cc_summary.md` from a Phase-1 investigation; docs are updated only when a lens actually lands.
