# LENS4_ADVERSARIAL_SUBSTRATE — Phase 2 BUILD Instructions

**This is a build phase — the first one in this arc.** Unlike the Phase-1 files, code *is* written here. But
the discipline is tighter, not looser: every expensive step is gated by a cheap empirical check that must
pass before the next step runs, and the whole build stays reversible and non-invasive to the existing
pipeline. This file builds the **surrogate NIDS classifier(s)** and the **Chale-approximation attack**, and
verifies them. It does **not** run the baseline-vs-Monkam comparison, compute capture rates on the full
dataset, or rewire the experiment drivers — that is **Phase 3**. Stop when the build is verified.

Carrying forward the settled decisions from the Phase-1 report: **Chale-first, Hore deferred**; primary
surrogate a small 1-D CNN (Chale's template), secondary an RF for the sensitivity probe; **relative
baseline-vs-extension framing** (Monkam's ~40–70% is a sanity neighborhood, never a reproduction target);
and the append + `is_poisoned` design is kept.

---

## Hard constraints (build phase)

- **New code goes in a new module** (e.g. `adversarial_attack.py`). **`poison.py` stays untouched** — it is
  the reference the equivalence gate (Gate A) compares against. **Do not rewire the experiment drivers**
  (`run_baseline.py` / `run_iterative.py` / `run_multi_seed.py` / `classifier_eval.py`) to the new attack —
  wiring the comparison is Phase 3's job, and leaving them alone keeps the current pipeline runnable
  throughout.
- **Attack surrogate ≠ downstream evaluator.** The RF used as the *sensitivity-probe surrogate* must be a
  separate instance/training from the RandomForest in `classifier_eval.py` that *judges sanitization
  success*. Never let Phase 3 report a `classifier_eval` number produced by a model the attack was optimized
  against — that would tune the attack against its own grader. Keep them conceptually and instance-separate
  even if the architecture is identical.
- **Determinism throughout.** Every stochastic step (sample selection, GA, surrogate init/training) is seeded
  via an explicit `random_state`, so runs are reproducible and the equivalence gate is exact.
- **Every gate is empirical and blocking.** Report the real numbers; if a gate fails, stop and report — do
  not proceed on a "should be fine."
- **UNSW-NB15 only**, **seed 42 / `MAX_SAMPLES=5000` / `POISON_RATE=0.10`**. Save trained surrogates to disk
  (e.g. a `models/` dir) so Phase 3 doesn't retrain them.
- Environment: `cd C:\TDA; .\venv312\Scripts\Activate.ps1`. Re-read the live `CLAUDE.md` / `cc_summary.md`
  first. At the end, **recommend** a git checkpoint commit of the new module + surrogate artifacts (wording
  and timing are the user's call, as with the Lens 2 commit).

---

## Build steps (each gate must pass before the next step)

### 1. Scaffold the new attack module + a `random_swap` reference mode
- Create `adversarial_attack.py` with one entry function that matches `poison_dataset`'s signature and return
  contract **exactly**: `(X, y, poison_rate=0.10, random_state=42, mode=..., ...) -> (X_combined, y_combined,
  is_poisoned)`, appending poisoned rows (never substituting in place). Add a `mode` argument.
- Implement `mode="random_swap"` to **replicate `poison.py`'s current behavior** — same random target
  selection, same `randint(10,50)` swap count, same swap mechanics, same append order.

> **Gate A — bit-for-bit equivalence (the Lens-2 discipline).** With the same seed and inputs, the new
> module's `random_swap` mode must produce **identical** `X_combined` and `is_poisoned` arrays to
> `poison.py`'s `poison_dataset`. Assert array equality directly; report pass/fail. **If it does not match,
> stop and report** — nothing downstream can be trusted until the scaffold is a faithful superset of the
> existing attack.

### 2. Build and train the surrogate(s)
- **Primary surrogate — 1-D CNN over the 1500 payload bytes** (Chale's template): benign(0)/malicious(1)
  binary, labels mapped from UNSW-NB15's category strings (`normal`→0, else→1). **Dependency note:** the
  pinned env (`requirements.txt`) has no torch/tensorflow. Adding one is a real change to a giotto-tda-pinned
  Python 3.12 env — verify compatibility in isolation *before* committing, and update `requirements.txt` if
  added. **Fallback:** if adding a deep-learning dependency is undesirable, an `sklearn` `MLPClassifier` (or
  even the RF) is a defensible primary surrogate, because the surrogate **cancels out of the relative
  baseline-vs-extension delta** — its fidelity is documented effort, not a claim. State which you used and why.
- **Secondary surrogate — RandomForest** on the same bytes, for the sensitivity probe. **Separate instance
  from `classifier_eval.py`'s evaluator** (see constraints).
- Save both to `models/`.

> **Gate B1 — the surrogate must actually work.** Report held-out benign/malicious accuracy for each
> surrogate. A surrogate at chance makes the whole attack meaningless (there is no loss gradient to climb).
> Require clearly-above-chance accuracy before using it to guide the attack; if the CNN underperforms the RF,
> say so and consider the RF as primary.

### 3. Gate B2 — compute + convergence probe (before committing any GA grid size)
Do **not** hardcode the Phase-1 report's "20 candidates × 30 generations" guess. Measure it.
- On a small set (e.g. 20–50 malicious samples): measure real surrogate **forward-pass latency**, and run the
  GA search while **logging surrogate loss (or benign-probability) vs. generation** for each sample.
- Report the actual convergence curve: how many generations are genuinely needed to meaningfully raise
  surrogate loss / flip predictions. It may be far fewer than 30 (early convergence) or more.
- **Fix the population/generation grid from this data**, then re-estimate total Phase-3 compute
  (grid × `n_poison` × surrogate latency) and flag whether it reopens the local-vs-VM question.

### 4. Build the surrogate-guided GA mode (`mode="chale_ga"`)
- **Target selection:** draw the `n_poison` samples to perturb from the **malicious** class (Chale/Hore are
  malicious→benign evasion attacks). Note this is a deliberate **divergence** from `poison.py`'s
  label-agnostic random selection — more faithful to the cited attacks, and to be documented as such. (The
  `random_swap` reference mode from step 1 keeps the legacy random selection, so Gate A is unaffected.)
- **Search:** per targeted sample, a generational search over sets of byte-position swaps, each candidate
  scored by the surrogate's **benign-class probability** (equivalently, classification loss w.r.t. the true
  malicious label — maximize it); selection + mutation across generations; grid from Gate B2. **Early-stop**
  a sample once the surrogate flips it to benign, to save compute.
- **Validity:** swaps only permute existing bytes, so range (0–255) and byte-multiset are preserved **by
  construction** — validity is structural, the key advantage over additive-perturbation methods (which the
  literature reports as up to ~80% invalid). Still **log a per-sample validity check** and report the rate;
  the point of Lens 4 is precisely *not* to assume what the literature has shown can fail.

> **Gate C — the attack must actually be adversarial.** On a held-out set of malicious samples, compare
> `chale_ga` against `random_swap` on the surrogate: report the malicious→benign flip rate and mean
> benign-probability increase for each. **`chale_ga` must clearly beat `random_swap`.** If the guided search
> is no better than random, the attack is not doing its job and Phase 3 would measure nothing — stop and
> report rather than proceeding.

### 5. Stop, report, checkpoint
- **Stop here.** Do **not** run the single-pass baseline on the full dataset, do **not** compute capture
  rates against Monkam's neighborhood, do **not** touch the experiment drivers — all Phase 3.
- Produce a completion report structured around steps 1–4, with **every gate's real numbers** quoted:
  Gate A pass/fail, Gate B1 surrogate accuracies, Gate B2 convergence curve + chosen grid + compute estimate,
  step-4 validity rate, Gate C flip-rate comparison.
- Recommend a git commit of `adversarial_attack.py` + `models/` as a checkpoint (user decides wording/timing).
- Do **not** update `CLAUDE.md` / `cc_summary.md` yet — the lens hasn't landed; docs update when Phase 3
  closes the substrate+baseline arc.

---

## What Phase 3 will do (context only — not part of this build)

For situational awareness, so the build's interfaces are Phase-3-ready: Phase 3 wires `chale_ga` into a
single-pass `run_all_clustering` + `classify_clusters` run on UNSW-NB15 (the Monkam static-threshold
baseline), reports `red_poison_capture_pct` against Monkam's documented ~40–70% neighborhood (as a sanity
check, not a bar), runs the surrogate-sensitivity probe (CNN vs RF delta-stability), and logs validity
metrics. Nothing in Phase 2 should pre-empt those runs — just leave the `chale_ga` attack callable behind the
same `(X_combined, y_combined, is_poisoned)` contract the drivers already expect.
