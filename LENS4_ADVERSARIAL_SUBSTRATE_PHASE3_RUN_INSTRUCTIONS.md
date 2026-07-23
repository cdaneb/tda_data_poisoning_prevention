# LENS4_ADVERSARIAL_SUBSTRATE — Phase 3 RUN + Attribution Instructions

**This phase runs experiments and lands the substrate+baseline arc.** It re-runs the **Monkam
static-threshold baseline** (single-pass, 100%-purity `classify_clusters`, no iteration) on the
Chale-approximation attack built in Phase 2, and — crucially — **attributes** the resulting capture rate to
its real drivers before interpreting it. It does **not** run the TVI / game-theory / iteration extensions;
those are later lenses that build on the substrate this phase finishes.

The load-bearing addition over the original Phase-3 sketch: an **attribution front gate**. Phase 2 found that
`chale_ga`'s strength came substantially from raising the swap count to `n_swaps=60` (vs `poison.py`'s legacy
`~10–50`). So a raw "capture rate under `chale_ga`" number confounds three things — **perturbation magnitude,
target selection, and adversarial guidance** — and under the relative-comparison framing an unattributed
number is nearly worthless. Phase 3 decomposes it first, then interprets.

---

## Hard constraints

- **Single-pass Monkam baseline only.** `run_all_clustering` + `classify_clusters` with the unmodified
  `poison_fraction == 0/1.0` rule, no iteration, no threshold relaxation, no TVI, no game theory. Those are
  separate lenses; keeping them out is what makes this the honest baseline they'll later be measured against.
- **Do not edit the existing drivers.** Add a new script (e.g. `run_lens4_baseline.py`) that imports
  `adversarial_attack`, `tda_pipeline`, and `clustering` and mirrors `run_baseline.py`'s single-pass logic.
  `poison.py`, `run_baseline.py`, and the other drivers stay intact and runnable.
- **Attacker-side and detector-side numbers stay distinct.** Phase 2's "83.2% flipped to benign" is the
  *surrogate* evasion rate (attacker's-eye). Phase 3's capture rate is what the *TDA detector* catches. Never
  let one bleed into the other in the report — they answer different questions.
- **Seed 42 first, multi-seed to confirm.** Establish the result on seed 42 (matches all prior single-run
  work), then re-run the key comparison across `SEEDS=[42,123,456,789,1024]` before any number is treated as
  writeup-ready.
- Environment / regime: `cd C:\TDA; .\venv312\Scripts\Activate.ps1`; `MAX_SAMPLES=5000`, `POISON_RATE=0.10`.
  Reuse the saved `models/` surrogates from Phase 2 — do not retrain. Re-read the live `CLAUDE.md` /
  `cc_summary.md` first.

---

## Steps

### 1. Generate the attribution ladder (seed 42)
Produce poisoned UNSW-NB15 under three attack conditions, all via the same `(X_combined, y_combined,
is_poisoned)` contract, differing in exactly one factor at a time so the capture rate can be decomposed:

| Variant | Target selection | Swaps | Guidance | Isolates |
|---|---|---|---|---|
| **L** (legacy) | random (label-agnostic) | `~10–50` random | none | the current baseline reference (~3.6% OPTICS) |
| **R60** (magnitude control) | malicious-only | `60` random | none | magnitude + target selection vs L |
| **G60** (full attack) | malicious-only | `60` guided (MLP surrogate) | yes | **guidance** vs R60 |

The clean guidance contrast is **R60 → G60** (identical except guidance). L is the legacy anchor. Record
generation time and per-sample validity for each. (If you want to further separate target-selection from
magnitude, an optional random-target-60 variant does it — but R60→G60 is the load-bearing control.)

### 2. FRONT GATE — attribute the capture rate on the detector (do this before interpreting anything)
Run the single-pass Monkam baseline on **L, R60, and G60**, all four algorithms, and report
`red_poison_capture_pct` (and sanitized purity / poisoned-pool precision) for each. Then decompose:
- **L → R60** = the effect of more perturbation + malicious targeting.
- **R60 → G60** = the effect of adversarial **guidance**, holding magnitude and targeting fixed.

**Interpret direction-agnostically — do not assume guidance raises capture.** All three outcomes are
scientifically meaningful and must be reported as whichever actually occurs:
- **G60 ≈ R60:** capture is driven by perturbation magnitude, not adversarial sophistication. The honest
  claim shrinks to "louder poison is more topologically detectable" — guidance is invisible to the detector.
- **G60 ≫ R60:** guided evasion produces *more* topologically distinct poison — surprising; investigate why
  (guided samples may cluster tightly).
- **G60 ≪ R60:** guided evasion *hides* from the topological detector better than random noise does. This is
  the most important possible result — it is direct evidence about whether the TDA defense is robust to an
  **adaptive** attacker, and it is exactly the tension that motivates Lens 3's game-theoretic framing (an
  attacker that responds to the defender). Flag it prominently if it appears.

Do not proceed to the Monkam-neighborhood comparison until this decomposition is in hand — the neighborhood
number means different things depending on which outcome above holds.

### 3. Surrogate-sensitivity of the substrate
Regenerate **G60 using the RF surrogate** (`G60-RF`), re-run the single-pass baseline, and compare its
capture rate to `G60-MLP`. Report whether the substrate's capture rate is **stable across which surrogate
guided the attack** (per the published finding that surrogate choice drives adversarial results). If capture
swings materially between MLP- and RF-guided attacks, that instability is itself a finding the downstream
extensions will have to account for; if it's stable, the substrate is trustworthy for the extension work.

### 4. Monkam-neighborhood sanity check + validity + divergences
- Compare the **G60** baseline capture (OPTICS is the meaningful algorithm; DBSCAN/HDBSCAN/MeanShift are
  ~0 as always) to Monkam's documented **~40–70%** population-share neighborhood — as a **sanity check, not
  a bar**. Pre-state (per Phase 1 step 6) that ~20–80% reads as "in the neighborhood, modulo documented
  differences," and that a mismatch is an attributable divergence, not a failure.
- Report the **validity rate** (expected ~100% by construction; confirm, don't assume).
- List **every divergence from Monkam** in one place for the limitations record: 60 vs 72 features; the
  binarizer 0.4-vs-0.3 ambiguity *in Monkam's own paper*; append + `is_poisoned` vs their in-place poisoning;
  malicious-only target selection vs their unspecified scheme; approximated-not-original attack; a surrogate
  Monkam never specified. Do not minimize any of these.

### 5. Multi-seed confirmation (gate before any writeup number)
Once the seed-42 picture is clear and sensible, re-run at least **R60 and G60** (the guidance contrast)
across `SEEDS=[42,123,456,789,1024]` and report **mean ± std** capture for each. Single-seed numbers are
indicative only; the multi-seed R60-vs-G60 delta is the result that can enter a writeup. If the sign of the
R60→G60 effect is not stable across seeds, say so — that instability is the honest finding.

### 6. Land the arc — report, doc update, commit
This phase **completes the substrate+baseline arc**, so the docs update now (the first time in this Lens-4
work that a doc update is warranted).
- Produce a report structured around steps 1–5, with the full attribution table (capture × variant ×
  algorithm), the surrogate-sensitivity comparison, the Monkam-neighborhood reading, validity, the divergence
  list, and the multi-seed R60-vs-G60 delta.
- **Update `CLAUDE.md` and `cc_summary.md`** to reflect what actually landed: a realistic Chale-approximation
  attack substrate + surrogate(s), and the Monkam static-threshold baseline capture rate under it —
  **including the attribution caveat** (how much of the capture is magnitude vs guidance) and any
  surrogate-sensitivity or seed-stability caveats. Do not write "reproduces Monkam"; write "faithful-as-
  possible Monkam-style substrate + baseline, with documented divergences."
- Recommend a git commit of `run_lens4_baseline.py` + the results/figures produced (wording/timing the
  user's call).

---

## What comes after (context only — not this phase)

With the substrate landed and the baseline capture attributed, the project is finally positioned to run the
actual thesis question — do **iteration**, **TVI**, or the **game-theoretic** threshold/defense (Lens 3
Reading T or D) improve on this baseline *against a realistic attack* — all on this common substrate, so the
comparisons are clean. If step 2 produced the **G60 ≪ R60** outcome (guided evasion hides from the detector),
that result should directly reshape how the Lens 3 game and any adaptive-attacker (true Lens-4-attacker)
work is framed: it would be the first concrete evidence in this project that the topological defense has an
adaptive-attack weakness worth defending against.
