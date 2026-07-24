# LENS4_ADVERSARIAL_SUBSTRATE — Phase 1 Feasibility Investigation

**Scope of this lens (agreed):** build the shared **realistic-attack + surrogate-NIDS-classifier
substrate** and re-run the **Monkam static-threshold baseline** on it (UNSW-NB15 first). This lens does
**not** cover the TVI / game-theory / iteration extensions — those run *on top of* this substrate in later
work. Within that arc, **this file is Phase 1 only: a read-only investigation** that de-risks the plan and
ends with a Phase-2 build proposal, then stops. Nothing is built or modified here — the build genuinely
cannot be specified until Phase 1 resolves the open questions below.

---

## The reframed goal (this is what keeps Lens 4 scientifically honest)

The overarching question is **"can TVI, game theory, or iteration improve on Monkam et al.'s results on the
same setup?"** — a **relative** comparison. That framing is what makes the whole project defensible, and it
changes what Lens 4 must deliver:

- The scientific weight is carried by the **internal delta**: the Monkam-style *baseline* and the
  *extensions* run against the **same surrogate and the same attacks**, so the surrogate/attack choices
  **cancel out** of the comparison. You are not claiming "we hit Monkam's 40–70%"; you are claiming "on our
  faithful-as-possible rebuild of their setup, extension X captures more than the static-threshold baseline."
- Therefore **reproduction fidelity is a documented *effort*, not a *claim***. Mimic Monkam's setup as
  closely as possible, and list every divergence (60 vs 72 features; append+`is_poisoned` vs their in-place
  poisoning; approximated vs original attacks; your surrogate vs their unspecified one). Monkam's published
  numbers become a **sanity-check reference** ("our baseline lands in a plausible neighborhood, given the
  documented differences"), never the benchmark the result lives or dies by.
- Do not let "Lens 4 reproduces Monkam's attacks" appear in any writeup. The honest headline is "a
  faithful-as-possible Monkam-style substrate for a controlled baseline-vs-extension comparison." This is
  the same overclaim discipline that forced the Lens 2 and Lens 3 rewrites — it applies most sharply here.

---

## What the source papers actually say (read before scoping — it redirects the attacker choice)

Both attacks Monkam cites are **model-relative evasion attacks**: they perturb payloads to fool a *trained
NIDS classifier*, scored by misclassification. Your pipeline is unsupervised and pre-training, so a
**surrogate NIDS classifier must be built** for either attack to run at all. Beyond that, the two attacks
are very differently approximable:

- **Chale et al. (2023, *Optimization Letters*) — the tractable one.** A meta-heuristic generative model
  that maximizes a surrogate model's classification loss by **repeatedly substituting functional units of
  the payload with functionally-equivalent counterparts** (payload-as-code), then transfers the result to
  separate test NIDS. Two facts make this the *primary* Lens-4 attack: (1) it operates on the **same
  payload-byte representation** the project already uses — no representation gap; (2) `poison.py`'s existing
  byte-position swaps are described in-repo as *"simulating functionally-equivalent code substitution from
  Chale et al."* — i.e. the project already has the **random** version of this attack. The Lens-4 upgrade is
  to make the substitution **surrogate-loss-guided** (search for swaps that increase surrogate loss) instead
  of random. That is an *extension of existing code*, not a new framework. Chale's own setup used a 1-D CNN
  surrogate plus three transfer NIDS — a concrete, citable template.
- **Hore et al. (2023, "Deep PackGen") — the hard, lower-fidelity one.** A full **deep-RL** framework whose
  agent learns functionality-preserving perturbations on **raw PCAP packets** (extracted, feature-engineered,
  normalized) — a **different data representation** than the project's 1500-byte Payload-Byte vectors, which
  Monkam did *not* bridge. The functionality-preserving action space is the paper's core contribution (prior
  packet attacks were "not playable"); reconstructing it without their code is a large effort. No public
  code surfaced in a web search (published arXiv 2305.11039 / ACM TOPS 2025). A *faithful* approximation is
  costly and low-confidence.
- **The surrogate choice is a published confound, not just a worry.** Recent work (arXiv 2505.01328, 2025)
  finds that **surrogate-model choice significantly drives** adversarial-example effectiveness, and that a
  large fraction of adversarial examples in some methods are **invalid / non-functional** (reported up to
  ~80%). Consequences for Lens 4: (a) a **surrogate-sensitivity check** is mandatory, not optional; (b)
  **validity/playability of generated samples is a first-class metric** — an attack that produces mostly
  invalid packets is not realistic, regardless of its capture rate.

**Net effect on scope:** approximate **Chale first** (primary, tractable, same representation, extends
`poison.py`); treat **Hore as an explicitly-flagged stretch** whose faithful reproduction may not be worth
the cost — that call is made *from Phase 1 evidence*, not pre-committed. "Approximate both to the best of
our ability" is honored by building the tractable one well and honestly documenting the other as a lighter
approximation or a deferral.

---

## Hard constraints (house style)

- **No repository code may be modified in Phase 1.** Read, quote, reason, design, estimate cost — do not
  write the surrogate, the attack, or the harness. Everything is a *proposal* in the report.
- **UNSW-NB15 only** for this whole arc's first pass; CICIDS2017 extension is deferred until the harness
  works (its ~4.6 GB CSV and longer runs are not worth paying during bring-up).
- **Keep the append + `is_poisoned` design.** It diverges from Monkam's in-place poisoning but is required
  for a clean capture-rate ground truth; document the divergence, don't erase it.
- Environment: `cd C:\TDA; .\venv312\Scripts\Activate.ps1`. Reference regime: **seed 42, `MAX_SAMPLES=5000`,
  `POISON_RATE=0.10`**. Re-read the live `CLAUDE.md` / `cc_summary.md` and both source PDFs (now in the repo)
  before starting.

---

## Numbered Phase 1 steps

### 1. Confirm the harness contract still holds under a surrogate-guided attacker
- Confirm the swap-in point: per cc_summary §10.F, replacing the attack means changing only `poison.py`'s
  output at its call sites — the attack must still return the `(X_combined, y_combined, is_poisoned)` triple
  so `tda_pipeline.py`, `clustering.py`, and `iterative_filter.py` are untouched. Verify this contract by
  reading the call sites (`run_baseline.py`, `run_iterative.py`, `run_multi_seed.py`), don't assume it.
- State plainly how the **append + `is_poisoned`** design gives capture-rate ground truth (the mask marks
  exactly which appended rows are adversarial), and record the divergence from Monkam's in-place poisoning
  (their setup has no clean ground-truth mask) for the writeup's limitations list.

### 2. Pin down Monkam's setup to mimic (read the in-repo Monkam PDF)
Extract, verbatim where it matters, exactly how Monkam used UNSW-NB15 so "the same setup" is concrete:
sample sizes and benign/attack composition, poison rate, the Payload-Byte 1500 format, and — critically —
**what classifier arrangement their cited attacks assume**. Monkam cites Chale/Hore as attack sources but
those attacks need a target model; determine what Monkam actually attacked, or record that it is
underspecified in the paper (a documented gap the project must fill with its own defensible surrogate). This
step defines the substrate the entire baseline-vs-extension comparison rests on.

### 3. Design the surrogate NIDS classifier + the sensitivity gate (design only, no build)
- Specify a defensible surrogate. Chale's 1-D CNN over payload bytes is a concrete, citable default; name at
  least one alternative (e.g. a RandomForest or shallow FFNN on the same bytes) for the sensitivity check.
- **Design the surrogate-sensitivity probe** and justify it with the published finding that surrogate choice
  drives adversarial results (arXiv 2505.01328): the probe re-runs the *eventual* baseline-vs-extension
  comparison against ≥2 surrogates and checks whether the **delta** (not the absolute capture rate) is
  stable. Under the reframed goal this is a **robustness check on the comparison**, not a go/no-go on a
  reproduction claim — say so. Specify what "stable enough to trust the delta" means before any numbers exist.

### 4. Chale approximation — feasibility + concrete design (the primary attack)
- Confirm `poison.py`'s current byte-swap is the random precursor; quote the relevant lines.
- Specify the upgrade: a meta-heuristic (GA-style) search over functionally-equivalent byte substitutions
  that **maximizes surrogate classification loss**, constrained to byte-validity (values 0–255; swaps
  preserve the byte multiset, so functional-equivalence is structurally maintained — note this is exactly
  why swap-based moves are a good fit for the validity constraint).
- Define the **validity metric** up front (fraction of generated samples that remain valid/playable) — the
  literature shows this is where these methods fail, so it must be measured, not assumed.
- Estimate compute: surrogate-loss evaluations per sample × `n_poison` per run, and whether that fits local
  overnight at `MAX_SAMPLES=5000` or reopens the VM question.

### 5. Hore approximation — honest feasibility call (the stretch attack)
State the gaps plainly and make a recommendation *from evidence*, not obligation:
- Representation gap (raw PCAP vs Payload-Byte), full DRL framework, functionality-preservation as the hard
  core, no public code found.
- Weigh three options: (a) a *faithful* Deep PackGen approximation (high cost, low confidence); (b) a
  *lighter* RL-flavored byte-perturbation variant that borrows the "sequential functionality-preserving
  perturbation" idea without the full framework; (c) **defer Hore** and run the comparison on the
  Chale-approximation alone, documenting Hore as future work. Recommend one, with reasoning. "Best of our
  ability" explicitly permits (b) or (c) if (a) isn't worth it.

### 6. Baseline re-run plan (the file's end target — plan only)
Specify how Phase 3 will re-run the **Monkam static-threshold, single-pass** `classify_clusters` (the
100%-purity rule, no iteration, no extension) on UNSW-NB15 under the realistic attack, and report the
baseline capture rate (share-of-poisoned-population, matching the project's `red_poison_capture_pct` and
Monkam's own population-share metric). Pre-declare what "lands in a plausible neighborhood of Monkam's
~40–70%" means, and state explicitly that a mismatch is a *documented divergence*, not a failure — the
baseline's only job is to be the honest comparison point the extensions are later measured against.

### 7. Phase-2/3 build proposal + stop
Sketch, in the report only: **Phase 2** (build surrogate + Chale-approximation + integration behind the
existing `(X_combined, y_combined, is_poisoned)` contract, with a zero-behavior-change check that random-swap
mode still reproduces current results before the surrogate-guided mode is trusted) and **Phase 3** (baseline
re-run + surrogate-sensitivity probe + validity metrics), each with its own investigate→implement stop
condition. Recommend **Chale-first**, with the Hore decision taken from step 5.

---

## Stop condition & completion report

After step 7, **stop**. No surrogate, attack, or harness code is written; no repo file is modified; no
baseline is run. Produce a report structured around the seven numbered steps, quoting `poison.py`'s current
swap code (step 4) and the relevant Monkam setup details (step 2) verbatim.

Close with three explicit recommendations for the user:

> **(a) Attacker scope:** Chale-first only, or Chale + a Hore variant/deferral — with the cost and fidelity
> tradeoff stated plainly (step 5).
>
> **(b) Surrogate(s):** which surrogate model to build, and which second one for the sensitivity probe.
>
> **(c) Framing confirmation:** re-confirm the relative baseline-vs-extension goal, so the reproduction
> language stays honest (documented-effort, not reproduction-claim) all the way into the writeup.

Report every divergence from Monkam honestly rather than rounding toward "reproduction." Do not update
`CLAUDE.md` / `cc_summary.md` from a Phase-1 investigation; docs are updated only when a lens actually lands.
