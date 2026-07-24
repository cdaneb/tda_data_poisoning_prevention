# Phase W Completion Report — WIRE Hardening and Provenance Audit

## W0 — Environment freeze ✅

`requirements.lock.txt` created and header-documented (Python 3.12.3, hostname `e2fe114bedaa`, 2026-07-23T18:57:09Z, tied to the 2.2000% reproduction value). `README.md` updated with one line distinguishing it from `requirements.txt`. Resolved versions: **giotto-tda 0.6.2, scikit-learn 1.3.2, numpy 1.26.4, scipy 1.17.1, pandas 3.0.5, joblib 1.5.3** (same set Phase R already verified).

## W1 — `paths.py` config module ✅

Created `paths.py` (env-var override → repo-relative default, resolved from `__file__`, never cwd). Updated `data_loader.py`, `run_lens4_baseline.py`, `run_multi_seed.py`, `run_iterative.py` — nothing else touched in those files. Deleted the Phase-R `C:\TDA\data` workaround; real wiring now lives in a proper `data/` symlink directory (already covered by the pre-existing `data/` gitignore rule). Created `.env.example` (tracked) and `.env.wire` (untracked, now correctly gitignored — I also had to fix two gitignore bugs while I was in there: `requirements.lock.txt` was being caught by the blanket `*.txt` rule, and the old `C:\\TDA\\data/` line needed removing). Verified empirically: correct default resolution, env-var override works, and — the actual point of this exercise — **cwd-independence**, which the old scheme never had.

### Diffs

```diff
--- a/data_loader.py
+++ b/data_loader.py
@@ -11,9 +11,8 @@ Column structure (1505 columns total):
 """
 import numpy as np
 import pandas as pd
-from pathlib import Path
 
-DATA_DIR = Path(r"C:\TDA\data")
+from paths import DATA_DIR
 
 PAYLOAD_COLUMNS = [f"payload_byte_{i}" for i in range(1, 1501)]
 LABEL_COLUMN = "label"
```

```diff
--- a/run_lens4_baseline.py
+++ b/run_lens4_baseline.py
@@ -15,7 +15,6 @@ import time
 import json
 import joblib
 import numpy as np
-from pathlib import Path
 
 from data_loader import load_unsw
 from tda_pipeline import extract_tda_features
@@ -24,9 +23,8 @@ from results_io import convert_for_json
 from adversarial_attack import (
     random_swap_attack, malicious_random_attack, chale_ga_attack, train_surrogates,
 )
+from paths import RESULTS_DIR, MODELS_DIR
 
-RESULTS_DIR = Path(r"C:\TDA\results")
-MODELS_DIR = Path(r"C:\TDA\models")
 SEEDS = [42, 123, 456, 789, 1024]
 MAX_SAMPLES = 5000
 POISON_RATE = 0.10
```

```diff
--- a/run_multi_seed.py
+++ b/run_multi_seed.py
@@ -10,12 +10,12 @@ Runs on BOTH datasets: UNSW-NB15 and CICIDS2017.
 import numpy as np
 import json
 import time
-from pathlib import Path
 from data_loader import load_unsw, load_cicids
 from poison import poison_dataset
 from iterative_filter import iterative_filter
 from clustering import run_all_clustering, classify_clusters
 from results_io import convert_for_json
+from paths import RESULTS_DIR
 
 
 # ============================================================
@@ -26,7 +26,6 @@ MAX_SAMPLES = 5000                  # samples per run
 POISON_RATE = 0.10
 MAX_ITERATIONS = 10
 ALGORITHMS = ["DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]
-RESULTS_DIR = Path(r"C:\TDA\results")
 # ============================================================
```

```diff
--- a/run_iterative.py
+++ b/run_iterative.py
@@ -8,11 +8,11 @@ comprehensive results for analysis and visualization.
 import numpy as np
 import json
 import time
-from pathlib import Path
 from data_loader import load_unsw
 from poison import poison_dataset
 from iterative_filter import iterative_filter
 from results_io import convert_for_json
+from paths import RESULTS_DIR
 
 
 def run_iterative_experiment(dataset_name, X, y, max_samples=5000,
@@ -122,7 +122,7 @@ def run_iterative_experiment(dataset_name, X, y, max_samples=5000,
                   f"{w_str:>10}")
 
     # Save results to JSON for later visualization
-    output_dir = Path(r"C:\TDA\results")
+    output_dir = RESULTS_DIR
     output_dir.mkdir(exist_ok=True)
 
     json_results = {}
```

```diff
--- a/.gitignore
+++ b/.gitignore
@@ -24,6 +24,10 @@ data/
 *.log
 *.txt
 !requirements.txt
+!requirements.lock.txt
+
+# WIRE-specific path overrides (machine-local; see .env.example)
+.env.wire
 
 # OS files
 .DS_Store
```

```diff
--- a/README.md
+++ b/README.md
@@ -49,6 +49,8 @@ source venv/bin/activate        # Linux/Mac
 pip install -r requirements.txt
 ```
 
+`requirements.txt` documents intent (loose `>=` version bounds); `requirements.lock.txt` documents the exact resolution known to reproduce this project's recorded results (see its header comment). New environments should install from `requirements.lock.txt`, not `requirements.txt`, unless intentionally testing against newer dependency versions.
+
 ## Datasets
```

## W2 — Blocking re-gate ✅ PASS

`tools/repro_check.py` (tracked, `--expect`/`--tol` flags) run through the new wiring: **OPTICS capture = 2.2000%, exact match**, `X_tda.shape == (5500, 60)` confirmed, wall-clock 66.9s (consistent with R3's 68.2s). The path rewiring changed no behavior.

## W3 — Provenance audit

**Prerequisite note carried forward:** no `docs/LENS4_ROBUSTNESS_TESTS_REPORT.md` exists anywhere in the repo. `docs/PROJECT_HANDOFF_1.md` §4/§6 contains everything Test A/B/C would have reported, so I used it as the sole source. Flagging again in case a separate report was expected to exist and doesn't.

| # | Result | Artifact | Code | Status |
|---|---|---|---|---|
| 1 | **N** — noise only, malicious-targeted, 5 seeds | none | none — `poison.py` always couples noise+swap and targets the whole dataset, not malicious-only | **ORPHANED** |
| 2 | **S** (=R60) — 60 random swaps, no noise, 5 seeds | `results/lens4_baseline_multiseed.json["R60"]` | `adversarial_attack.py::malicious_random_attack` | **REPRODUCIBLE** (W2 just re-confirmed this exactly) |
| 3 | **SG** (=G60-MLP) — 60 GA-guided swaps, MLP surrogate, 5 seeds | `results/lens4_baseline_multiseed.json["G60-MLP"]` | `adversarial_attack.py::chale_ga_attack` + `models/surrogate_mlp.joblib` | **REPRODUCIBLE** |
| 4 | **SG-RF** — same, RF surrogate, 5 seeds | **partial** — only seed-42 (11.8%) in `lens4_baseline_seed42_ladder.json`; no 5-seed file backs the reported 7.92±3.86 | `chale_ga_attack` + `models/surrogate_rf.joblib` exists; `run_lens4_baseline.py::main()` just doesn't sweep this variant past seed 42 | **CODE EXISTS, ARTIFACT INCOMPLETE** |
| 5 | **NS** — noise + 60 random swaps, malicious-targeted, 5 seeds | none | none — no function combines noise with fixed-count malicious-only swap targeting | **ORPHANED** |
| 6 | **NSG** — noise + 60 guided swaps, 5 seeds | none | none | **ORPHANED** |
| 7 | **L_unmatched** — legacy, 5 seeds | **partial** — only seed-42 (6.6%) in the ladder file; the reported 2.56±2.21 has no 5-seed backing artifact | `adversarial_attack.py::random_swap_attack` (=`poison.py`) exists; not swept past seed 42 in `run_lens4_baseline.py` | **CODE EXISTS, ARTIFACT INCOMPLETE** |
| 8 | **Test A** — S & SG at threshold 0.3, 5 seeds | none | none — `Binarizer(threshold=0.4, ...)` is hardcoded at exactly one site, `tda_pipeline.py:47`, confirming Phase R's finding; no parameter exists anywhere to override it | **ORPHANED** |
| 9 | **Test B** — transpositions | = row 2 (S), already counted | — | **REPRODUCIBLE** |
| 10 | **Test B** — block reversal (k=120) | none | **none anywhere in the repo** | **ORPHANED** |
| 11 | **Test B** — block swap (2×k=60) | none | **none anywhere in the repo** | **ORPHANED** |
| 12 | **Test B** — cyclic shift | none | **none anywhere in the repo** | **ORPHANED** |
| 13 | **Test C** — CICIDS surrogate accuracies (MLP 97.7%, RF 99.4%) | `models/surrogate_mlp_cicids.joblib`, `surrogate_rf_cicids.joblib` exist on disk | **zero `.py` file references either filename** — `train_surrogates()` is dataset-agnostic and could reproduce this, but no committed script calls `load_cicids()` + `train_surrogates()` together | **ARTIFACT-ONLY** |
| 14 | **Test C** — S/SG capture on CICIDS (partial, 2 seeds) | none — no `lens4_baseline_*cicids*` file exists | `run_lens4_baseline.py` is hardcoded to `load_unsw()` (confirmed by grep — zero "cicids" hits); attack functions themselves are dataset-agnostic | **ORPHANED** |
| 15 | **Test C** — HDBSCAN=5.0% anecdote | none | falls out for free once #14 exists (`run_all_clustering` always runs all 4 algorithms) | **ORPHANED** |
| 16 | **Step 0** — count-invariance (0/200 @ 0.4 and 0.3) | none | **none anywhere** — no code computes a foreground-pixel count or compares binarized images; per Phase R's R5, binarized images aren't even retained by the current pipeline | **ORPHANED** |
| 17 | **Swap-count sweep** (10/20/30/60/100) | none | **mostly exists** — `malicious_random_attack(..., n_swaps=N)` already takes `n_swaps` as a free parameter; only the sweep-loop driver itself is missing | **CODE MOSTLY EXISTS, driver missing** (cheapest orphan by far) |
| 18 | **Phase 2 gate** — bit-equivalence to `poison.py` | n/a | `random_swap_attack` delegates to `poison_dataset` verbatim | **guaranteed by construction**, not a runtime-checked gate — no assertion script exists, but it can't silently drift either |
| 19 | **Phase 2 gate** — surrogate accuracies (MLP 0.9690/RF 0.9850, UNSW) | fitted models exist; the accuracy scalars themselves are **not saved anywhere**, and the exact held-out test split (`X_test`/`y_test`) is returned in-memory only, never persisted | `train_surrogates()` computes exactly these, deterministically (`random_state=42`) | **REPRODUCIBLE VIA RETRAIN**, not a free re-read from disk |
| 20 | **Phase 2 gate** — latency (0.215 ms/call) | none | none — only coarse total-elapsed-time prints exist, no per-call instrumentation | **ORPHANED** (trivial to add) |
| 21 | **Phase 2 gate** — guided flip rate (83.2%, full run) | none (not persisted; only printed) | `chale_ga_attack` computes and prints this inline | **REPRODUCIBLE via re-run**, not currently saved to disk |
| 22 | **Phase 2 gate** — "10% compute-matched random" control | none | **none** — `malicious_random_attack` does one random draw per sample, not a compute-matched multi-draw search budget-matched to the GA | **ORPHANED** |

**Counts:** REPRODUCIBLE: 2 (S, SG — note transpositions is the same result as S, not a third). ARTIFACT-ONLY: 1 (CICIDS surrogate models). CODE-EXISTS-BUT-NOT-FULLY-RUN (a real 4th bucket the 3-category scheme doesn't cleanly fit — flagging rather than force it): 2 (SG-RF, L_unmatched — both single-seed-only on disk). ORPHANED: 12. Two results (bit-equivalence, surrogate accuracy) don't fit the scheme cleanly either — noted individually above rather than mis-bucketed.

**Targeted checks (W3.5):**
- **Noise-based attacks (N/NS/NSG):** confirmed absent from committed code — `poison.py::poison_dataset` is the only noise-capable path, and it always couples Gaussian noise with an *uncontrolled* (10–50, random) swap count applied to a *whole-dataset* (not malicious-only) sample — structurally incompatible with N/NS/NSG's apparent design (fixed swap count, malicious-only targeting). No function isolates noise as an independent variable.
- **Binarizer threshold:** hardcoded, confirmed at exactly one site (`tda_pipeline.py:47`), no parameterization anywhere in the codebase.
- **Permutation families beyond transpositions:** confirmed **zero** — `_random_swap_set`/`malicious_random_attack`/`chale_ga_attack` are all disjoint-pair transpositions; no block-reversal, block-swap, or cyclic-shift function exists anywhere in the repo.
- **CICIDS path for the Lens 4 substrate:** confirmed absent from `adversarial_attack.py`/`run_lens4_baseline.py` (zero references). The *models* exist (someone ran `train_surrogates()` against CICIDS interactively or via an uncommitted script); the *driver* doesn't. `run_multi_seed.py` (an older lens) already has a clean, reusable `run_dataset_experiment(dataset_name, X_full, y_full)` pattern that `run_lens4_baseline.py` could mirror — precedent exists, it's just not been applied here.

## W4 — Documentation corrections

All three "known errors" turned out, on direct verification, not to require the described fix:

1. **Radial centers:** grepped `cc_summary.md`, `CLAUDE.md`, `docs/*.md`, `README.md`, `LENS1`/`LENS3` instruction files for the incorrect `[0,1500]`/`[0,750]`/`[1500,0]` values — **zero occurrences anywhere**. `cc_summary.md:250` already states the correct values verbatim (`[0, 50]`, `[0, 25]`, `[30, 0]]`, matching `tda_pipeline.py` exactly). **0 locations changed.** The untracked companion files this step also asked about (`TERMINOLOGY.md`, `poster_blocks.tex`, referenced in `PROJECT_HANDOFF_1.md`'s own header) don't exist anywhere on this machine either — nothing to check or report on there.
2. **GUDHI backend:** verified directly by reading `gtda/homology/cubical.py` and `gtda/externals/`. `CubicalPersistence` imports `CubicalComplex`/`PeriodicCubicalComplex` from `gtda.externals.python.cubical_complex_interface`, which wraps compiled modules `gtda_cubical_complex.so` / `gtda_persistent_cohomology.so` — a vendored, compiled port of GUDHI's own C++ classes (verbatim GUDHI API naming: `Bitmap_cubical_complex_base_interface`, `Persistent_cohomology_interface`, `homology_coeff_field`). `giotto-ph` appears only in `gtda/homology/simplicial.py` (Vietoris-Rips), never in the cubical path. **`docs/PROJECT_HANDOFF_1.md`'s claim ("GUDHI C++ backend") is correct as stated — no doc change needed.** This also means **Phase R's own inference was imprecise** (it reasoned from `pip show`'s dependency list to "giotto-ph is the cubical backend," which conflates giotto-tda's two separate persistence backends). Noting that correction here since `R_PHASE_REPORT.md` isn't one of the docs this step authorized me to edit.
3. **Stale `.gitignore` claim:** re-grepped `CLAUDE.md` fully for "gitignore," "exclude," and "results" — **the claim "`.gitignore` excludes `results/*.json`" does not appear anywhere in `CLAUDE.md`'s actual text.** This means my own `R_PHASE_REPORT.md` (Phase R, R2) misattributed that claim to `CLAUDE.md` — it was actually a line from the Phase R task prompt itself, not something I verified against the live file at the time. **Nothing to correct in `CLAUDE.md` — the error was in my prior report, not the doc.** Flagging this plainly rather than quietly rounding it away; happy to correct `R_PHASE_REPORT.md` itself if asked.
4. `flask_metadata.json`'s CICIDS row-count discrepancy — noted, not touched (read-only mount, not ours to fix), per instruction.

**Net result: zero documentation edits made in W4.** Every described error either doesn't exist in the current tracked docs, or (GUDHI) was actually correct and it was the prior phase's inference that needed the correction.

## W5 — Parallelization and cost blockers (analysis only)

1. **Non-namespaced output paths, confirmed and enumerated exactly:**
   - `run_lens4_baseline.py`: `lens4_baseline_seed42_ladder.json` (seed baked into the name, single-seed only), `lens4_baseline_multiseed.json` (**written repeatedly mid-loop, after every seed × variant** — the worst collision risk of the three: parallel seed processes wouldn't just collide at the end, they'd stomp on each other throughout the run), `lens4_baseline_full.json`.
   - `run_multi_seed.py`: `multi_seed_per_seed_{dataset}.json` / `multi_seed_aggregated_{dataset}.json` — namespaced by dataset, not seed; written once, after the internal 5-seed loop completes in-process, so collision only arises if the 5-seed loop itself is split across processes.
   - `run_iterative.py`: `iterative_results_{dataset}.json` — namespaced by dataset only; not currently a live risk since the script hardcodes seed 42 with no seed loop, but would collide if ever parallelized across seeds.
2. **Proposed (not implemented) naming scheme:** suffix every output filename with `_seed{N}` (e.g. `lens4_baseline_multiseed_seed123.json`), write to per-process files unconditionally, then a small merge step (a script that globs `*_seed*.json` per experiment family and folds them into the existing combined-file shape) reassembles the current aggregate format. This keeps the on-disk *final* format unchanged — only the intermediate per-process writes change.
3. **G60-RF's ~2366s cost, inspected directly:** the population **is** already batched — `score_population()` (`adversarial_attack.py:172-180`) does one `predict_proba` call per generation across all 50 candidates, exactly to amortize RF's per-call dispatch overhead, per the code's own comment. That axis is not the opportunity. The *unbatched* axis is **across the 500 targeted samples** — each runs its own independent, sequential `ga_search_one_sample()` call. Two compounding factors likely drive the MLP-vs-RF gap (labeling this inference, since the per-sample `generations_used` log isn't persisted anywhere — see W3 row 16's sibling finding — so this can't be computed exactly from disk): (a) RF's larger fixed per-call dispatch cost, and (b) RF's much lower flip rate (CLAUDE.md's seed-42 anecdote: 17.6% RF vs. 83.8% MLP) means most RF samples never trigger the early-stop and run the full 100 generations, while most MLP samples stop early — so RF plausibly pays both a higher per-call cost *and* far more total generations. A cross-sample batching redesign (running many samples' GA searches in lockstep so one generation's `predict_proba` call spans all in-flight samples × population, not just one sample × population) would amortize RF's fixed per-call overhead far more aggressively than today's per-sample batching — plausibly a large speedup if that fixed cost dominates over per-row tree traversal, but I can't quantify it precisely without the missing per-sample timing data. Not implemented, per instruction.
4. **`n_jobs` usage:** `tda_pipeline.py` already passes `n_jobs=-1` at every pipeline step **except one** — `PersistenceEntropy(nan_fill_value=-1)` (line 66) omits it, even though `PersistenceEntropy.__init__` does accept `n_jobs` (confirmed via `inspect.signature`). Every other transformer (`HeightFiltration`, `RadialFiltration`, `Binarizer`, `CubicalPersistence`, `Scaler`, `Amplitude`, and the outer `make_union`) already has it set. So: **the pipeline is not single-threaded** — Phase R's 51.8s figure already reflects full parallelism across all 128 available cores, with one small, easily-fixed gap (`PersistenceEntropy`). This corrects an assumption embedded in this phase's own framing of that number.

---

## Summary

- **Environment:** ✅ **locked** — `requirements.lock.txt` written and staged (commit pending — see blocker below); giotto-tda 0.6.2 / scikit-learn 1.3.2 / numpy 1.26.4 / scipy 1.17.1 / pandas 3.0.5 / joblib 1.5.3.
- **Re-gate:** ✅ **pass — 2.2000%**, exact, through the new `paths.py` wiring.
- **Provenance table:** 22 rows enumerated. **REPRODUCIBLE: 2. ARTIFACT-ONLY: 1. CODE-EXISTS-NOT-FULLY-RUN: 2. ORPHANED: 12.** (Remaining rows are the main-effects derivations and two gate results that don't fit the 3-bucket scheme cleanly, noted individually.)
- **Highest-priority orphans, in order — confirming the expectation on Test B and Test C:**
  1. **Test B (permutation families)** — confirmed high-value: it's this project's stated best poster figure (§4's own framing) and currently has **zero supporting code** for 3 of its 4 families. Effort: moderate (~3 new functions, each a straightforward variant of the existing `malicious_random_attack` pattern) + trivial compute (~11 min for all three × 5 seeds).
  2. **Swap-count sweep** — not high-stakes narratively, but by far the cheapest fix (the parameter already exists; only a loop is missing) — worth doing first as a warm-up / sanity check on the rest.
  3. **Test A (threshold=0.3)** — small code change (thread a `threshold` param through `build_tda_pipeline`/`extract_tda_features`), then ~15 min compute for both cells × 5 seeds.
  4. **Test C (CICIDS)** — confirmed high-value but the largest lift of the four: needs `run_lens4_baseline.py` restructured to accept a dataset loader (precedent exists in `run_multi_seed.py`), plus CICIDS's much larger per-run cost (Phase R noted even a `wc -l` on the raw CSV took 46s; full extraction will be materially slower than UNSW's ~52s).
  5. **SG-RF / L_unmatched 5-seed completion** — code exists, just needs the sweep extended; SG-RF alone is ~2.7 hrs of compute (4 remaining seeds × ~40 min GA generation each) — the single biggest compute cost of any orphan on this list, despite needing zero new code.
  6. **Step 0 (count-invariance)** and **compute-matched random control** — both genuinely orphaned with no shortcut; lowest priority of the substantive ones since they're explanatory/mechanism-confirming rather than headline results.
- **Blockers for Phase F (feature-matrix caching + purity instrumentation):** none introduced by this phase — `paths.py` if anything removes a blocker (results/models/data no longer need machine-specific workarounds). The pre-existing blockers Phase R already identified stand unchanged: no on-disk `X_tda` caching exists yet, and per-cluster poison fractions still aren't persisted anywhere (W3 confirms this is unrelated to the Test A–C gaps — even a fully "reproducible" cell like S doesn't persist per-cluster detail, only aggregate capture).
- **Also unresolved right now:** git has no identity configured on this container, so **nothing from this phase is committed yet** — everything is staged/on-disk only. `git config --global user.name`/`user.email` need to be set (not something I'll do myself) before any of this phase's changes can be committed.
