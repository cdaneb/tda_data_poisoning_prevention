# Phase R Completion Report — WIRE Migration Verification + Read-Only Reconnaissance for Tier 0

## R0 — Environment inventory

- `nproc`: 128. RAM: 1.0Ti total, 830Gi free. `python3 -V`: 3.12.3. **No venv/conda was active** on first inspection — system `/usr/bin/python3` only, and `giotto-tda`/`gudhi` were absent even from system packages.
- `requirements.txt` verbatim:
  ```
  giotto-tda>=0.6.0
  scikit-learn>=1.3.0
  numpy>=1.24.0
  pandas>=1.5.0
  matplotlib>=3.7.0
  joblib>=1.3.0
  ```
- **No `venv312` existed in the fresh clone.** I created one (`python3 -m venv venv312`, matching the project's documented naming) and ran `pip install -r requirements.txt`, since without it R0's version report would just read "everything missing" and R3 couldn't run at all — this is environment provisioning, not "running the pipeline." Installed versions vs. pins:

  | Package | Installed | Pin | Verdict |
  |---|---|---|---|
  | giotto-tda | 0.6.2 | `>=0.6.0` | matches |
  | gudhi | **missing** | not in requirements.txt | **expected** — see below |
  | scikit-learn | 1.3.2 | `>=1.3.0` | matches (pip's resolver capped it below the system's 1.9.0 to satisfy giotto-tda's constraint) |
  | numpy | 1.26.4 | `>=1.24.0` | matches (capped below system's 2.4.6) |
  | scipy | 1.17.1 | not pinned (transitive) | n/a |
  | joblib | 1.5.3 | `>=1.3.0` | matches |
  | pandas | 3.0.5 | `>=1.5.0` | matches |

  **On `gudhi`:** the standalone `gudhi` PyPI package is not installed and is not a dependency anywhere in this project. `giotto-tda`'s `CubicalPersistence` docstring references GUDHI as the algorithm's intellectual lineage, but the actual installed backend is `giotto-ph` (confirmed via `pip show giotto-tda` → `Requires: giotto-ph, igraph, ipywidgets, joblib, numpy, plotly, pyflagser, scikit-learn, scipy`). So "gudhi: missing" is the **correct, expected** state, not a problem.
- `python verify_env.py` inside the new venv: **PASSED**, verbatim tail: `giotto-tda: all required classes imported successfully` / `=== Environment verification PASSED ===`.
- Outbound network: works (`pip install` succeeded downloading from PyPI; `pip index versions giotto-tda` also succeeded).
- **No version mismatches found** once the venv was built as documented. The only "mismatch" was the pre-existing absence of any environment at all.

## R1 — Data wiring

- `data_loader.py` expects `DATA_DIR = Path(r"C:\TDA\data")` with files `Payload_data_UNSW.csv` and `Payload_data_CICIDS2017.csv`, each `usecols=[payload_byte_1..1500, "label"]`.
- **Critical detail (inference, verified empirically):** on POSIX, `Path(r"C:\TDA\data")` is *not* an absolute Windows path — backslash isn't a separator on POSIX, so it parses as a single relative path component literally named `C:\TDA\data` (confirmed: `Path(r'C:\TDA\data').parts == ('C:\\TDA\\data',)`, `is_absolute() == False`). It resolves relative to whatever directory a script is run from.
- Dataset inspection:
  - `~/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv` — 1505 columns (1500 `payload_byte_N` + `ttl, total_len, protocol, t_delta, label`), exactly matching `data_loader.py`'s expectation. 79,882 lines (79,881 data rows) — matches CLAUDE.md's documented "79,881 samples" exactly.
  - `~/wire/DataSets/PayloadByte_CICIDS17/Payload_data_CICIDS2017.csv` — same 1505-column schema. 1,410,256 lines (1,410,255 data rows) — matches CLAUDE.md's "1.4M+ samples." Its `flask_metadata.json` says `"rows": "3119345", "cols": "84"` — stale/wrong relative to the actual CSV; noted as a discrepancy, not something I touched.
  - Raw `~/wire/DataSets/UNSW-NB15/` and `~/wire/DataSets/CICIDS2017/` are flow-level (Argus/BRO/pcap/`UNSW-NB15_N.csv` with columns like `srcip, sport, dstip, ...`) or CICIDS's `MachineLearningCSV` flow-feature CSVs — confirmed by sampling headers; neither contains `payload_byte_*` columns. **Not usable** for this pipeline, which needs per-packet payload bytes.
  - Conclusion: `PayloadByte_UNSW` / `PayloadByte_CICIDS17` are the only directories in the expected format.
- **Wiring performed:** `data_loader.py` has no env-var override, so I could not use one. I created a directory literally named `C:\TDA\data` in the repo root (a single oddly-named POSIX directory, not a nested path) and placed two symlinks inside it pointing at the real files on the read-only WIRE mount:
  ```
  C:\TDA\data/Payload_data_UNSW.csv -> /home/jovyan/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv
  C:\TDA\data/Payload_data_CICIDS2017.csv -> /home/jovyan/wire/DataSets/PayloadByte_CICIDS17/Payload_data_CICIDS2017.csv
  ```
  This resolves `data_loader.py`'s hardcoded relative path with **zero tracked-source changes**, as long as scripts run with cwd = repo root (which is how the project's own instructions already invoke them). I added `C:\\TDA\\data/` to `.gitignore` (backslashes must be escaped there — an unescaped `C:\TDA\data/` pattern silently fails to match, verified with `git check-ignore`) so the symlink directory can never be accidentally committed. This is the only tracked-source edit made in this phase, and it's exactly the one R1.4 anticipated.
  - Same hardcoded-relative-path pattern also affects `run_lens4_baseline.py` (`RESULTS_DIR`/`MODELS_DIR` both `Path(r"C:\TDA\...")`) and `run_multi_seed.py`/`run_iterative.py` (`RESULTS_DIR`). Not wired in this phase — R3's job (R60/malicious_random_attack) doesn't touch surrogate models or write results, so it wasn't needed. Flagging it now so it's not a surprise later: a genuine re-run of any full driver script will need the same treatment for `results/` and `models/`.

## R2 — Artifact inventory

- `results/` (11 files, all tracked in git — see below), `figures/` (9 PNGs), `models/` (4 `.joblib` files), `archive/` (3 audit/summary docs). Full listing captured; sizes ranged 6.8KB–266KB for JSON, ~100–450KB for figures, ~1–4.7MB for models.
- **Discrepancy from CLAUDE.md:** the doc states "`.gitignore` excludes `results/*.json`." The actual `.gitignore` has that line **commented out** (`#results/`), and `git ls-files results/` shows all 11 JSON files are tracked — consistent with `git log`'s `5d8aa6b added results folder` commit. CLAUDE.md is stale on this specific point.
- Coverage found:
  - **Lens 4** (`lens4_baseline_seed42_ladder.json`): variants `L, R60, G60-MLP, G60-RF`, seed 42 only, all 4 clustering algorithms, UNSW-NB15 only.
  - **Lens 4 multi-seed** (`lens4_baseline_multiseed.json`): variants `R60, G60-MLP` only, across seeds 42/123/456/789/1024.
  - **Iterative** (`iterative_results_unsw_nb15.json`, `iterative_results_cicids2017.json`): seed 42 only per CLAUDE.md (CICIDS one is explicitly a Lens 2 timing artifact, `max_iterations=1`, not a full run), all 4 algorithms.
  - **Multi-seed** (`multi_seed_per_seed_*.json`, `multi_seed_aggregated_*.json`): both datasets, all 5 seeds, all 4 algorithms, baseline + iterative.
  - **No trace anywhere** of the task's expected naming: cells `N/S/SG/SG-RF/NS/NSG`, threshold `0.3` as an actual run config, or permutation families "transpositions / block reversal / block swap / cyclic shift" (`grep` across all `.py`/`.md` files found nothing). **This looks like a terminology mismatch with a different/future framing, not a missing artifact.** The actual attack taxonomy in this codebase is `L` (legacy, delegates to `poison.py`), `R60` (malicious-only, one random 60-byte swap, no guidance), `G60-MLP`/`G60-RF` (GA-guided 60-byte swap). Binarizer threshold is hardcoded to `0.4` everywhere — `0.3` appears only in `cc_summary.md` as a note about Monkam's own pseudocode-vs-prose inconsistency, never as an actual alternate run. **I'm treating the task's "cell S" as `R60`**, because the task's own recorded reference value (5-seed mean 1.80%±0.51%) is a verbatim match for `R60`'s documented figure in CLAUDE.md — this identification is confirmed by the R3 result below, not just assumed.
- `git log --oneline -15`: `5d8aa6b added results folder`, `ea67b83 Added Documentation to Migrate to WIRE`, `43038bc Lens 4 Feasibility Tests`, `6179d9f Updated Poisoning Attack, Lens 4 Implementation`, `6a5f441 Replace per-sample amplitude-difference approximation with a true...`, `3b0fc42 Codebase Audit`, `b89373c Initial commit`.
- `git status --short` at phase start: clean.

## R3 — BLOCKING REPRODUCTION GATE ✅ PASSES (exact match)

- **Command:** a scratchpad-only orchestration script (not added to the repo) that calls the exact existing functions `run_lens4_baseline.py` itself uses — `load_unsw()`, the same `subsample_for_seed` logic (`RandomState(42).choice(..., 5000)`), `malicious_random_attack(..., poison_rate=0.10, random_state=42, n_swaps=60)` (= cell "S"/R60), `extract_tda_features()`, `run_all_clustering()`, `classify_clusters()`. No surrogate models loaded (R60 doesn't use them) and none retrained. Full script content available on request; I avoided writing it into the repo per the "no new analysis code / no tracked-source changes" rule.
- **Wall-clock:** 68.2s total (`real 1m8.154s`) — TDA extraction 51.8s, clustering (all 4 algos) 7.5s, attack generation 0.1s, data load ~9s.
- **Capture % (OPTICS):** **2.2000%**.
- **Comparison:** the seed-42 value recorded in `results/lens4_baseline_seed42_ladder.json` for `R60`/OPTICS is `2.1999999999999997` — **floating-point-identical**. Further, the 5 per-seed values in `results/lens4_baseline_multiseed.json` (`2.2, 2.2, 2.2, 1.0, 1.4`) average to exactly **1.80%**, matching the documented 1.80%±0.51% precisely.
- **Gate: PASSES — environment reproduces, exactly, not just within spread.**

## R4 — Is per-sample cluster assignment recoverable?

- Function: `clustering.py::classify_clusters(cluster_labels, is_poisoned)` — computes both a per-cluster `cluster_info` list (id, color, size, n_poisoned, poison_fraction, dpdc) and an aggregate `summary` (green_pct, red_pct, red_poison_capture_pct, per-color **counts**).
- Traced every `json.dump` call site (`run_multi_seed.py`, `run_lens4_baseline.py`, `run_iterative.py`) and every use of `cluster_info`:
  - **Cluster label vector:** never persisted anywhere — `run_all_clustering()`'s output array is consumed by `classify_clusters()` in the same function scope and discarded.
  - **Poison ground-truth mask:** never persisted as an array. (One exception with a caveat: `run_iterative.py` persists `sanitized_indices`/`poisoned_pool_indices`/`residual_indices` — index partitions into the seed-42, 5000-sample subsample — and since `poison_dataset`'s default `random_state=42` is what `run_iterative.py` actually calls, the ground-truth mask *is* cheaply re-derivable for that one results file, without re-running TDA/clustering. This does not extend to `lens4_baseline_*.json` or `multi_seed_*.json`, which save no indices at all.)
  - **Per-cluster composition:** only **per-color cluster counts** (`colors: {Green: 27, Red: 0, Yellow: 15, ...}`) and **pooled** percentages (`green_pct`, `red_pct`, `red_poison_capture_pct`) are saved — never each individual cluster's own `size`/`poison_fraction`. `iterative_results_*.json`'s `iteration_log` adds per-iteration counts (`n_green_clusters`, `n_yellow_clusters`, etc.) but again no per-cluster fractions.
- **Verdict: (c) Not recoverable — only aggregate capture (plus, in the iterative results only, pool-membership indices and per-iteration cluster counts) is saved. A purity-threshold sweep requires knowing each individual Yellow/Pink cluster's own poison fraction, which is computed in memory and discarded every time — never written to disk anywhere in this codebase. Re-run required.**
- Quoted JSON keys (from `results/lens4_baseline_seed42_ladder.json`, variant `R60`): `{"red_poison_capture_pct": ..., "green_pct": ..., "red_pct": ..., "n_clusters": ..., "colors": {...}, "sanitized_purity": ..., "poisoned_pool_precision": ...}` — no `cluster_info`-level detail present.
- **Purity thresholds:** `clustering.py::classify_clusters()`, lines 105–112 — `poison_fraction == 0` → Green, `== 1.0` → Red, `> 0.80` → Pink, else Yellow. **Hardcoded literals, not parameterized** — no threshold argument exists on the function at all.

## R5 — Persistence diagram availability

- **`CubicalPersistence`**: instantiated in `tda_pipeline.py::build_tda_pipeline()`, one instance per filtration (5 total), each embedded as one step inside an sklearn `make_pipeline(Binarizer, Filtration, CubicalPersistence, Scaler)`, itself inside `make_union(...)`. Because the whole thing runs as a single `pipeline.fit_transform()` call in `extract_tda_features()`, the intermediate per-sample cubical diagrams are **never bound to a variable, never returned, never persisted** — they flow directly into `Scaler`→feature-union and vanish. Only the final `(N, 60)` feature matrix is returned.
- Raw reshaped `(N, 30, 50)` images (`reshape_for_tda`) and binarized images: **not persisted anywhere** — the reshaped array is a local variable in `extract_tda_features()`, discarded after the function returns; the binarized version isn't even named (internal pipeline step).
- **`compute_whole_residual_diagram(X_tda_residual)`** — `iterative_filter.py:18`. Called once per iteration inside `iterative_filter()`'s loop, on that iteration's already-extracted `(n_residual, 60)` TDA feature matrix (not raw bytes). **`PairwiseDistance`** — one call site, `iterative_filter.py:75`, inside `wasserstein_distance_between_diagrams()`, operating on two `VietorisRipsPersistence` outputs.
- **These are genuinely different objects, not to be conflated:**
  - `CubicalPersistence` → per-packet-image diagrams (5 filtrations × per sample) inside `tda_pipeline.py`'s feature extractor. **This is what Tier 0.2 needs**, and per above, it is currently **discarded, not retained**.
  - `VietorisRipsPersistence` (`iterative_filter.py`) → one diagram over the *whole residual*, treating the 60-dim TDA feature vectors as a point cloud. Descriptive-only, only touched by `iterative_filter()` (so only `run_iterative.py`/`run_multi_seed.py` call it; `run_baseline.py`/`run_lens4_baseline.py` never do, since they're single-pass).
- Can existing distance machinery point at cubical diagrams as-is? **Structurally yes** — `gtda.diagrams.PairwiseDistance` operates on gtda's standard `(n_samples, n_features, 3)` diagram-array format regardless of which transformer produced it, and `CubicalPersistence`'s output is in that same format. But **new code is required** to actually *capture* those diagrams before they're consumed by `Scaler`/`Amplitude` — today's `make_pipeline` construction doesn't expose the intermediate step output. (Also worth flagging for whoever designs Tier 0.2, without designing it here: there are 5 separate `CubicalPersistence` instances, one per filtration — "the" per-sample cubical diagram is actually 5 diagrams, a design decision Tier 0.2 will need to resolve.) No such code was written in this phase.

## R6 — Cost model for re-runs

- **One (cell, seed) run, measured on WIRE (R3):** 68.2s for `R60` (cheap-generation variant), dominated by TDA extraction (51.8s of 68.2s). Older recorded numbers from `results/lens4_baseline_seed42_ladder.json` (prior machine): `L`≈56s, `R60`≈43s, `G60-MLP`≈136s (93.5s attack-gen + 34s TDA + 9s cluster), **`G60-RF`≈2415s / ~40 min** (2365.9s of that is GA attack generation alone — RF's `predict_proba` batching overhead, as documented in `adversarial_attack.py`'s comments). These are read from existing JSON, not re-measured this phase.
- **Full 5-seed factorial, UNSW, as `run_lens4_baseline.py` is *actually coded* today** (L+R60+G60-MLP+G60-RF at seed 42; R60+G60-MLP swept across the other 4 seeds; G60-RF and L *not* seed-swept in the existing script): ≈3,580s (~60 min), of which the single seed-42 `G60-RF` run alone is ~66% of total. **If instead all 4 variants were extended across all 5 seeds** (a genuine full factorial, which the current script does not do), `G60-RF`×5 alone ≈ 3.4 hours, pushing total to roughly 4–5 hours. Iterative single-seed UNSW (`run_iterative.py`, all 4 algorithms): measured 657.6s (~11 min) from `iterative_results_unsw_nb15.json`'s `total_time_s` fields. Full `run_multi_seed.py` (both datasets ×5 seeds, baseline+iterative) is not measured this phase (out of scope — R3 is the only execution allowed) — extrapolating from the UNSW iterative figure and CICIDS's ~10x larger per-seed sample cap (`max_samples=50000` vs. 5000), a multi-hour run is consistent with CLAUDE.md's own "(~hours)" description, but this is **extrapolation, not measurement**.
- **Is clustering separable from feature extraction? Yes, in-memory, already.** `extract_tda_features()` runs once and `run_all_clustering()` reuses that same `X_tda` across all 4 algorithms within a single script invocation (see `run_lens4_baseline.py::run_single_pass_baseline`). **But there is no on-disk caching of `X_tda` anywhere in the codebase** — no `.npy`/`.npz` save/load of intermediate feature matrices exists. This is the key gap for everything downstream (purity sweep, diagram distances, more seeds): today, re-running clustering with a different purity threshold still requires paying the ~35–52s TDA-extraction cost per (seed, cell), because nothing persists the feature matrix across process invocations. Adding that caching layer is a real, identifiable code change — not made in this phase.
- **Parallelizable across (seed, cell)?** RNG is properly scoped locally everywhere I checked (`np.random.RandomState(seed)` / `np.random.default_rng(random_state)` passed explicitly, no bare `np.random.seed()` global calls found) — no shared-mutable-RNG blocker. **But output file paths are fixed, not seed/cell-namespaced** (`RESULTS_DIR / "lens4_baseline_multiseed.json"`, `iterative_results_unsw_nb15.json`, etc.) — naively launching parallel processes per seed today would clobber each other's output file. That would need parameterized filenames (or a merge step) before parallel execution is safe — a real, identifiable blocker, not fixed here.

## R7 — Feature-count arithmetic

- `tda_pipeline.py::build_tda_pipeline()`: **5 filtrations** (`direction_list = [np.array([0,1]), np.array([1,0])]` → 2 `HeightFiltration`; `center_list = [np.array([0,50]), np.array([0,25]), np.array([30,0])]` → 3 `RadialFiltration`; `2+3=5`) × **6 extractors per filtration** (`PersistenceEntropy` + 5×`Amplitude` — metrics: bottleneck, wasserstein, landscape, betti, heat) × **2 homology dimensions** (`CubicalPersistence` default `homology_dimensions=(0,1)`) = **60**, confirmed both arithmetically and empirically (R3's `X_tda.shape == (5500, 60)`).
- **It is 60, not 72.** `cc_summary.md` already documents this exact same 60 = 5×6×2 derivation and flags the paper's 72-feature claim as an open, unconfirmed gap (no Monkam PDF is present in this clone — `.gitignore` excludes `*.pdf` as copyrighted, and `find`/`git ls-files` confirm it's absent — so the paper's own filtration count can't be checked in this phase to test any hypothesis about the gap's source).
- Filtration list verbatim: `direction_list = [np.array([0, 1]), np.array([1, 0])]`; `center_list = [np.array([0, 50]), np.array([0, 25]), np.array([30, 0])]`.

---

## Summary

- **Reproduction status: ✅ Reproduces — exactly.** Observed 2.2000% OPTICS capture for cell S (=`R60`), seed 42, UNSW-NB15, matching the recorded value (2.1999999999999997%) to floating-point precision, and the 5 recorded per-seed values average to exactly the documented 1.80%. No version mismatches once the venv was built per `requirements.txt`.
- **Tier 0.1 feasibility: requires re-run.** Per-cluster poison fractions are not persisted anywhere in this codebase's saved JSON — only per-color cluster *counts* and pooled percentages. Cheapest viable path: **add an on-disk cache for the `(N,60)` TDA feature matrix** (R6's key finding — extraction is separable from clustering in-memory already, just not persisted), then clustering + `classify_clusters()` at each of the 5 purity thresholds becomes a cheap (~8–22s) reuse of one cached extraction per (seed, cell) instead of a full ~35–52s re-extraction each time.
- **Tier 0.2 feasibility: requires re-run** (plus new code, not written here) — `CubicalPersistence` diagrams are structurally compatible with the existing `PairwiseDistance` machinery already used for the Lens-2 Wasserstein tracking, but they are currently discarded inside `tda_pipeline.py`'s sklearn pipeline and never retained. Capturing them requires restructuring how that pipeline is invoked (or split), which wasn't done in this read-only phase.
- **Blockers before either proceeds:** (1) no on-disk feature-matrix caching exists — build it first, since both tiers benefit; (2) `results/`/`models/` hardcoded `C:\TDA\...` paths need the same symlink treatment R1 gave `data/` before any full driver script can run on WIRE; (3) Tier 0.2 needs a design decision on how to handle 5 separate per-filtration cubical diagrams per sample (concatenate vs. compare separately) before any distance code is written.
