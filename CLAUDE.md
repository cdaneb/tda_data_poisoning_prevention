# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research implementation of **"A Topological Data Analysis Approach for Detecting Data Poisoning Attacks Against Machine Learning Based Network Intrusion Detection Systems"** (Monkam et al., Computers & Security 144, 2024). The pipeline detects poisoned network traffic data by extracting topological features via giotto-tda and clustering them with unsupervised algorithms.

The project has since grown beyond reproduction into a comparison/extension study: it tests whether concepts from Ferrara (2025) — a Topological Vulnerability Index, per-point topological influence, and a game-theoretic (minimax) defense framing — can improve detection over Monkam et al.'s binary purity-threshold clustering baseline. This work is organized as four "lenses," each tracked in a dedicated instruction file as it's investigated/implemented (LENS1_TOPOLOGICAL_INFLUENCE_INSTRUCTIONS.md, LENS2_WASSERSTEIN_INSTRUCTIONS.md, LENS3_MINIMAX_PHASE1_INSTRUCTIONS.md, LENS4_ADVERSARIAL_SUBSTRATE_PHASE1/2/3_*.md). See "In-Progress Investigations" below for current status. **Lens 4 landed a faithful-as-possible Monkam-style attack substrate + attributed baseline — not a reproduction of Monkam's results; see below for the documented divergences and why the baseline capture stays well under Monkam's reported range.**

## Environment Setup

All commands use the local virtual environment:

```powershell
# Activate venv (PowerShell)
.\venv312\Scripts\Activate.ps1

# Verify environment
python verify_env.py
```

Dependencies are pinned in `requirements.txt`.

## Running Experiments

```powershell
# Phase 1: Baseline TDA + clustering on UNSW-NB15
python run_baseline.py

# Phase 2: Iterative filtration on UNSW-NB15
python run_iterative.py

# Phase 3: Multi-seed (5 seeds) on both datasets — long-running (~hours)
python run_multi_seed.py

# Downstream classifier evaluation
python classifier_eval.py

# Visualizations (generate figures/ from results/)
python visualize.py
python visualize_comparison.py
```

There are no automated tests. Correctness is validated by checking results JSON files and generated figures.

## Architecture

### Data Flow

```
CSV (1500 payload byte features)
  → data_loader.py          # loads UNSW-NB15 (79,881 samples) or CICIDS2017 (1.4M+ samples)
  → poison.py               # injects 10% Gaussian-noise + byte-swap poisoning
  → tda_pipeline.py         # 1500→60 features (Algorithm 1 from paper)
  → clustering.py           # 4 algorithms → color-coded cluster labels
  → iterative_filter.py     # iterative removal of Green/Red clusters (sanitized/poisoned pools) until convergence
  → results/*.json          # metrics per seed and aggregated mean±std
  → figures/*.png           # convergence curves, comparison plots, summary table
```

### Key Modules

**`tda_pipeline.py`** — Core TDA feature extraction. Reshapes (N,1500) byte arrays to (N,30,50) images, applies 5 filtrations (2 HeightFiltration + 3 RadialFiltration) via giotto-tda, extracts 6 persistence metrics each (bottleneck, wasserstein, landscape, betti, heat, entropy) → (N,60) feature matrix. (The source paper claims 72 features; the code's own `__main__` block checks for this and warns on mismatch, attributing the gap to giotto-tda version/configuration differences.)

**`clustering.py`** — Runs DBSCAN, HDBSCAN, OPTICS, and Mean Shift on the 60-feature matrix. Assigns semantic colors to clusters based on poison fraction: Green (0% poisoned), Red (100% poisoned), Pink (>80% poisoned), Yellow (mixed). Computes DPDC, sanitized purity, and poisoned pool precision.

**`iterative_filter.py`** — Iteratively extracts TDA features, removes Green clusters into a "sanitized pool" and Red clusters into a "poisoned pool," and repeats on the residual. Stops when no Green/Red clusters remain, the residual drops below 10 samples, or `max_iterations` is reached — the stopping condition does not reference topological distance. Each iteration also computes a real Wasserstein distance (`gtda.diagrams.PairwiseDistance`, p=1) between the current and previous iteration's whole-residual persistence diagram — one `VietorisRipsPersistence` diagram per iteration over that iteration's 60-dim TDA feature vectors treated as a single point cloud, per Cohen-Steiner stability / Ferrara (2025) Eq 3.1/5.1. This value is logged and plotted for convergence tracking only; it is not (yet) load-bearing in the stopping condition. Returns sanitized pool, poisoned pool, and residual.

**`data_loader.py`** — Extracts `payload_byte_1` through `payload_byte_1500` columns; maps attack category strings to binary poison labels.

**`poison.py`** — Default 10% poison rate; Gaussian noise perturbation + byte swapping; deterministic via `random_state`. Unmodified by Lens 4 — it remains the reference `adversarial_attack.py`'s `random_swap` mode is checked against for bit-for-bit equivalence.

**`adversarial_attack.py`** — [Lens 4] Surrogate-guided (Chale-approximation) attack substrate, additive to `poison.py`, not a replacement for it (no existing driver is rewired to use it). Three attack modes, all returning the same `(X_combined, y_combined, is_poisoned)` contract as `poison_dataset`: `random_swap` (delegates to `poison.poison_dataset` verbatim), `malicious_random_attack`/"R60" (malicious-class-only targeting, one random 60-byte-swap set per sample, no guidance — isolates perturbation magnitude + targeting from guidance), and `chale_ga_attack`/"G60" (malicious-only targeting, a generational search over 60-byte-swap sets maximizing a surrogate classifier's benign-class probability; population=50, generations=100, early-stops at benign-probability≥0.5). Also trains and exposes two surrogate NIDS classifiers (`train_surrogates`): a primary `MLPClassifier` (chosen over a torch CNN to avoid a large new dependency in this pinned env — a `pip install torch --dry-run` showed no conflicts, but the surrogate's fidelity is documented effort, not a claim, per the Lens 4 framing) and a secondary `RandomForestClassifier` for surrogate-sensitivity checks (kept a separate instance from `classifier_eval.py`'s evaluator). Trained surrogates are saved to `models/surrogate_mlp.joblib` / `models/surrogate_rf.joblib`. Byte-swap moves preserve value range and multiset by construction, so validity is structural (confirmed empirically at 100% across all runs, not just assumed).

**`run_lens4_baseline.py`** — [Lens 4] Single-pass Monkam-style baseline runner (`run_all_clustering` + `classify_clusters`, unmodified 100%-purity rule, no iteration) against the attack variants above, UNSW-NB15 only. Does not modify or import from `run_baseline.py`/`run_iterative.py`/`run_multi_seed.py`. **Capture-rate finding (OPTICS; DBSCAN/HDBSCAN/MeanShift stay ~0 across every variant): the legacy attack (L) captures 6.6%; a magnitude/targeting-matched random control (R60) drops to 2.2%; adding surrogate guidance (G60-MLP) recovers to 6.2% (multi-seed: R60 1.80%±0.51% vs. G60-MLP 6.48%±1.24%, a +4.68±1.57 pt delta, positive in all 5 seeds).** Guided evasion is consistently *more* topologically detectable than magnitude-matched random noise — evidence against an adaptive attacker quietly evading this TDA defense, at least for this attack/surrogate pair. Capture is **not stable across which surrogate guided the attack** (G60-MLP 6.2% vs. G60-RF 11.8% at seed 42, despite RF succeeding at evasion far less often — 17.6% vs. 83.8% flip rate): a real substrate fragility, not yet explained (the RF-produces-larger-perturbations-when-it-does-move hypothesis is unverified). **No variant reaches Monkam's reported ~40–70% capture neighborhood** (best case 11.8%) — this is reported as a genuine, unresolved gap, not minimized.

### Results Layout

- `results/multi_seed_per_seed_<dataset>.json` — per-seed detailed metrics
- `results/multi_seed_aggregated_<dataset>.json` — mean ± std across 5 seeds (42, 123, 456, 789, 1024)
- `results/iterative_results_unsw_nb15.json` — single-seed (42), all-4-algorithm iterative run (source for `visualize.py`'s figures)
- `results/iterative_results_cicids2017.json` — **not a full run**: a Lens 2 verification artifact only (`max_iterations=1`, used solely to time the whole-residual diagram computation on real CICIDS2017 data). CICIDS2017 is not otherwise in `run_iterative.py`'s scope — its iterative results live only in the multi-seed files above.
- `figures/baseline_vs_iterative_capture.png` — headline comparison figure
- `figures/summary_table.png` — poster-ready results table
- `results/lens4_baseline_seed42_ladder.json` — [Lens 4] seed-42 attribution ladder (L, R60, G60-MLP, G60-RF), all 4 algorithms
- `results/lens4_baseline_multiseed.json` — [Lens 4] R60 and G60-MLP across all 5 seeds
- `results/lens4_baseline_full.json` — [Lens 4] combined output of both of the above
- `models/surrogate_mlp.joblib`, `models/surrogate_rf.joblib` — [Lens 4] trained surrogate NIDS classifiers, reused (not retrained) across seeds

### Phase Progression

The three instruction files (`cc_instructions_1/2/3.md`) document the original build-out:
- Phase 1: baseline pipeline
- Phase 2: iterative filtration algorithm
- Phase 3: multi-seed validation on both datasets + comparative visualizations

## In-Progress Investigations

**Lens 1 (per-point topological influence):** Phase 1 (feasibility investigation) is complete — see `LENS1_TOPOLOGICAL_INFLUENCE_INSTRUCTIONS.md`. Key findings: Yellow/Pink cluster sizes vary hugely by algorithm (Mean Shift produces near-whole-residual "clusters" and has been excluded from this lens's scope on methodological, not just compute, grounds); the originally proposed per-cluster sigma design fails on real data (most Yellow clusters have near-zero internal variance in most of the 60 feature dimensions) and is being redesigned: a pure whole-residual-anchored sigma was also tested and found to be 7–70× oversized relative to real clusters' own point spacing, so neither the original per-cluster-std nor a simple global-std anchor works as specified — a redesign (e.g. no perturbation in degenerate dimensions, or a within-cluster subspace approach) is still pending. **No repository code has been modified for this work yet.** Pending: a decision on local vs. remote (VM) execution for the full-scale diagnostic run, and confirmation of the redesigned sigma, before Phase 2 (diagnostic validation) begins.

**Lens 3 (game-theoretic minimax defense):** Phase 1 (read-only feasibility investigation) complete — see `LENS3_MINIMAX_PHASE1_INSTRUCTIONS.md`. No code written. Recommends Reading T (threshold relaxation, reusing the `cc_summary.md` §10.A sweep as an honest non-game-theoretic baseline) over Reading D (Ferrara-faithful but no concrete `D_α` transformation exists yet in this codebase); the "game" is vacuous either way until Lens 4 supplies an adaptive attacker — not yet decided/started as an implementation.

**Lens 4 (adversarial substrate — Chale-approximation attack + attributed Monkam-style baseline):** Phases 1–3 complete and landed — see `LENS4_ADVERSARIAL_SUBSTRATE_PHASE1/2/3_*.md` and `adversarial_attack.py` / `run_lens4_baseline.py` above for what was built. **This is a documented-effort substrate, not a reproduction of Monkam et al.'s results** — no variant reaches their reported ~40–70% capture neighborhood (best case 11.8%, via G60-RF). The headline finding: guided (surrogate-loss-optimized) evasion is consistently *more* detectable by the TDA pipeline than magnitude-matched random perturbation (R60 1.80%±0.51% → G60-MLP 6.48%±1.24% OPTICS capture, multi-seed, sign stable across all 5 seeds) — evidence against an adaptive attacker evading this defense, for this attack/surrogate pair specifically. Two open caveats carried into any future lens building on this substrate: (1) capture is **not stable across which surrogate guided the attack** (6.2% MLP vs. 11.8% RF at seed 42 — nearly 2×, mechanism not yet confirmed); (2) Hore et al.'s deep-RL attack was deliberately deferred (Phase 1 decision), not built — this substrate covers Chale-style byte-swap evasion only. Lens 3 and Lens 1's future work should treat this substrate's capture numbers as the new baseline reference in place of the original `poison.py`-only figures, with these caveats attached.
