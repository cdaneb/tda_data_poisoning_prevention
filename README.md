# What Can a Persistence Diagram See? Invariance Structure and the Detectability of Data Poisoning

Christian Dane Beels, advised by Dr. Joseph Dorta — Dept. of Mathematical Sciences, USMA West Point.
Target venue: MAA MathFest 2026, Boston (Aug 5–8).

**`CLAUDE.md` is the authoritative, current-state document for this project** (claim status, results
of record, known errors, open work). This README is an orientation layer on top of it — setup and
"what's in this repo," not a restatement of findings. If anything here conflicts with `CLAUDE.md`,
`CLAUDE.md` wins.

## What this is

The project reconstructs the pipeline from Monkam, De Lucia & Bastian, *Computers & Security* 144
(2024) 103929: map network-packet payload bytes to grayscale images, compute cubical persistence
across several filtrations, and cluster the resulting features to separate poisoned packets from
clean ones before training a NIDS classifier. The reconstruction reproduces the mechanics but
captures only single-digit percentages against the paper's reported 40–70%.

That gap was originally suspected to come from unrealistically weak synthetic poisoning; a
surrogate-guided attack was built to test that and the hypothesis was refuted (guided attacks turned
out to be *more* topologically detectable than magnitude-matched random noise, not less). The project
pivoted to the current question: **what does the relationship between an attack's invariance
structure and a filtration's sensitivities determine about whether poisoning is topologically
detectable at all?** The primary result is deductive: pixelwise binarization is equivariant under
byte-position permutation, and summation is symmetric, so the foreground pixel count is *provably*
invariant under every permutation — a concrete, provable blind spot in this class of detector. See
`CLAUDE.md` §2 for full claim status and §7 for the language conventions used when describing it.

## Repo layout

```
paths.py                    # DATA_DIR/RESULTS_DIR/MODELS_DIR/FIGURES_DIR, env-var overridable
data_loader.py               # load_unsw() / load_cicids() -- Payload-Byte CSVs -> (N,1500) bytes + labels
tda_pipeline.py               # build_tda_pipeline()/extract_tda_features() -- bytes -> 60-dim TDA features
clustering.py                 # run_all_clustering() (DBSCAN/HDBSCAN/OPTICS/MeanShift) + classify_clusters()
poison.py                     # legacy attack: Gaussian noise + uncontrolled 10-50 swap count (do not use
                               #   to isolate noise as a variable -- see CLAUDE.md SS5)
adversarial_attack.py         # current attack substrate: malicious_random_attack, block_reversal_attack,
                               #   block_swap_attack, cyclic_shift_attack, chale_ga_attack (surrogate-
                               #   guided GA search), train_surrogates()
invariance_check.py           # foreground_count/positions_changed/crossed_threshold/max_value_check --
                               #   empirical instrumentation backing the count-invariance proof
iterative_filter.py           # iterative Green/Red cluster removal + Wasserstein convergence tracking
                               #   (descriptive VietorisRipsPersistence on the 60-dim residual -- see
                               #   CLAUDE.md SS4 on the two distinct persistence computations)
results_io.py                 # convert_for_json() -- shared numpy->JSON serialization helper
classifier_eval.py            # downstream NIDS classifier evaluation
visualize.py / visualize_comparison.py   # figure generation from results/
verify_env.py                 # environment sanity check
explore_data.py                # ad-hoc data exploration

run_baseline.py                # single-pass baseline (TDA + clustering, no iteration)
run_iterative.py                # single iterative-filter run
run_multi_seed.py               # multi-seed statistical validation, both datasets
run_lens4_baseline.py            # attribution-ladder runner (L / R60 / G60-MLP / G60-RF variants)
run_test_b_capture.py            # Test B: 4 permutation families x 5 seeds x 4 clustering algorithms
test_b_diagnostics.py             # Step 0 count-invariance gate + bit-identity/effective-swap diagnostics
tools/repro_check.py               # tracked regression test -- see "Reproduction gate" below

models/                        # trained surrogate classifiers (surrogate_mlp*.joblib, surrogate_rf*.joblib)
results/                       # experiment output JSON (seed-namespaced; see CLAUDE.md SS6 for which
                               #   result sets are committed-and-backed vs. orphaned/unreproducible)
figures/                       # generated plots
data/                          # Payload-Byte CSVs (gitignored -- see Datasets below)

docs/                          # historical/reference documents -- PROJECT_HANDOFF_1.md,
                               #   TERMINOLOGY.md, ABC_PHASE_REPORT.md, retired LENS4 instruction files.
                               #   CLAUDE.md SS0 lists exactly which sections of each are stale.
P_PHASE_REPORT.md, R_PHASE_REPORT.md, W_PHASE_REPORT.md   # completion reports for the phases that
                               #   built the current WIRE environment and Test B rebuild (accurate for
                               #   what they report; CLAUDE.md SS9 lists their known errors)
```

## Environment

Two environments are in play:

- **Local (this checkout):** Windows, Python 3.12 venv at `venv312/`. Loose dependency bounds are in
  `requirements.txt`.
- **WIRE** (the compute environment experiments are actually run on -- 128 cores, 1.0 TiB RAM, Python
  3.12.3, CPU-only): pinned exact versions are in `requirements.lock.txt` (giotto-tda 0.6.2,
  scikit-learn 1.3.2, numpy 1.26.4, scipy 1.17.1, pandas 3.0.5, joblib 1.5.3). Install from the lock
  file, not `requirements.txt`, to reproduce recorded results exactly:

  ```bash
  python3 -m venv venv312
  source venv312/bin/activate        # or venv312\Scripts\Activate.ps1 on Windows
  pip install -r requirements.lock.txt
  ```

All data/results/models/figures paths are resolved by `paths.py`, repo-relative by default and
overridable via `TDA_DATA_DIR` / `TDA_RESULTS_DIR` / `TDA_MODELS_DIR` / `TDA_FIGURES_DIR` -- copy
`.env.example` to `.env.wire` and fill in machine-specific paths (e.g. WIRE's NFS mounts), then
`set -a; source .env.wire; set +a` before running a driver script. Nothing loads it automatically.

```bash
python verify_env.py
```

## Datasets

Payload-Byte format CSVs (1505 columns: 1500 `payload_byte_N` + ttl, total_len, protocol, t_delta,
label), not tracked in git:

- **UNSW-NB15** -- 79,881 rows -- `data/Payload_data_UNSW.csv`
- **CICIDS2017** -- 1,410,255 rows -- `data/Payload_data_CICIDS2017.csv`

## Reproduction gate

Run before and after any pipeline change:

```bash
python tools/repro_check.py --expect 2.2000
```

Must return **2.2000% exact**, `X_tda.shape == (5500, 60)` -- seed 42, threshold 0.4, UNSW-NB15,
OPTICS, `malicious_random_attack` with `n_swaps=60`. This is the project's standing regression test
for "does this environment/these path changes still reproduce the recorded number."

## Running experiments

```bash
# Count-invariance gate + Test B diagnostics (cheap; run first)
python test_b_diagnostics.py

# Test B: four permutation families across 5 seeds, all clustering algorithms
python run_test_b_capture.py

# Attribution ladder: legacy vs. magnitude-matched-random vs. surrogate-guided attacks
python run_lens4_baseline.py

# Original baseline / iterative-filter / multi-seed drivers (still runnable, unmodified)
python run_baseline.py
python run_iterative.py
python run_multi_seed.py            # long-running (~hours), both datasets

# Downstream classifier evaluation + figures
python classifier_eval.py
python visualize.py
python visualize_comparison.py
```

## References

- Monkam, G. F., De Lucia, M. J., & Bastian, N. D. (2024). A topological data analysis approach for
  detecting data poisoning attacks against machine learning based network intrusion detection systems.
  *Computers & Security*, 144, 103929.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2005). Stability of persistence diagrams. *SoCG
  '05*, Pisa, 263-271.
- Ferrara, M. (2025). *AIMS Mathematics*, 10(7), 15457-15475.
- Zheng, S., Zhang, Y., Wagner, S., Goswami, M., & Chen, C. (2021). Topological detection of trojaned
  neural networks. *NeurIPS 2021*.
- Garin, A., & Tauzin, G. A topological reading lesson: Classification of MNIST using TDA.
  arXiv:1910.08345.
- Sethi, T. S., & Kantardzic, M. (2018); see also arXiv:1802.07295.

See `CLAUDE.md` §10 for the full prior-art list and how each source is positioned relative to this
work (GENEO in particular -- cite it whenever using the phrase "invariance structure").

## Acknowledgments

This research builds upon the TDA pipeline and methodology introduced by Monkam, De Lucia & Bastian
(2024) at the Army Cyber Institute, United States Military Academy.
