# What Can a Persistence Diagram See? Invariance Structure and the Detectability of Data Poisoning

Christian Dane Beels, advised by Dr. Joseph Dorta — Dept. of Mathematical Sciences, USMA West Point.
Target venue: MAA MathFest 2026, Boston (Aug 5–8).

## What this is

The project reconstructs the pipeline from Monkam, De Lucia & Bastian, *Computers & Security* 144
(2024) 103929: map network-packet payload bytes to grayscale images, compute cubical persistence
across several filtrations, and cluster the resulting features to separate poisoned packets from
clean ones before training a NIDS classifier. The reconstruction reproduces the mechanics but
captures only single-digit percentages against the paper's reported 40–70%.

That gap was originally suspected to come from unrealistically weak synthetic poisoning. A
surrogate-guided attack was built to test it, and the hypothesis was refuted: guidance raised capture
on UNSW-NB15 but came nowhere near closing the gap, and the effect does not replicate on CICIDS2017
(−0.60 ± 3.03, 2/5 seeds positive — sign flips per seed, mean crosses zero). The gap is structural.

The project pivoted to the current question: **what is it about the interaction between an attack's
invariance structure and a filtration's sensitivities that determines whether poisoning is
topologically detectable?**

The primary result is deductive. Binarization is pointwise, hence *equivariant* under byte-position
permutation — permuting then binarizing equals binarizing then permuting. Summation is symmetric, so
the foreground pixel count is *invariant*: unchanged under every permutation, on every input.
(The two words name different things and the distinction is load-bearing; the binarized image moves,
the count does not.) The threshold is a fraction of a *fitted* maximum, i.e. data-dependent — the
argument survives because a maximum is itself permutation-invariant, and this is verified in code
rather than assumed.

Consequence: the foreground-count channel is **provably insensitive** to permutation attacks, and
whatever detection survives must flow through the position-dependent Height and Radial filtrations.
This is a blind spot *arising from the feature map's invariances*; the algebra does not characterize
every failure mode of the detector.

The primary empirical evidence is Test B — four permutation families, 5 seeds, UNSW-NB15, threshold
0.4:

| Family | Count change | Mean pos. changed | % zero footprint | Capture % |
|---|---|---|---|---|
| Transpositions (60) | 0/200 | 22.93 | 12.5% | 1.80 ± 0.51 |
| Block reversal (k=120) | 0/200 | 14.08 | 84.5% | **0.00 ± 0.00** |
| Block swap (2×k=60) | 0/200 | 17.63 | 78.5% | **0.00 ± 0.00** |
| Cyclic shift | 0/200 | 281.09 | 0.0% | 6.28 ± 1.31 |

Detectability is not ordered by attack realism; it tracks **spatial disruption**. Unguided cyclic
shift (6.28%) is statistically indistinguishable from surrogate-guided search (6.48 ± 1.24). Count
invariance is confirmed at 1,600 checks (200 packets × 4 families × 2 thresholds) with zero
exceptions.

See `CLAUDE.md` §2 for full claim status, §6 for results of record, and §7 for the language
conventions used when describing all of this. In particular: never "invisible," "evades," or
"defeats" — the phrasing is *attenuated on the foreground-count channel*, because the highest capture
observed under any committed configuration is 6.48%, not zero.

## Repo layout

```
paths.py                    # DATA_DIR/RESULTS_DIR/MODELS_DIR/FIGURES_DIR, env-var overridable
data_loader.py               # load_unsw() / load_cicids() -- Payload-Byte CSVs -> (N,1500) bytes + labels
tda_pipeline.py               # build_tda_pipeline()/extract_tda_features() -- bytes -> 60-dim TDA features
                               #   (60, not the 72 the source paper claims -- see CLAUDE.md §4)
clustering.py                 # run_all_clustering() (DBSCAN/HDBSCAN/OPTICS/MeanShift) + classify_clusters()
poison.py                     # legacy attack: Gaussian noise + uncontrolled 10-50 swap count (do not use
                               #   to isolate noise as a variable -- see CLAUDE.md §5)
adversarial_attack.py         # current attack substrate: malicious_random_attack, block_reversal_attack,
                               #   block_swap_attack, cyclic_shift_attack, chale_ga_attack (surrogate-
                               #   guided GA search), train_surrogates()
invariance_check.py           # binarize/foreground_count/positions_changed/crossed_threshold/
                               #   max_value_check -- empirical instrumentation backing the proof.
                               #   binarize() is the single thresholding rule; foreground_count()
                               #   is a thin count on top of it
iterative_filter.py           # iterative Green/Red cluster removal + Wasserstein convergence tracking
                               #   (descriptive VietorisRipsPersistence on the 60-dim residual -- see
                               #   CLAUDE.md §4 on the two distinct persistence computations)
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

make_figure_v2.py              # poster figure V2: binarized clean/permuted/noised triptych.
                               #   Single combined Binarizer fit -- see "Combined-fit rule" below
make_figure_v3.py              # poster figure V3: four-family comparison (poster centerpiece)
poster_blocks.tex              # MathFest poster \block{} fragment. NOT a standalone document --
                               #   see "Poster" below

models/                        # trained surrogate classifiers (surrogate_mlp*.joblib, surrogate_rf*.joblib)
results/                       # experiment output JSON (seed-namespaced). GITIGNORED -- see
                               #   "Version-control hazards" below. CLAUDE.md §6 lists which result
                               #   sets are committed-and-backed vs. orphaned/unreproducible
figures/                       # generated plots and poster figures (*.pdf re-included via .gitignore
                               #   negation; see below)
data/                          # Payload-Byte CSVs (gitignored -- see Datasets below)

docs/                          # reference and historical documents:
                               #   TERMINOLOGY.md         -- definitions (radial centers corrected)
                               #   PROJECT_HANDOFF_1.md   -- historical; §4/§5/§6 stale
                               #   ABC_PHASE_REPORT.md    -- historical
                               #   P_/R_/W_PHASE_REPORT.md -- completion reports for the WIRE build and
                               #                             Test B rebuild. Accurate for what they
                               #                             report; CLAUDE.md §9 lists known errors,
                               #                             and P_PHASE_REPORT.md carries an appended
                               #                             erratum for the 2p(1-p) figure
                               #   retired LENS4 instruction files (dead -- do not revive)
                               # CLAUDE.md §0 lists exactly which sections of each are stale.
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

CUDA is present on WIRE but **irrelevant** — GUDHI cubical persistence and sklearn OPTICS are
CPU-only.

On WIRE the repo lives at `~/projects/tda_data_poisoning_prevention`. Do **not** work under `~/wire`,
which is the read-only view of the same NFS share. `local_scratch` is exfat (no symlinks, no POSIX
modes, may not survive rescheduling) — disposable intermediates only.

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

On WIRE these live under `~/wire/DataSets/PayloadByte_UNSW/` and `~/wire/DataSets/PayloadByte_CICIDS17/`.

**Do not use the raw `UNSW-NB15/` and `CICIDS2017/` directories** — they are flow-level and contain no
payload bytes, so they are unusable for this pipeline.

## Reproduction gate

Run before and after any pipeline change:

```bash
python tools/repro_check.py --expect 2.2000
```

Must return **2.2000% exact**, `X_tda.shape == (5500, 60)` -- seed 42, threshold 0.4, UNSW-NB15,
OPTICS, `malicious_random_attack` with `n_swaps=60`. This is the project's standing regression test
for "does this environment/these path changes still reproduce the recorded number."

## Combined-fit rule

When calling `extract_tda_features` (or any `Binarizer`-backed helper) to compare clean against
perturbed data, **always concatenate, fit once, and split** — never fit the two batches separately.
`Scaler` normalizes per batch, and separate fits manufacture spurious feature differences. An early
bit-identity check reported 16/200 false discrepancies exactly this way. See `CLAUDE.md` §9 item 8.

Note the scope: `test_b_diagnostics.py:117–118` uses two calls for clean-vs-permuted and is safe
*by theorem*, because a maximum is permutation-invariant. Lines 130–131 (clean vs noisy) are safe
only *by coincidence* — byte payloads already saturate at 255, so the batch maximum is unchanged.
That protection would not survive normalized floats or any non-saturating encoding.

## Version-control hazards

Two `.gitignore` rules are broader than intended and have silently swallowed files:

- **`results/` is active.** Existing result JSONs are tracked only because they predate the rule.
  **Anything new saves cleanly and stays invisible to git.** This must be resolved before any work
  that writes new results (see `CLAUDE.md` §8 items 5–8). Run `git check-ignore <path>` — with no
  `-v`, since `-v` reports negation matches and exits 0 regardless — on any new artifact.
- **`*.pdf`** was intended to exclude the copyrighted source papers but also caught generated
  figures. A `!figures/*.pdf` negation now re-includes them. A built poster PDF outside `figures/`
  would still be swallowed.
- **`docs/*` + `!docs/*.md`** replaced a bare `docs/` rule, which had made three phase completion
  reports invisible. The negation covers direct children only; a new file in a `docs/` subdirectory
  would still be ignored.

## Poster

`poster_blocks.tex` is a fragment of tikzposter `\block{}` calls — **not a standalone document.** It
must be `\input` by a parent that supplies `\documentclass{tikzposter}`, the title block, layout, and
a References block. **No such parent is currently in the repo**, so the poster copy has never been
compiled and the layout is unvalidated.

The file is pure ASCII and must stay that way (`latin9` inputenc): every symbol is a LaTeX macro,
never a literal glyph. Verify with:

```bash
LC_ALL=C grep -n '[^ -~[:space:]]' poster_blocks.tex   # must return nothing
```

Figures V2 and V3 are built and live in `figures/`.

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

# Poster figures
python make_figure_v2.py
python make_figure_v3.py

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
  '05*, Pisa, 263–271. (The 2005 conference version is the one read; not the 2007 *DCG* version.)
- Bergomi, M. G., Frosini, P., Giorgi, D., & Quercioli, N. (2019). Towards a topological–geometrical
  theory of group equivariant non-expansive operators for data analysis and machine learning.
  *Nature Machine Intelligence*, 1, 423–433.
- Ferrara, M. (2025). *AIMS Mathematics*, 10(7), 15457–15475. Positioned as a complementary axis
  (which points to attack) rather than a contrasting result; its Theorem 2's hypotheses do not hold
  here and it is **not** under test in this work.
- Zheng, S., Zhang, Y., Wagner, S., Goswami, M., & Chen, C. (2021). Topological detection of trojaned
  neural networks. *NeurIPS 2021*.
- Garin, A., & Tauzin, G. (2019). A topological reading lesson: Classification of MNIST using TDA.
  arXiv:1910.08345.
- Sethi, T. S., & Kantardzic, M. (2018); see also arXiv:1802.07295 for the closer
  impact-vs-detectability framing.

GENEO is cited above rather than deferred, because this work uses the phrase "invariance structure"
throughout and that literature exists precisely because persistence is "too invariant." See
`CLAUDE.md` §10 for how each source is positioned relative to this work.

## Acknowledgments

This research builds upon the TDA pipeline and methodology introduced by Monkam, De Lucia & Bastian
(2024) at the Army Cyber Institute, United States Military Academy.