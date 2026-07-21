# Claude Code Instructions — Generate Codebase Summary

## Task

Read and analyze every Python file in `C:\TDA`. Then generate a new file called `C:\TDA\cc_summary.md` that comprehensively documents the entire project: what it does, how it works, what results have been achieved, and what remains to be done.

## Steps

### Step 1: Read all source files

Read the following files in order. For each file, understand its purpose, its inputs/outputs, its dependencies on other files, and any hardcoded configuration:

1. `C:\TDA\requirements.txt`
2. `C:\TDA\verify_env.py`
3. `C:\TDA\explore_data.py`
4. `C:\TDA\data_loader.py`
5. `C:\TDA\poison.py`
6. `C:\TDA\tda_pipeline.py`
7. `C:\TDA\clustering.py`
8. `C:\TDA\iterative_filter.py`
9. `C:\TDA\run_baseline.py`
10. `C:\TDA\run_iterative.py`
11. `C:\TDA\run_multi_seed.py`
12. `C:\TDA\classifier_eval.py`
13. `C:\TDA\visualize.py`
14. `C:\TDA\visualize_comparison.py`
15. `C:\TDA\README.md`

Also check for any other `.py` files in `C:\TDA` not listed above and read those too.

Check what exists in:
- `C:\TDA\results\` — list all JSON files present
- `C:\TDA\figures\` — list all PNG/image files present
- `C:\TDA\data\` — list all data files present (do NOT read them, just list filenames and sizes)

### Step 2: Generate `C:\TDA\cc_summary.md`

Create the file `C:\TDA\cc_summary.md` with the following structure and content. Be thorough and specific — a new agent reading only this file should be able to understand the entire project, modify any component, and continue development without needing to ask clarifying questions.

---

**Required sections for cc_summary.md:**

#### 1. Project Overview
- Project title and one-paragraph description
- The research problem being solved
- The seed paper this extends (Monkam et al. 2024, full citation)
- What is novel about this project vs the seed paper

#### 2. Environment and Setup
- Python version required
- Virtual environment location and activation command
- All package dependencies with version constraints
- Any known compatibility issues encountered during setup (e.g., giotto-tda not supporting Python 3.13, hdbscan replaced with sklearn.cluster.HDBSCAN)

#### 3. Data
- What datasets are used (UNSW-NB15, CICIDS2017)
- File locations and sizes
- Data format: exact column names, data types, shape
- How the data is loaded (reference data_loader.py, note the exact column names used)
- How data is poisoned (reference poison.py, describe the perturbation method, default parameters)

#### 4. Pipeline Architecture
- A complete description of the data flow from raw CSV → final results
- For each Python file, document:
  - Purpose (one sentence)
  - Key functions with their signatures (name, arguments, return values)
  - Dependencies (which other project files it imports)
  - Any hardcoded parameters or configuration values
- Draw the dependency graph as a text diagram showing which files import from which

#### 5. TDA Feature Extraction Details
- Describe exactly what `tda_pipeline.py` does step by step
- The reshape from (N, 1500) to (N, 30, 50)
- The 5 filtrations (2 HeightFiltration directions + 3 RadialFiltration centers) — list exact parameter values
- The diagram steps: Binarizer threshold, CubicalPersistence, Scaler
- The 5 metrics with exact parameters
- The feature union: PersistenceEntropy + 5 Amplitudes
- The actual output feature count (60, not 72) and why it differs from the paper's claim
- Computational cost: approximately how long TDA extraction takes per 1000 samples

#### 6. Clustering Details
- The four algorithms used: DBSCAN, HDBSCAN, OPTICS, MeanShift
- Default hyperparameters for each
- The cluster color classification scheme: Green (0% poisoned), Red (100% poisoned), Yellow (mixed), Pink (>80% poisoned)
- How noise points (label -1) are handled

#### 7. The Novel Algorithm: Iterative Topological Filtration
- Full pseudocode of the iterative algorithm (read it from iterative_filter.py and present it clearly)
- The three pools: sanitized (Green), poisoned (Red), residual (Yellow)
- Why TDA features are re-extracted fresh each iteration (not reused)
- The Wasserstein distance convergence tracking: how it's computed, what it measures
- Stopping conditions: no Green/Red clusters found, residual too small, max iterations reached
- The key design decision: 100% purity thresholds for Green/Red classification (a cluster must be entirely clean or entirely poisoned)

#### 8. Experiments Run and Results Achieved
- **Baseline (single-pass):** What was run, key finding (OPTICS only algorithm producing Red clusters, ~6.6% poison capture on first run)
- **Single-seed iterative:** OPTICS ran 4 iterations, improved capture to 8.2%, zero false positives, Wasserstein convergence 8.58 → 0.62
- **Multi-seed validation:** 5 seeds × 2 datasets × 4 algorithms. Summarize the aggregated results. List exact filenames of saved JSON results and what each contains.
- **Downstream classifier evaluation:** Clean vs poisoned vs sanitized accuracy comparison. Note that the gap was small due to simplified poisoning.
- List all generated figures with descriptions of what each shows

#### 9. Known Limitations
- Simplified poisoning (Gaussian noise + byte swaps) vs the paper's genetic algorithm / deep RL attacks
- The 100% purity threshold means most clusters are classified as Yellow
- Only ~2-3% improvement in poison capture from iterative approach over baseline
- The authors of the seed paper were contacted for their poisoned datasets but have not yet responded
- Feature count is 60 instead of the paper's 72 due to giotto-tda version differences

#### 10. Next Steps and Improvement Strategies

This section is critical. Describe the following potential improvements in detail, with enough specificity that a new agent could implement any of them:

**A. Threshold Relaxation (Highest Priority)**
- Currently clusters must be 100% pure to be classified as Green or Red
- Proposed change: introduce a threshold parameter (e.g., clusters >90% poisoned → Red, <10% poisoned → Green)
- Implementation: modify the `classify_clusters()` function in `clustering.py` to accept threshold parameters
- Analysis: sweep thresholds from 50% to 100% and plot precision vs recall curve
- This directly addresses the Yellow cluster problem and requires minimal code changes

**B. Hybrid Supervised Classification of Yellow Residual**
- Use Green cluster samples as "clean" training labels, Red cluster samples as "poisoned" training labels
- Train a binary classifier (Random Forest, SVM) on TDA features of labeled samples
- Apply classifier to Yellow residual samples to predict clean vs poisoned
- This bootstraps supervised learning from unsupervised results

**C. Enhanced TDA Feature Space**
- Current: 60 features from Amplitude + PersistenceEntropy
- Add persistence images (grid discretization of persistence diagrams)
- Add persistence landscapes (functional summaries)
- Both supported by giotto-tda
- More features give clustering algorithms more dimensions for separation

**D. Adaptive Clustering Hyperparameters**
- Current: same hyperparameters used at every iteration
- The residual dataset changes size and structure each iteration
- Proposed: scale min_samples or eps based on residual size
- Or switch algorithms between iterations (e.g., OPTICS for early iterations, spectral clustering for later)

**E. Mapper Algorithm Integration**
- Apply the Mapper algorithm to Yellow residual as a complementary TDA tool
- Mapper builds a graph representation that can reveal connectivity structure clustering misses
- Supported by giotto-tda and kmapper libraries
- Would add a second TDA method (valuable for a math conference presentation)

**F. Obtain Real Poisoned Data**
- The authors (Monkam, Bastian) at West Point were emailed requesting their poisoned datasets
- Their poisoned data uses Chale et al. 2023 (genetic algorithm) and Hore et al. 2023 (deep RL) attack methods
- These create more topologically distinct perturbations than our Gaussian noise approach
- If received: swap into the pipeline by replacing the `poison_dataset()` call with loading their pre-poisoned data
- No code changes needed to the iterative algorithm itself

#### 11. File Manifest
- Complete list of every file in the project with its path, purpose, and approximate line count

---

### Step 3: Verify the output

After generating `C:\TDA\cc_summary.md`, verify that:
- All Python files listed in Step 1 are documented in the summary
- All results and figures files are listed
- The dependency graph is accurate (check actual imports in each file)
- The function signatures match the actual code
- The hardcoded parameter values match the actual code

If any discrepancies are found, fix them in the summary.
