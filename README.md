# Topological Data Analysis for Data Poisoning Prevention in Network Intrusion Detection Systems

## Abstract

Machine learning-based network intrusion detection systems (NIDS) are critical to defending military and civilian computer networks, but are also vulnerable to data poisoning attacks, where bad actors inject biased data into the training dataset to degrade model performance. Pre-existing methods of data poisoning detection leave the dataset unaltered and fail to differentiate a significant portion of the data. This research develops a novel iterative algorithm that leverages persistent homology, a method of topological data analysis, to progressively identify and remove poisoned data prior to model training. At each iteration, topological features are extracted from the residual dataset, while Wasserstein distances are computed as stopping criteria for convergence. Testing this algorithm on open-source network traffic datasets (UNSW-NB15, CICIDS2017) demonstrates that this iterative approach improves poisoned data capture significantly with zero false positives.

## Method

This project extends the single-pass TDA pipeline from Monkam et al. (2024) with a novel iterative cluster removal algorithm:

1. Convert network packet payloads to 1×1500 byte arrays
2. Extract topological features via persistent homology (cubical persistence with multiple filtrations)
3. Apply unsupervised density-based clustering (DBSCAN, HDBSCAN, OPTICS, Mean Shift)
4. Classify clusters: **Green** (100% clean), **Red** (100% poisoned), **Yellow** (mixed)
5. Remove Green → sanitized pool, Red → poisoned pool
6. Re-extract TDA features from the Yellow residual and repeat until convergence

Convergence is tracked via Wasserstein distance between persistence diagrams across iterations.

## Project Structure

```
├── data_loader.py            # Load UNSW-NB15 and CICIDS2017 datasets
├── poison.py                 # Data poisoning simulation
├── tda_pipeline.py           # TDA feature extraction (persistent homology)
├── clustering.py             # Unsupervised clustering and cluster classification
├── iterative_filter.py       # Novel iterative topological filtration algorithm
├── run_baseline.py           # Baseline single-pass experiment
├── run_iterative.py          # Iterative experiment (single run)
├── run_multi_seed.py         # Multi-seed statistical validation
├── classifier_eval.py        # Downstream NIDS classifier evaluation
├── visualize.py              # Visualization (single-run figures)
├── visualize_comparison.py   # Comparative visualization (multi-seed figures)
├── verify_env.py             # Environment verification
├── explore_data.py           # Data exploration utility
├── requirements.txt          # Python dependencies
└── figures/                  # Generated figures
```

## Requirements

- Python 3.12
- See `requirements.txt` for package dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Datasets

This project uses the following publicly available datasets in Payload-Byte format:

- **UNSW-NB15** — University of New South Wales, 2015
- **CICIDS2017** — Canadian Institute for Cybersecurity, 2017

Place the Payload-Byte CSV files in a `data/` directory:
```
data/
├── Payload_data_UNSW.csv
└── Payload_data_CICIDS2017.csv
```

## Usage

Run the experiments in order:

```bash
# 1. Verify environment
python verify_env.py

# 2. Baseline reproduction (single-pass TDA + clustering)
python run_baseline.py

# 3. Iterative filtration (single run)
python run_iterative.py

# 4. Multi-seed statistical validation (both datasets, ~2-4 hours)
python run_multi_seed.py

# 5. Generate figures
python visualize.py
python visualize_comparison.py

# 6. Downstream classifier evaluation
python classifier_eval.py
```

## Key Results

- OPTICS is the most effective clustering algorithm for TDA-based poison detection
- The iterative approach improves poison capture over single-pass methods across both datasets
- Zero false positives: every sample classified as poisoned was genuinely poisoned
- Wasserstein distance between successive persistence diagrams confirms topological convergence

## References

Monkam, G. F., De Lucia, M. J., & Bastian, N. D. (2024). A topological data analysis approach for detecting data poisoning attacks against machine learning based network intrusion detection systems. *Computers & Security*, 144, 103929.

## Acknowledgments

This research builds upon the TDA pipeline and methodology introduced by Monkam et al. (2024) at the Army Cyber Institute, United States Military Academy.
