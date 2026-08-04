"""Verify that all required packages are installed and importable."""
import sys

print(f"Python: {sys.version}")

import numpy as np
print(f"numpy: {np.__version__}")

import pandas as pd
print(f"pandas: {pd.__version__}")

import sklearn
print(f"scikit-learn: {sklearn.__version__}")

from sklearn.cluster import HDBSCAN
HDBSCAN(min_cluster_size=10)
print(f"sklearn.cluster.HDBSCAN: imported and instantiated OK")

import matplotlib
print(f"matplotlib: {matplotlib.__version__}")

# giotto-tda specific imports needed for the pipeline
from gtda.images import Binarizer, HeightFiltration, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude
from sklearn.pipeline import make_pipeline, make_union
print("giotto-tda: all required classes imported successfully")

print("\n=== Environment verification PASSED ===")
