"""
Central path configuration.

Resolves DATA_DIR / RESULTS_DIR / MODELS_DIR / FIGURES_DIR from environment
variable overrides (TDA_DATA_DIR, TDA_RESULTS_DIR, TDA_MODELS_DIR,
TDA_FIGURES_DIR) if set, else repo-relative defaults. Always resolved from
this file's own location, never from the process's current working
directory, so behavior doesn't depend on where a script happens to be
launched from.

Replaces the previous hardcoded `Path(r"C:\\TDA\\...")` constants, which on
POSIX silently parsed as relative paths literally named `C:\\TDA\\...`
rather than the intended Windows absolute paths (see Phase R / Phase W
reports).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _resolve(env_var, default_name):
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    return REPO_ROOT / default_name


DATA_DIR = _resolve("TDA_DATA_DIR", "data")
RESULTS_DIR = _resolve("TDA_RESULTS_DIR", "results")
MODELS_DIR = _resolve("TDA_MODELS_DIR", "models")
FIGURES_DIR = _resolve("TDA_FIGURES_DIR", "figures")
