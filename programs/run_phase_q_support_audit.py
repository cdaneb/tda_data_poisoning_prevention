"""Phase Q D0: audit padding no-ops and the repaired attack substrate.

This is intentionally separate from the historical Phase P/M drivers.  It
compares their generators with support-restricted counterparts on the same
UNSW dataset without overwriting any recorded artifact.
"""
import json

import numpy as np

from programs.adversarial_attack import (
    block_reversal_attack,
    block_swap_attack,
    cyclic_shift_attack,
    malicious_random_attack,
)
from programs.data_loader import load_unsw_with_metadata
from programs.phase_m_env import env_block
from programs.phase_q_attacks import SUPPORTED_FAMILIES, conservative_support_lengths
from programs.paths import RESULTS_DIR
from programs.results_io import convert_for_json


N_DIAG = 200
SEED = 42
OUTPUT_NAME = "phase_q_d0_support_audit.json"

LEGACY_FAMILIES = {
    "transpositions": (malicious_random_attack, {"n_swaps": 60}),
    "block_reversal": (block_reversal_attack, {"k": 120}),
    "block_swap": (block_swap_attack, {"k": 60}),
    "cyclic_shift": (cyclic_shift_attack, {}),
}


def _attack_summary(fn, kwargs, X, y):
    rate = N_DIAG / len(X)
    Xc, _, _, log = fn(X, y, poison_rate=rate, random_state=SEED, **kwargs)
    clean = X[np.array([entry["target_index"] for entry in log])]
    poison = Xc[len(X):]
    changed = np.count_nonzero(clean != poison, axis=1)
    return {
        "n": int(len(changed)),
        "raw_noop_count": int(np.count_nonzero(changed == 0)),
        "raw_noop_fraction": float(np.mean(changed == 0)),
        "positions_changed_mean": float(changed.mean()),
        "positions_changed_median": float(np.median(changed)),
        "positions_changed_min": int(changed.min()),
        "positions_changed_max": int(changed.max()),
    }


def run_audit():
    X, y, metadata = load_unsw_with_metadata(max_samples=None)
    support = conservative_support_lengths(X)
    total_len = metadata["total_len"].to_numpy(dtype=float)
    gap = total_len - support

    result = {
        "phase": "Q-D0",
        "random_state": SEED,
        "n_diag": N_DIAG,
        "support_definition": {
            "name": "conservative_support",
            "rule": "one plus the last nonzero payload-byte index; zero for all-zero rows",
            "guarantee": "cannot include NumPy-appended zero padding",
            "limitation": "may omit legitimate trailing zero-valued payload bytes",
            "total_len_warning": "Payload-Byte total_len is IPv4 packet length, not transport payload length",
        },
        "dataset_support": {
            "n_rows": int(len(X)),
            "support_min": int(support.min()),
            "support_median": float(np.median(support)),
            "support_mean": float(support.mean()),
            "support_max": int(support.max()),
            "fraction_support_ge_120": float(np.mean(support >= 120)),
            "total_len_minus_support": {
                "median": float(np.median(gap)),
                "mean": float(gap.mean()),
                "q05": float(np.quantile(gap, 0.05)),
                "q95": float(np.quantile(gap, 0.95)),
            },
        },
        "legacy": {},
        "support_restricted_nontrivial": {},
        "env": env_block(),
    }

    for name, (fn, kwargs) in LEGACY_FAMILIES.items():
        result["legacy"][name] = _attack_summary(fn, kwargs, X, y)
    for name, (fn, kwargs) in SUPPORTED_FAMILIES.items():
        result["support_restricted_nontrivial"][name] = _attack_summary(fn, kwargs, X, y)
    return result


def main():
    result = run_audit()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / OUTPUT_NAME
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, default=convert_for_json)
    print(json.dumps({
        "legacy": result["legacy"],
        "support_restricted_nontrivial": result["support_restricted_nontrivial"],
        "dataset_support": result["dataset_support"],
    }, indent=2))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
