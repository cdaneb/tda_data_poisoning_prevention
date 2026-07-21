"""
Shared JSON-serialization helper for experiment result files.

numpy scalar/array types aren't natively JSON-serializable; every results-saving
script needs the same conversion, so it lives here once instead of being
reimplemented per script.
"""
import numpy as np


def convert_for_json(obj):
    """Convert numpy types to native Python types for json.dump(default=...)."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
