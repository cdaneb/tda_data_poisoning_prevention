"""Cross-platform process resource measurements for experiment metadata."""
from __future__ import annotations

import sys


def peak_rss_kib() -> int | None:
    """Return peak resident memory in KiB when the platform exposes it.

    ``resource`` is available on POSIX systems but not on Windows. Linux
    reports ``ru_maxrss`` in KiB, while macOS reports bytes. Returning ``None``
    on unsupported platforms keeps result schemas stable without making a
    non-scientific telemetry field prevent an experiment from importing.
    """
    try:
        import resource
    except ImportError:
        return None

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if sys.platform == "darwin" else value
