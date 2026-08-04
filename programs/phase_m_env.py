"""
Phase M provenance helper (A9 + A10).

Every results/phase_m_*.json must embed an `env` block so a Phase M number
never becomes the next orphan (CLAUDE.md §6). A recorded commit hash alone is
a false claim when the working tree is dirty — which it is throughout M3–M7
with uncommitted attack/driver code — so A10 requires a `dirty` flag and a
short diffstat when true. The pandas 3.0.2-vs-lock-3.0.5 and Python
3.12.5-vs-recorded-3.12.3 deltas travel with the artifact as recorded facts
rather than notes in a report someone has to go find.

Usage:
    from phase_m_env import env_block
    results["env"] = env_block()
"""
import importlib.metadata as _md
import platform
import socket
import subprocess
from datetime import datetime, timezone

# The six version-sensitive pins the phase reproduces against (CLAUDE.md §3).
_PINNED = ["giotto-tda", "scikit-learn", "numpy", "scipy", "pandas", "joblib"]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=__import__("pathlib").Path(__file__).resolve().parent.parent,
                              timeout=15).stdout.strip()
    except Exception as e:  # git absent / not a repo — record the failure, don't crash a run
        return f"<git-error: {e.__class__.__name__}>"


def env_block():
    porcelain = _git("status", "--porcelain")
    dirty = bool(porcelain.strip()) and not porcelain.startswith("<git-error")
    versions = {}
    for pkg in _PINNED:
        try:
            versions[pkg] = _md.version(pkg)
        except Exception as e:
            versions[pkg] = f"<missing: {e.__class__.__name__}>"

    block = {
        "python_version": platform.python_version(),
        "packages": versions,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if dirty:
        # A10: a hash that silently omits uncommitted changes is how an artifact
        # becomes unreproducible while looking rigorous. Record what differs.
        block["git_diffstat"] = _git("diff", "--stat")
        block["git_untracked_or_staged"] = porcelain
    return block


if __name__ == "__main__":
    import json
    print(json.dumps(env_block(), indent=2))
