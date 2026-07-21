# Codebase Audit — Pre-Refactor Cleanup Pass

## Purpose

Before implementing new functionality, we need a clean, well-understood codebase.
This document asks you (Claude Code) to **audit** `C:\TDA` for redundant, unused,
orphaned, or stale files/dependencies — things that waste tokens, disk, or compute
without contributing to the project's goals.

**This is an audit-only pass. Do not delete, move, rename, refactor, or edit any
file during this task.** Your only output should be a written report and, at the
very end, a request for my sign-off before anything changes.

---

## Context: where the project is headed (for relevance judgments only)

You do not need to implement any of this now — it's provided only so you can judge
whether a given file is "working toward the goal" or is dead weight. The project
currently reproduces Monkam et al. (2024)'s TDA + unsupervised-clustering data
poisoning detector for NIDS datasets, with a novel iterative filtration loop. The
next phase of work (not part of this task) will add four extensions drawn from
Ferrara (2025):

1. **Lens 1** — per-point topological influence (TI) scoring to partially resolve
   "Yellow" (mixed) clusters, instead of relying only on 100%-purity thresholds.
2. **Lens 2** — replacing the current approximate Wasserstein-distance heuristic
   (`iterative_filter.py::wasserstein_distance_between_diagrams`) with a true
   diagram-to-diagram distance (e.g., via `gtda.diagrams.PairwiseDistance`).
3. **Lens 3** — reframing threshold selection as a minimax/game-theoretic problem
   rather than a flat parameter sweep.
4. **Lens 4** — a new "topological" and "combined" poisoning attack generator
   (high-topological-influence points, optionally near a decision boundary) to
   replace/augment the current Gaussian-noise + byte-swap poisoning in `poison.py`.

Use this only to judge relevance — e.g., a file that has nothing to do with any of
the above, isn't part of the working pipeline, and isn't useful documentation of
*why* the project is built the way it is, is a cleanup candidate.

---

## What to audit

Work through the codebase systematically. For each finding, note the file path,
what it is, and which category below it falls into. Be conservative — if you're
unsure whether something is safe to flag, note the uncertainty rather than
guessing.

### 1. Re-verify the active dependency graph
Trace forward from every actual entry point (`run_baseline.py`, `run_iterative.py`,
`run_multi_seed.py`, `classifier_eval.py`, `visualize.py`, `visualize_comparison.py`,
`verify_env.py`) through every `import`/`from` statement, and build the real,
current dependency graph — don't assume prior documentation (including
`cc_summary.md`, if present) is still accurate. Flag any discrepancy you find
between what's documented and what the code actually does.

### 2. Orphaned files
Identify any `.py` file in the repo root that is **not imported by anything** and
**not itself an entry point someone would run directly** (i.e., not referenced in
`README.md` or `CLAUDE.md` as a usage step). Known suspects to specifically check
(verify, don't just take this list on faith):
- `explore_data.py` — previously noted as a one-off exploration script, not
  imported elsewhere.
- Any other `.py` file that doesn't show up in the dependency graph from step 1.

### 3. Unused or mismatched dependencies
Check `requirements.txt` against actual imports across all `.py` files. Known
suspect: the standalone `hdbscan` PyPI package is listed and imported in
`verify_env.py` for a smoke test, but the actual clustering code
(`clustering.py`) uses `sklearn.cluster.HDBSCAN` instead. Confirm whether this
is still the case, and flag any other requirements-vs-actual-usage mismatches.

### 4. Duplicate or overlapping logic
Check whether any TDA feature extraction, clustering, or poisoning logic is
implemented more than once across different files (e.g., a script that
reimplements something already in `tda_pipeline.py`/`clustering.py`/`poison.py`
instead of importing it). Flag any duplication, however small.

### 5. Documentation/code mismatches
Flag places where a docstring, comment, or markdown doc describes behavior the
code doesn't actually have. Known suspect: `clustering.py`'s docstring mentions a
"Blue" cluster category that is never actually assigned in `classify_clusters()`.
Look for others.

### 6. Stale or superseded artifacts
Look for files that were useful once but no longer reflect the current state of
the project:
- Old console-log dumps (e.g., `run_baseline_test1.txt`) — check whether these are
  referenced anywhere or are just a leftover snapshot.
- Duplicate virtual environments (e.g., an old `venv` alongside the active
  `venv312`) — confirm which one is actually used per `README.md`/`CLAUDE.md` and
  flag the other as a disk-space (not token) cleanup candidate.
- Any `results/*.json` or `figures/*.png` that a current script no longer
  generates or reads (i.e., outputs from a since-removed or since-changed code
  path).

### 7. Context-bloat risk files
Identify any large markdown/text files in the repo root that would consume
significant tokens if read into context during future sessions, but whose
content is historical rather than actively needed (e.g., long
phase-by-phase build instructions from earlier development). Known suspects:
`cc_instructions_1.md`, `cc_instructions_2.md`, `cc_instructions_3.md` (reported
as ~995/1085/838 lines). Don't recommend deleting project history — instead
assess whether these belong in an `archive/` subfolder so they're not
automatically pulled into context on every session, versus staying in root.

### 8. Incomplete-but-not-redundant files (flag only, no action needed)
Some files are incomplete relative to the rest of the pipeline but are NOT
redundant — don't recommend removing these, just confirm their current state so
we have an accurate baseline before touching them in the next phase:
- `run_baseline.py` — confirm whether the CICIDS2017 call is still commented
  out in `__main__`, and whether results are still console-only (no JSON saved).
- `classifier_eval.py` — confirm whether it still only supports UNSW-NB15 +
  OPTICS, and still only prints to console with no saved results file.

---

## Deliverable

Produce a single report file named `CODEBASE_AUDIT_REPORT.md` in the repo root
with these sections:

1. **Verified dependency graph** — the real, current one, noting any drift from
   prior documentation.
2. **Orphaned files** — path, what it is, confidence level that it's unused.
3. **Dependency mismatches** — declared vs. actually used.
4. **Duplicate logic** — where, and which version looks canonical.
5. **Documentation/code mismatches**.
6. **Stale artifacts** — logs, extra venvs, superseded results/figures.
7. **Context-bloat files** — candidates for archiving vs. keeping in root.
8. **Confirmed state of the flagged-incomplete files** (section 8 above) — facts
   only, no recommendation to change them yet.
9. **Proposed actions** — for every item above, propose one of: `safe to delete`,
   `archive (move, don't delete)`, `keep as-is`, or `needs your decision` — with a
   one-line rationale each. This is a proposal list, not an action list.

---

## Hard constraints

- **Do not delete, move, rename, edit, or refactor any file in this pass.** This
  task is read-only analysis.
- **Do not open or read the large CSV data files** (`Payload_data_UNSW.csv`,
  `Payload_data_CICIDS2017.csv`) — their existence and location is enough;
  reading them wastes tokens for no benefit to this audit.
- **Do not read entire large `results/*.json` files** if their size/structure can
  be confirmed via file size and a partial read/`head` — we only need to know
  whether they're stale, not re-verify every number in them.
- Do not begin any implementation work related to Lenses 1–4. That is
  intentionally out of scope for this task.

## Stop condition

Once `CODEBASE_AUDIT_REPORT.md` is written, **stop and tell me it's ready for
review.** Do not act on any of the proposed actions — including anything marked
`safe to delete` or `archive` — until I've reviewed the report and explicitly
confirm each action (or a batch of them) in our next conversation.
