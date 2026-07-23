# SESSION_KICKOFF_VERIFICATION — Read-Only Reconciliation Before Any Lens Work

**Type:** Read-only verification. **No repository code is to be modified by this file.**
**Runs:** First, at the start of the new session, *regardless* of the Lens 1-vs-Lens 3 pivot decision.
**House style:** This is a Phase-0 "confirm assumptions" pass — investigate, report against the numbered
steps, then **stop**. Do not begin any lens's work off the back of it without an explicit go-ahead.

---

## Why this file exists

Context was migrated from a prior chat via `PROJECT_HANDOFF.md`. That handoff flagged several claims as
**unconfirmed** — things a previous session *drafted or recommended* but never verified were actually done.
Before building anything new, reconcile those claims against the live repository so no new work is stacked
on an unconfirmed premise. Treat `CLAUDE.md` and `cc_summary.md` in the repo as the source of truth for
codebase facts; treat the handoff as a strategic snapshot that may have drifted.

Work entirely from the repo at `C:\TDA`. Activate the environment first:

```powershell
cd C:\TDA
.\venv312\Scripts\Activate.ps1
python verify_env.py   # sanity-check the environment before anything else
```

---

## Numbered verification steps

### 1. Re-read the living docs and confirm both source papers are in the repo
- Read the current `C:\TDA\CLAUDE.md` and `C:\TDA\cc_summary.md` **in full** from disk.
- List any place where they disagree with each other, or with the `PROJECT_HANDOFF.md` snapshot you were
  given. Do not "fix" anything yet — just record the discrepancies for the report.
- Confirm **both** methodology-source PDFs are checked in: the Monkam paper (`Monkam_DeLucia_Bastian.pdf`,
  already listed in the manifest) **and the Ferrara (2025) paper** (add it to the repo if it is not there —
  it is the primary source for Lens 1's Eq 3.3 and Lens 3's Eq 4.3/Algorithm 3, and those lenses must read
  the real equations, not the summaries). Note in particular, for whoever runs Lens 1/3, two exact facts
  from Ferrara that the summaries elide: **Eq 3.3's noise is isotropic `N(0, σ²I)` on standardized data**
  (App A.3), and **Eq 4.3's defender action `D_α` is a defense *mechanism* with payoff `−W_p` topological
  distortion**, not a purity threshold. The Lens 1 and Lens 3 instruction files are written around these.

### 2. Confirm whether the Lens 2 Wasserstein rebuild was committed to git
The handoff records that a commit message was *drafted and recommended* for the Lens 2 change to
`iterative_filter.py`, but there is **no confirmation `git commit` was ever run**.
- Run, and capture output:
  ```powershell
  git log --oneline -n 15
  git status
  git diff --stat HEAD
  ```
- Confirm the current `iterative_filter.py` on disk actually contains
  `compute_whole_residual_diagram(...)` and `wasserstein_distance_between_diagrams(...)`
  (the real `VietorisRipsPersistence` + `PairwiseDistance(metric="wasserstein", p=1)` implementation),
  and that the superseded `compute_persistence_diagrams(...)` name is **absent** repo-wide:
  ```powershell
  Select-String -Path *.py -Pattern "compute_whole_residual_diagram|wasserstein_distance_between_diagrams|compute_persistence_diagrams"
  ```
- Report: is the Lens 2 work (a) present on disk, and (b) committed? If present-but-uncommitted, **do not
  commit it yourself** — surface it in the report and let the user decide, since a commit is a checkpoint
  they may want to word/time themselves.

### 3. Confirm whether the 5-seed cluster-size probe was completed
The handoff notes a request to extend the Lens 1 Yellow/Pink **cluster-size probe** from 1 seed to a full
5-seed sweep (**DBSCAN / HDBSCAN / OPTICS only — Mean Shift is out of Lens 1 scope**, both datasets), to
get an honest Phase-2 cost estimate — but it is **unclear whether this was ever run and reported**.
- Search the repo for any artifact of it: a probe/scan script, a saved log, a results JSON, or a section
  in `LENS1_TOPOLOGICAL_INFLUENCE_INSTRUCTIONS.md` reporting per-seed cluster sizes.
  ```powershell
  Get-ChildItem -Recurse -Include *.md,*.json,*.txt,*.py | Select-String -Pattern "cluster.?size|yellow.*probe|5.?seed" -List
  ```
- Report one of: **done** (cite the artifact and summarize the per-seed sizes), **partial** (single-seed
  only — the seed-42 numbers in the handoff: 790 Yellow / 3 Pink / ~75,385 pts all-algos, ~47,970 excl.
  Mean Shift), or **not found**. Do not run the sweep now — just establish its status.

### 4. Resolve the `results_io.py` drift
The handoff (§3) lists `results_io.py` as a new post-cleanup shared JSON-serialization helper, but the
(post-Lens-2) `cc_summary.md` **file manifest does not list it**.
- Check: does `C:\TDA\results_io.py` exist? If so, which drivers import it
  (`run_baseline` / `run_iterative` / `run_multi_seed`)? If not, is JSON serialization still duplicated
  across those drivers?
  ```powershell
  Test-Path .\results_io.py
  Select-String -Path *.py -Pattern "results_io|json.dump" 
  ```
- Report the actual state so the docs can be reconciled later. Do not create or delete the file.

### 5. Confirm the two Lens 1 reference clusters are reproducible
Lens 1's next test (see `LENS1_SIGMA_OPTIONA_INSTRUCTIONS.md`) reuses two reference clusters from
**seed 42, `MAX_SAMPLES=5000`**: a small one (**~10 points**) and a large one (**~971 points**).
- Without modifying repo code, confirm you can regenerate a residual and identify a Yellow cluster of each
  size, and record **which algorithm and which dataset** each came from (needed so the Lens 1 file targets
  the same reference points). If the exact sizes 10 and 971 are not reproducible under the current seed,
  report the nearest reproducible small/large Yellow clusters and their sizes instead.

---

## Stop condition & completion report

After step 5, **stop**. Do not start Lens 1 or Lens 3 work. Produce a completion report structured around
the five numbered steps above, each answered explicitly as **confirmed / partial / not found**, with the
supporting command output quoted (not paraphrased). End the report with a single short section:

> **Open decision for the user:** Lens 1 is genuinely stuck (two sigma designs empirically ruled out, a
> third — option A — not yet tested) and Lens 3 has *no dependency on Lens 1* (it needs only Lens 2's real
> distance, already in place, plus the existing `classify_clusters()` thresholds). There is a live proposal
> to **pivot to starting Lens 3 now**. This is the user's call — present it and wait. Whichever branch they
> pick, the matching instruction file (`LENS1_SIGMA_OPTIONA_INSTRUCTIONS.md` or
> `LENS3_MINIMAX_PHASE1_INSTRUCTIONS.md`) is ready to run.

Report deviations and partial results honestly — "recommended but not confirmed run" is a finding, not a
failure. Do not round an unconfirmed claim up to "done."
