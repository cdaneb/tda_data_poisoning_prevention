# Claude Code handoff: complete Phase Q on WIRE

## Objective

Complete the Phase Q R1 experiment on WIRE: compare the repaired single-threshold
control against the fixed multithreshold stack over all four permutation families
and all five seeds, validate the resulting artifact, interpret it using the frozen
matched-clean-cost metrics, and update the research documentation truthfully.

Do not commit or push. Leave the working tree ready for Christian to review and
commit.

## Scientific constraints

Do not change any preregistered experimental choice unless an actual software bug
prevents execution:

- Same support-restricted, guaranteed-nontrivial attack realization in both arms.
- Families: transpositions, block reversal, block swap, and cyclic shift.
- Seeds: `42, 123, 456, 789, 1024`.
- Control: threshold `0.4`, 60 features.
- Repair: thresholds `{0.1, 0.2, ..., 0.9}`, 540 concatenated features.
- Stack normalization: fixed global factor `1/sqrt(9)`.
- Keep the 30x50 raster, five filtrations, cubical persistence, `Scaler`, six
  diagram summaries, and OPTICS parameters unchanged.
- OPTICS remains `min_samples=5, max_eps=2.0`.
- Frozen purity points are exact `1.0`, then `>0.95`, `>0.90`, `>0.80`, and
  `>0.50`.
- Primary interpretation is poison removal at matched clean false-removal cost.
- Also report exact-purity capture, clean false-removal, removal precision,
  unclustered fractions, cluster counts, and exact-duplicate fractions.

The multithreshold theorem buys "not trivially blind," not guaranteed detection.
Do not use "invisible," "evades," or "defeats." Do not call a higher capture rate
an improvement if it is purchased through higher clean removal or fragmentation.

The current D1-D4 result is deliberately uncomfortable: the stack removes exact
feature identity but did not improve attack displacement relative to clean-clean
variation. Report the R1 result whichever way it lands; do not tune after seeing
it.

## Paths

- Writable repository: `~/projects/tda_data_poisoning_prevention`
- Read-only UNSW Payload-Byte directory:
  `~/wire/DataSets/PayloadByte_UNSW`
- Required CSV:
  `~/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv`
- R1 artifact:
  `results/phase_q_r1_multithreshold_capture.json`
- Runtime log: `results/phase_q_wire_run.log`

Do not work from the read-only `~/wire` repository view. Use only the writable
clone under `~/projects`.

## 1. Start a persistent session

Run:

```bash
tmux new -s phaseq
```

To detach without stopping the job, press `Ctrl-b`, then `d`.

To reattach later:

```bash
tmux attach -t phaseq
```

## 2. Preflight and environment setup

Inside the `tmux` session, run this block exactly:

```bash
set -o pipefail

cd ~/projects/tda_data_poisoning_prevention
git pull --ff-only
git status --short

test -f phase_q_attacks.py
test -f phase_q_pipeline.py
test -f phase_q_metrics.py
test -f run_phase_q_experiment.py
test -f tools/test_phase_q.py
test -f docs/PHASE_Q_MULTITHRESHOLD_REPORT.md
test -f requirements.lock.txt
test -f "$HOME/wire/DataSets/PayloadByte_UNSW/Payload_data_UNSW.csv"

python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

test -x venv312/bin/python || python3 -m venv venv312
source venv312/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt

export TDA_DATA_DIR="$HOME/wire/DataSets/PayloadByte_UNSW"
export TDA_RESULTS_DIR="$PWD/results"
export LOKY_MAX_CPU_COUNT="$(nproc)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$TDA_RESULTS_DIR"
```

If the Python version assertion, data-file check, dependency installation, or
any required-file check fails, stop and diagnose the environment. Do not weaken
the check or substitute a different dataset.

## 3. Required gates

Run:

```bash
python verify_env.py
python -m unittest tools.test_phase_q -v
python tools/repro_check.py --expect 2.2000
```

Required outcomes:

- All Phase Q unit tests pass.
- The 0.4 block remains exactly equal to the legacy feature pipeline.
- The legacy gate returns OPTICS capture exactly `2.2000%` and feature shape
  `(5500, 60)`.

If a gate fails, do not start R1. Capture the exact error and investigate. Do not
change thresholds, attacks, scaling, metrics, seeds, or OPTICS to make a gate
pass.

## 4. Run the complete R1 grid

Run exactly one writer:

```bash
python -u run_phase_q_experiment.py --family all --seed all \
  2>&1 | tee results/phase_q_wire_run.log
```

Do not launch concurrent copies against the same JSON artifact.

The runner writes after every completed family/seed cell. If interrupted, rerun
the identical command; existing completed cells are skipped and the remaining
grid resumes. Never delete a partial valid artifact merely to restart from zero.

Monitor from another terminal if needed:

```bash
tail -f ~/projects/tda_data_poisoning_prevention/results/phase_q_wire_run.log
```

## 5. Validate the completed artifact

After the experiment exits successfully, run:

```bash
cd ~/projects/tda_data_poisoning_prevention
source venv312/bin/activate

python -m json.tool \
  results/phase_q_r1_multithreshold_capture.json \
  >/dev/null

if git check-ignore -q results/phase_q_r1_multithreshold_capture.json; then
  echo "ERROR: R1 artifact is ignored"
  exit 1
else
  echo "PASS: R1 artifact is Git-visible"
fi

git status --short
```

Then verify programmatically or by careful JSON inspection that:

- All four family keys exist.
- Every family contains all five seed keys.
- Every run has `n_clean == 5000`, `n_poison == 500`, and
  `raw_noop_count == 0`.
- Control feature count is 60.
- Repair feature count is 540.
- Each arm contains five frozen removal-curve points.
- Each run contains five matched-clean-cost records.
- Every run includes an environment/provenance block.
- No numeric field required for interpretation is NaN or infinite.

If a cell is absent, rerun the same resumable command. Do not fabricate or
manually copy a result into the artifact.

## 6. Analyze without tuning

For each family, summarize across the five seeds using population standard
deviation (`ddof=0`):

- Exact-purity poison removal for control and repair.
- Clean false-removal at each frozen purity point.
- Removal precision.
- Poison and clean unclustered fractions.
- Number of clusters.
- Exact-duplicate-with-clean fraction.
- Matched-clean-cost poison-removal delta.

Answer these questions in order:

1. Did the repair improve poison removal at the same or lower clean cost?
2. Is any apparent improvement explained by greater fragmentation or a larger
   unclustered fraction?
3. Did exact-duplicate frequency fall without producing usable cluster
   separation?
4. Is the result consistent across all five seeds, or driven by one seed?
5. Does any family improve while another worsens?

Do not perform threshold selection, block weighting, feature selection, OPTICS
tuning, attack resampling, seed deletion, or post-hoc metric changes. Those would
be a new experiment and require a new preregistration.

## 7. Update documentation after results exist

Update only the research record, not the poster, unless Christian separately
requests poster changes:

- `docs/PHASE_Q_MULTITHRESHOLD_REPORT.md`
- `CLAUDE.md`
- `README.md`

Change "capture pending" to the observed status and add the five-seed R1 table.
Preserve the distinction between:

- algebraic repair: exact stack identity is removed on the tested frame;
- statistical separation: attack displacement relative to clean variation;
- downstream detection: matched-clean-cost OPTICS removal.

If R1 fails to improve matched-cost removal, state plainly that the simple
unweighted/equally weighted threshold-stack repair is falsified as a detector,
even though it closes the exact single-cutoff identity mechanism. If it improves,
report the clean-cost and fragmentation checks beside the improvement.

Do not overwrite or reinterpret the historical Phase P/M artifacts.

## 8. Final verification and handoff

After documentation edits, run:

```bash
source venv312/bin/activate
python -m unittest tools.test_phase_q -v
python -m json.tool results/phase_q_r1_multithreshold_capture.json >/dev/null
git diff --check
git status --short
```

Report to Christian:

- Whether the full 20-cell grid completed.
- Total wall-clock time.
- The principal matched-clean-cost result.
- Any fragmentation or unclustered-fraction change.
- The exact files changed or generated.
- Every verification command and whether it passed.
- Any warnings or unresolved limitations.

Do not commit or push. Human will review and commit.
