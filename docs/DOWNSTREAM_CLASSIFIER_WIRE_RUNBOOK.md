# WIRE runbook: preregistered downstream classifier

Run only in the writable checkout `~/projects/tda_data_poisoning_prevention`. Do not run in the read-only `~/wire` view. The commands below assume the version-controlled design and runner have already been committed and pushed from `C:\TDA`.

## Phase 1: prepare and externally register (no outcomes)

```bash
cd ~/projects/tda_data_poisoning_prevention
git pull --ff-only
source venv312/bin/activate
set -a
source .env.wire
set +a

python programs/verify_env.py
python tools/test_downstream_classifier_preregistered.py
python programs/run_downstream_classifier_preregistered.py --write-design
python programs/run_downstream_classifier_preregistered.py --prepare-only
python programs/run_downstream_classifier_preregistered.py --audit-preregistration
```

The final command must report `passed: true`, no feature cache, no outcome cells, exact UNSW reproduction, corrected CICIDS attacks, malicious-only poison parents, and raw-identity-disjoint tests.

At this point, stop. Do not use `--run`. Commit and push only the generated frozen manifest:

```bash
git add results/downstream_classifier_preregistration.json
git commit -m "Freeze downstream classifier preregistration"
git push
git rev-parse HEAD
```

Upload the human-readable record, machine-readable design, frozen manifest, and the exact Git commit identifier to an immutable OSF registration. Record the registration time in UTC exactly as displayed by OSF.

Create the local receipt, replacing all placeholders:

```bash
python programs/run_downstream_classifier_preregistered.py \
  --record-registration \
  --registration-url https://osf.io/ABCDE/ \
  --registered-at-utc 2026-09-XXTXX:XX:XXZ \
  --registered-code-commit FULL_40_CHARACTER_GIT_SHA \
  --visibility public

git add results/downstream_classifier_registration_receipt.json
git commit -m "Record OSF downstream preregistration"
git push
```

Use `--visibility embargoed` if the actual OSF registration is embargoed. The URL must be the immutable registration, not an editable project page.

## Phase 2: execute registered outcomes

Start from a clean, synchronized WIRE checkout containing the receipt:

```bash
cd ~/projects/tda_data_poisoning_prevention
git pull --ff-only
source venv312/bin/activate
set -a
source .env.wire
set +a

git status --short --branch
python programs/verify_env.py
python tools/test_downstream_classifier_preregistered.py
python tools/repro_check.py --expect 2.2000
python programs/run_downstream_classifier_preregistered.py --run
python programs/run_downstream_classifier_preregistered.py --merge
python programs/run_downstream_classifier_preregistered.py --audit
```

The first `--run` invocation verifies both mounted dataset hashes and reconstructs any missing ignored raw caches. It then executes the 45 cells sequentially. This is intentional: the TDA pipelines already use all available cores internally. Completed atomic cells are skipped on a safe restart.

If the prior `.confirmation_cache/` still exists, exact-hash-verified UNSW 60/540 feature matrices may be reused automatically. CICIDS feature caches are never reused; all corrected CICIDS representations and detector decisions are computed after registration.

To resume or isolate a cell without overwriting completed work:

```bash
python programs/run_downstream_classifier_preregistered.py \
  --run --population unsw_matched --seed 2026 --family transpositions
```

Do not use `--overwrite` for a completed registered cell unless a documented protocol amendment explicitly authorizes it. Do not run several TDA-generating cells concurrently; that would oversubscribe the node.

Expected tracked outputs after a successful merge are:

- `results/downstream_classifier_cells/*.json` (45 atomic cells)
- `results/downstream_classifier_results.json`
- `results/downstream_classifier_summary.csv`
- `results/downstream_classifier_cell_metrics.csv`
- `docs/DOWNSTREAM_CLASSIFIER_RESULTS.md`

The `.downstream_cache/` directory is ignored and disposable. Before ending the WIRE instance, commit and push the tracked results, then verify that local `HEAD` and `origin/main` (or the active branch) are synchronized. Do not include dataset files, virtual environments, or caches.

## Required terminal evidence

Preserve the terminal output showing:

- environment verification and historical 2.2000% gate pass;
- 45/45 unique registered cells;
- zero preregistration, receipt, test-identity, cost-matching, or finite-metric failures;
- the four primary gate decisions;
- clean/oracle equality;
- successful process exit and a clean Git synchronization check.
