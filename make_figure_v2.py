"""
Figure V2 — binarized triptych: clean / permuted / noised (Phase V2).

Goes in the poster's Methods I block, adjacent to the invariance argument it
illustrates. Shows one packet's binarized 30x50 image under three treatments
with the foreground count beneath each: a permutation rearranges foreground
pixels and leaves the count exactly unchanged; Gaussian noise changes byte
values, crosses the threshold, and moves the count.

SINGLE COMBINED FIT. gtda's Binarizer cuts at a fraction of the *fitted*
max_value_ = np.max over the collection it is fit on, so binarizing clean and
perturbed batches separately can in principle put them on different scales and
manufacture spurious differences (CLAUDE.md §9 item 8, the Scaler analogue that
produced 16/200 false discrepancies). Every branch here is concatenated into one
array and binarized in ONE Binarizer.fit_transform call -- see
`binarize_all_branches()`, the single `binarize(...)` call marked SINGLE FIT --
then split back apart by offset.

Note that Phase P's run_p4() (test_b_diagnostics.py:117-118, 130-131) instead
fits clean and perturbed separately. For permutations that is provably harmless,
because max_value_ is a maximum and a maximum is permutation-invariant -- which
is exactly what max_value_check() verifies rather than assumes. This module
re-derives the same quantities under the stricter single-fit discipline so the
two can be compared; agreement is reported as a gate, not assumed.

Usage:
    python make_figure_v2.py --check     # population gates + sample selection
    python make_figure_v2.py --render    # gates, then write the PDF
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from paths import FIGURES_DIR
from data_loader import load_unsw
from invariance_check import binarize, max_value_check
from tda_pipeline import reshape_for_tda
from test_b_diagnostics import (
    FAMILIES, N_DIAG_SAMPLES, RANDOM_STATE,
    get_clean_and_perturbed, noise_only,
)

THRESHOLD = 0.4
NOISE_SCALE = 30

# The permuted panel. Cyclic shift is a permutation in S_1500 exactly as the
# other three are, and it is the one whose rearrangement is legible at poster
# distance -- 60 transpositions move a median of 8 byte positions out of 1500
# and render as visually indistinguishable from clean. Disclosed on the figure
# itself; the count-invariance gate below covers all four families regardless.
PANEL_FAMILY = "cyclic_shift"

# Phase P reference values (CLAUDE.md §6), for the blocking cross-check.
EXPECTED_CLEAN_MEAN = 62.305
EXPECTED_NOISE_ABS_DELTA = 14.02

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
C_SERIES = "#2a78d6"
C_REFERENCE = "#eb6834"
GRID = "#e1e0d9"


def binarize_all_branches(X, y):
    """
    Build every branch, concatenate, and binarize in ONE fit.

    Returns a dict of branch-name -> (binarized images, foreground counts),
    plus the fitted max_value_ shared by all of them and the per-branch raw
    maxima used for the max_value_ scope caveat.
    """
    branches = {}   # name -> raw (200, 1500) byte array
    order = []

    clean_ref = None
    for family, (fn, kwargs) in FAMILIES.items():
        X_clean, X_perm, _ = get_clean_and_perturbed(fn, kwargs, X, y)
        if clean_ref is None:
            clean_ref = X_clean
            branches["clean"] = X_clean
            order.append("clean")
        else:
            # All four families must draw the same 200 targets, or "clean" is
            # not a shared baseline and the whole comparison is unfounded.
            assert np.array_equal(clean_ref, X_clean), \
                f"{family} drew different clean targets than the first family"
        branches[family] = X_perm
        order.append(family)

    X_clean_n, X_noisy, _ = noise_only(X, y, noise_scale=NOISE_SCALE)
    assert np.array_equal(clean_ref, X_clean_n), \
        "noise_only drew different clean targets than the attack families"
    branches["noise"] = X_noisy
    order.append("noise")

    per_branch_max = {name: float(np.max(branches[name])) for name in order}

    stacked = np.concatenate([reshape_for_tda(branches[n]) for n in order], axis=0)

    # >>> SINGLE FIT <<< one Binarizer, fit on every branch at once.
    binarized_all, max_value_ = binarize(stacked, threshold=THRESHOLD)

    n = len(clean_ref)
    out = {}
    for i, name in enumerate(order):
        b = binarized_all[i * n:(i + 1) * n]
        out[name] = (b, b.reshape(len(b), -1).sum(axis=1))

    return out, max_value_, per_branch_max, branches, order


def check(res, max_value_, per_branch_max, branches, order):
    failures = []

    print(f"Sample frame: {N_DIAG_SAMPLES} malicious-only targets, "
          f"random_state={RANDOM_STATE}, threshold={THRESHOLD}")
    print(f"SINGLE combined fit over {len(order)} branches "
          f"({len(order)} x {N_DIAG_SAMPLES} = {len(order) * N_DIAG_SAMPLES} images), "
          f"one Binarizer.fit_transform")
    print(f"Fitted max_value_ (shared by every branch): {max_value_}")
    print()

    print(f"{'Branch':<18}{'raw max':>10}{'== clean':>10}{'mean fg count':>16}"
          f"{'Δcount vs clean':>18}")
    print("-" * 72)
    clean_counts = res["clean"][1]
    for name in order:
        counts = res[name][1]
        raw_max = per_branch_max[name]
        same_max = "yes" if raw_max == per_branch_max["clean"] else "NO"
        if name == "clean":
            delta = "—"
        else:
            n_changed = int((counts != clean_counts).sum())
            delta = f"{n_changed}/{len(counts)} changed"
        print(f"{name:<18}{raw_max:>10.1f}{same_max:>10}{counts.mean():>16.3f}{delta:>18}")
    print("-" * 72)
    print()

    # --- Gate 1: clean mean foreground count -------------------------------
    clean_mean = float(clean_counts.mean())
    print(f"Gate 1  clean mean foreground count: {clean_mean:.3f} / 1500 "
          f"(expected {EXPECTED_CLEAN_MEAN})")
    if abs(clean_mean - EXPECTED_CLEAN_MEAN) > 5e-3:
        failures.append(f"clean mean foreground count {clean_mean!r} != {EXPECTED_CLEAN_MEAN}")

    # --- Gate 2: count invariance, all four families ------------------------
    for family in FAMILIES:
        n_changed = int((res[family][1] != clean_counts).sum())
        print(f"Gate 2  {family:<16} count change: {n_changed}/{len(clean_counts)} (expected 0)")
        if n_changed != 0:
            failures.append(f"{family} changed the count in {n_changed} samples")

    # --- Gate 3: max_value_ identical clean vs each permuted ---------------
    for family in FAMILIES:
        mc, mp, eq = max_value_check(branches["clean"], branches[family])
        print(f"Gate 3  {family:<16} max_value_ clean={mc} perm={mp} equal={eq}")
        if not eq:
            failures.append(f"{family} max_value_ differs: {mc} vs {mp}")

    # --- Gate 4: noise positive control ------------------------------------
    delta = res["noise"][1].astype(int) - clean_counts.astype(int)
    mean_abs = float(np.abs(delta).mean())
    print(f"Gate 4  noise sigma={NOISE_SCALE} mean |Δ| = {mean_abs:.3f} "
          f"(expected {EXPECTED_NOISE_ABS_DELTA}), mean Δ = {delta.mean():.3f}")
    if abs(mean_abs - EXPECTED_NOISE_ABS_DELTA) > 5e-3:
        failures.append(f"noise mean |Δ| {mean_abs!r} != {EXPECTED_NOISE_ABS_DELTA}")

    return failures


def select_sample(res, verbose=False):
    """
    Representative-sample criterion, disclosed on the figure.

    The panel has to be typical on BOTH axes it depicts, so a criterion on the
    clean count alone is not enough: selecting purely on it returns local index
    151, whose noise delta happens to be -1 against a population mean |delta| of
    14.02. That packet would understate the noise channel by a factor of ~14 and
    make the positive control look like it did nothing.

    So: minimise the sum of two standardised deviations -- clean count from its
    population mean, and |delta| under noise from its population mean. The
    result is the jointly most typical packet. Deterministic, ties to the lowest
    index, no separate seed, and no search over which sample looks best.
    """
    clean = res["clean"][1].astype(float)
    delta = np.abs(res["noise"][1].astype(int) - res["clean"][1].astype(int)).astype(float)

    z_clean = np.abs(clean - clean.mean()) / clean.std(ddof=0)
    z_delta = np.abs(delta - delta.mean()) / delta.std(ddof=0)
    score = z_clean + z_delta
    idx = int(np.argmin(score))

    if verbose:
        print(f"  criterion: min |z(clean count)| + |z(|Δ| under noise)|")
        print(f"    population clean count  mean={clean.mean():.3f} sd={clean.std(ddof=0):.3f}")
        print(f"    population |Δ| (noise)  mean={delta.mean():.3f} sd={delta.std(ddof=0):.3f}")
        print(f"    chosen idx={idx}  clean={clean[idx]:.0f} (z={z_clean[idx]:.3f})  "
              f"|Δ|={delta[idx]:.0f} (z={z_delta[idx]:.3f})  score={score[idx]:.3f}")
    return idx


def render(res, max_value_, idx, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    clean_b, clean_c = res["clean"]
    perm_b, perm_c = res[PANEL_FAMILY]
    noise_b, noise_c = res["noise"]

    panels = [
        ("Clean", clean_b[idx], int(clean_c[idx]), INK_SECONDARY, None),
        ("Permuted", perm_b[idx], int(perm_c[idx]), C_SERIES,
         f"$\\sigma \\in S_{{1500}}$ (cyclic shift)"),
        ("Noised", noise_b[idx], int(noise_c[idx]), C_REFERENCE,
         f"Gaussian $\\sigma={NOISE_SCALE}$"),
    ]

    cmap = ListedColormap(["#ffffff", "#0b0b0b"])

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.40))
    fig.subplots_adjust(left=0.035, right=0.965, top=0.72, bottom=0.30, wspace=0.16)

    base = panels[0][2]
    for ax, (title, img, count, color, sub) in zip(axes, panels):
        ax.imshow(img.astype(int), cmap=cmap, interpolation="nearest",
                  aspect="equal", vmin=0, vmax=1)
        for s in ax.spines.values():
            s.set_color(GRID)
            s.set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=color, pad=23)
        if sub:
            ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8.4, color=INK_MUTED)

        delta = count - base
        if title == "Clean":
            dtxt = "baseline"
            dcolor = INK_MUTED
        else:
            dtxt = f"Δ = {delta:+d}" if delta else "Δ = 0"
            dcolor = INK_MUTED if delta == 0 else C_REFERENCE
        ax.text(0.5, -0.075, f"{count} / 1500", transform=ax.transAxes,
                ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
        ax.text(0.5, -0.235, dtxt, transform=ax.transAxes,
                ha="center", va="top", fontsize=9.6,
                fontweight="bold" if delta else "normal", color=dcolor)

    fig.text(0.035, 0.965,
             "Permutation rearranges foreground pixels; it never changes how many",
             fontsize=12.0, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.035, 0.905,
             f"One representative packet, binarized at threshold {THRESHOLD} "
             f"(fitted max_value_ = {max_value_:g}); foreground pixel count beneath",
             fontsize=8.5, color=INK_MUTED, ha="left", va="top")
    fig.text(0.035, 0.135,
             "Illustrative single packet, selected as the jointly most typical of the 200 "
             "targets (smallest combined\nstandardised deviation of clean count and noise "
             "|Δ| from their population means).\n"
             f"Population:  clean {EXPECTED_CLEAN_MEAN}/1500  ·  permutation Δcount 0/200, "
             f"all four families  ·  noise mean |Δ| {EXPECTED_NOISE_ABS_DELTA}.",
             fontsize=7.9, color=INK_MUTED, ha="left", va="top", linespacing=1.5)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or (FIGURES_DIR / "figure_v2_binarized_triptych.pdf")
    fig.savefig(out, dpi=200, transparent=(out.suffix == ".pdf"))
    plt.close(fig)
    return out, fig.get_size_inches()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not (args.check or args.render):
        p.error("pass --check or --render")

    print("Loading full UNSW-NB15 dataset...")
    X, y = load_unsw(max_samples=None)

    res, max_value_, per_branch_max, branches, order = binarize_all_branches(X, y)
    failures = check(res, max_value_, per_branch_max, branches, order)

    print()
    print("Sample selection")
    idx = select_sample(res, verbose=True)
    print(f"Selected sample: local index {idx} of {N_DIAG_SAMPLES} "
          f"(jointly most typical on clean count and noise |Δ|)")
    print(f"  clean  foreground count = {int(res['clean'][1][idx])} / 1500")
    print(f"  {PANEL_FAMILY:<6} foreground count = {int(res[PANEL_FAMILY][1][idx])} / 1500")
    print(f"  noised foreground count = {int(res['noise'][1][idx])} / 1500")
    mc, mp, eq = max_value_check(branches["clean"][idx:idx + 1],
                                 branches[PANEL_FAMILY][idx:idx + 1])
    print(f"  max_value_ for this sample: clean={mc} permuted={mp} equal={eq}")
    print(f"  max_value_ fitted across the single combined fit: {max_value_}")

    print()
    if failures:
        print("FAIL — V2 gates did not pass:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("PASS — all V2 population gates match the Phase P reference values.")

    if args.render:
        out, size = render(res, max_value_, idx, Path(args.out) if args.out else None)
        print(f"\nWrote {out}")
        print(f"Figure size: {size[0]:.2f} x {size[1]:.2f} in "
              f"({size[0] * 25.4:.0f} x {size[1] * 25.4:.0f} mm), vector PDF")


if __name__ == "__main__":
    main()
