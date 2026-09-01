"""
Figure V3 — four-family permutation comparison (Phase V1).

The poster's Results centerpiece. Encodes Claim 3: detectability is not
ordered by attack realism, it tracks *spatial disruption*. Families are
ordered left-to-right by mean positions value-changed, which makes the
capture ordering read as a consequence rather than a coincidence.

Capture and strict-purity ceilings are parsed from the M7 artifact
`results/phase_m_m7_capture.json`; spatial-disruption statistics are read from
the validated Test B diagnostics artifact. Nothing is hand-entered from
CLAUDE.md. The
guided-search reference at 6.48% is parsed from
`results/lens4_baseline_multiseed.json` (cell G60-MLP), the one committed
artifact backing it.

Capture mean and spread are recomputed here from the per-seed values with
population SD (ddof=0), matching the project convention, and are then
checked against CLAUDE.md §6 before anything is drawn. A mismatch is a hard
stop — `--check` exits nonzero and `--render` refuses to draw.

Usage:
    python make_figure_v3.py --check     # parse + verify, print the table
    python make_figure_v3.py --render    # verify, then write the PDF
"""
import argparse
import json
import sys

import numpy as np

from programs.paths import RESULTS_DIR, FIGURES_DIR

SEEDS = ["42", "123", "456", "789", "1024"]
ALGO = "OPTICS"

# Display order == ascending mean positions value-changed. Set from the data
# below, not asserted here; this list only fixes the labels.
# DELIBERATELY DECOUPLED (Phase M, A13): these four names are the Test B families
# and must NOT be regenerated from either driver's FAMILIES dict. Both drivers now
# carry a fifth "noise" entry, which is a count-channel positive control, not a
# Test B family; sourcing this list from them would silently add a fifth bar to
# the poster's centerpiece figure.
FAMILIES = [
    ("block_reversal", "Block reversal", r"$k=120$"),
    ("block_swap",     "Block swap",     r"$2\times k=60$"),
    ("transpositions", "Transpositions", r"$60$ swaps"),
    ("cyclic_shift",   "Cyclic shift",   "full rotation"),
]

# CLAUDE.md §6, for the blocking cross-check only. Never plotted.
REFERENCE_S6 = {
    "block_reversal": {"capture_mean": 0.00, "capture_sd": 0.00, "pos_mean": 14.08,
                       "pos_median": 0.0, "frac_zero": 0.845},
    "block_swap":     {"capture_mean": 0.00, "capture_sd": 0.00, "pos_mean": 17.63,
                       "pos_median": 0.0, "frac_zero": 0.785},
    "transpositions": {"capture_mean": 1.80, "capture_sd": 0.51, "pos_mean": 22.93,
                       "pos_median": 8.0, "frac_zero": 0.125},
    "cyclic_shift":   {"capture_mean": 6.28, "capture_sd": 1.31, "pos_mean": 281.09,
                       "pos_median": 89.0, "frac_zero": 0.0},
}
REFERENCE_SG = {"capture_mean": 6.48, "capture_sd": 1.24}

# Light-surface print palette (dataviz reference instance). Capture is the
# subject and carries the one series hue; the footprint panel is context and
# wears neutral ink; the guided-search reference is not a Test B family and
# takes the second categorical slot so it can never be misread as one.
C_SERIES = "#2a78d6"
C_REFERENCE = "#eb6834"
C_CONTEXT = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse():
    """Parse and recompute every plotted quantity from the artifacts."""
    fam_path = RESULTS_DIR / "phase_m_m7_capture.json"
    diag_path = RESULTS_DIR / "test_b_diagnostics.json"
    sg_path = RESULTS_DIR / "lens4_baseline_multiseed.json"

    fam = json.load(open(fam_path))
    diag = json.load(open(diag_path))["p5_swap_fraction_bit_identity"]
    sg = json.load(open(sg_path))

    rows = []
    for key, label, sublabel in FAMILIES:
        per_seed = [fam[key][s]["per_algo"][ALGO]["red_poison_capture_pct"] for s in SEEDS]
        a = np.array(per_seed, dtype=float)
        d = diag[key]
        rows.append({
            "key": key, "label": label, "sublabel": sublabel,
            "per_seed": per_seed,
            "capture_mean": float(a.mean()),
            "capture_sd": float(a.std(ddof=0)),
            "purity_ceiling": float(np.mean([
                fam[key][s]["per_algo"][ALGO]["ceilings"]["union_ceiling"]
                for s in SEEDS
            ]) * 100),
            "pos_mean": float(d["positions_changed_mean"]),
            "pos_median": float(d["positions_changed_median"]),
            "frac_zero": float(d["positions_changed_frac_zero"]),
        })

    sg_per_seed = [sg["G60-MLP"][s][ALGO]["red_poison_capture_pct"] for s in SEEDS]
    a = np.array(sg_per_seed, dtype=float)
    sg_row = {"per_seed": sg_per_seed,
              "capture_mean": float(a.mean()),
              "capture_sd": float(a.std(ddof=0))}

    return rows, sg_row


def check(rows, sg_row, tol=5e-3):
    """Blocking cross-check of every parsed value against CLAUDE.md §6."""
    failures = []

    print(f"{'Family':<16}{'Capture % (parsed)':>22}{'§6':>16}"
          f"{'Mean pos':>11}{'§6':>9}{'Median':>9}{'§6':>8}"
          f"{'% zero fp':>12}{'§6':>9}")
    print("-" * 113)
    for r in rows:
        ref = REFERENCE_S6[r["key"]]
        parsed_cap = f"{r['capture_mean']:.4f} ± {r['capture_sd']:.4f}"
        ref_cap = f"{ref['capture_mean']:.2f} ± {ref['capture_sd']:.2f}"
        print(f"{r['label']:<16}{parsed_cap:>22}{ref_cap:>16}"
              f"{r['pos_mean']:>11.2f}{ref['pos_mean']:>9.2f}"
              f"{r['pos_median']:>9.1f}{ref['pos_median']:>8.1f}"
              f"{100 * r['frac_zero']:>11.1f}%{100 * ref['frac_zero']:>8.1f}%")
        for field, refkey, scale in [("capture_mean", "capture_mean", 1),
                                      ("capture_sd", "capture_sd", 1),
                                      ("pos_mean", "pos_mean", 1),
                                      ("pos_median", "pos_median", 1),
                                      ("frac_zero", "frac_zero", 1)]:
            if abs(r[field] * scale - ref[refkey]) > tol:
                failures.append(f"{r['label']}.{field}: parsed {r[field]!r} vs §6 {ref[refkey]!r}")

    print("-" * 113)
    sg_cap = f"{sg_row['capture_mean']:.4f} ± {sg_row['capture_sd']:.4f}"
    ref_cap = f"{REFERENCE_SG['capture_mean']:.2f} ± {REFERENCE_SG['capture_sd']:.2f}"
    print(f"{'Guided (SG)':<16}{sg_cap:>22}{ref_cap:>16}"
          f"{'—':>11}{'—':>9}{'—':>9}{'—':>8}{'—':>12}{'—':>9}")
    for field in ("capture_mean", "capture_sd"):
        if abs(sg_row[field] - REFERENCE_SG[field]) > tol:
            failures.append(f"SG.{field}: parsed {sg_row[field]!r} vs §6 {REFERENCE_SG[field]!r}")

    # The display order must actually be ascending spatial disruption, or the
    # figure's central claim is not what the geometry shows.
    pos = [r["pos_mean"] for r in rows]
    if pos != sorted(pos):
        failures.append(f"display order is not ascending in mean positions changed: {pos}")

    print()
    print("Per-seed capture % (population SD over these 5 seeds; none appears in the figure):")
    for r in rows:
        print(f"  {r['label']:<16}{r['per_seed']}")
    print(f"  {'Guided (SG)':<16}{sg_row['per_seed']}")

    return failures


def render(rows, sg_row, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    # Poster aspect (V7.2). The figure sits in the 0.33 column at 0.98\linewidth,
    # so its rendered height on the board is set entirely by this ratio. Dropping
    # 6.0 -> 3.6 in is what pays down the right column's overflow. The annotation
    # offsets below are in *points* and do not scale with the canvas, so the
    # margins and hspace are retuned here rather than left at their 6.0in values.
    fig = plt.figure(figsize=(7.5, 3.6))
    gs = GridSpec(3, 1, height_ratios=[3.0, 1.30, 1.05], hspace=0.30,
                  left=0.105, right=0.985, top=0.790, bottom=0.020)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax3 = fig.add_subplot(gs[2], sharex=ax)

    x = np.arange(len(rows))
    caps = [r["capture_mean"] for r in rows]
    sds = [r["capture_sd"] for r in rows]

    # ---- Panel A: capture % -------------------------------------------------
    band = ax.axhspan(sg_row["capture_mean"] - sg_row["capture_sd"],
                      sg_row["capture_mean"] + sg_row["capture_sd"],
                      color=C_REFERENCE, alpha=0.13, lw=0, zorder=1)
    ax.axhline(sg_row["capture_mean"], color=C_REFERENCE, lw=2.0, ls=(0, (5, 3)), zorder=2)
    # Annotated on the left, over the two zero-height bars, so it cannot
    # collide with the cyclic-shift bar or its error bar.
    ax.text(-0.55, sg_row["capture_mean"] + 0.28,
            f"Surrogate-guided search  {sg_row['capture_mean']:.2f}% "
            f"± {sg_row['capture_sd']:.2f}",
            ha="left", va="bottom", fontsize=9.5, color=C_REFERENCE, fontweight="bold",
            zorder=6)
    ax.text(-0.55, sg_row["capture_mean"] - 0.34,
            "reference — not a Test B family",
            ha="left", va="top", fontsize=8.2, color=C_REFERENCE, style="italic", zorder=6)

    ax.bar(x, caps, width=0.60, color=C_SERIES, zorder=3)
    # Only the families with nonzero spread get caps; a zero-SD cap would draw
    # a stray tick across the zero rule and read as chart junk. The 0.00 ± 0.00
    # cells carry their spread in the label instead.
    nz = [i for i, s in enumerate(sds) if s > 0]
    ax.errorbar([x[i] for i in nz], [caps[i] for i in nz], yerr=[sds[i] for i in nz],
                fmt="none", ecolor=INK, elinewidth=1.5,
                capsize=6, capthick=1.5, zorder=5)

    for xi, (c, s) in enumerate(zip(caps, sds)):
        if c == 0.0:
            # A measured zero, not a missing bar: an explicit zero rule plus a
            # labelled value. The 0.00 +/- 0.00 cells are results.
            ax.plot([xi - 0.30, xi + 0.30], [0, 0], color=C_SERIES, lw=3.2,
                    solid_capstyle="butt", zorder=4)
            ax.annotate("0.00", xy=(xi, 0), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                        color=C_SERIES, zorder=6)
            ax.annotate("measured zero", xy=(xi, 0), xytext=(0, 21),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.2, color=INK_MUTED, style="italic", zorder=6)
        else:
            ax.annotate(f"{c:.2f}", xy=(xi, c + s), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=11.5, fontweight="bold", color=INK, zorder=6)

    ax.set_ylim(0, 9.2)
    ax.text(-0.55, 8.55, "Capture (%)", fontsize=8.6, color=INK,
            ha="left", va="bottom")
    # Both ylabels are pinned to the same axes-x so they form one left rail;
    # left to themselves they track their own tick-label widths and, at the
    # compressed poster aspect, drift into each other. They are also nudged
    # apart along y: Panel B's label is longer than Panel B is tall at this
    # aspect, so it overhangs its panel and would meet Panel A's label.
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK_SECONDARY, length=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---- Panel B: strict-purity ceiling (M7 mechanism) ---------------------
    ceilings = [r["purity_ceiling"] for r in rows]
    ax2.bar(x, ceilings, width=0.60, color=C_CONTEXT, zorder=3)
    for xi, v in enumerate(ceilings):
        ax2.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold",
                     color=INK_SECONDARY, zorder=6)
    ax2.set_ylim(0, 16)
    ax2.set_yticks([0, 8, 16])
    # One line, not two: the two-line form is taller than this panel at the
    # poster aspect and rides up into Panel A's ylabel.
    ax2.text(-0.55, 14.8, "100%-pure ceiling (%)", fontsize=8.2, color=INK,
             ha="left", va="bottom")
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", labelsize=9.5, colors=INK_SECONDARY, length=0)
    ax2.tick_params(axis="x", length=0)
    for side in ("top", "right", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["left"].set_color(BASELINE)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ---- Panel C: category labels + numeric strip ---------------------------
    # The category labels live in this panel rather than hanging off ax2 on a
    # point offset. At 6.0in the offsets cleared panel C; at the poster aspect
    # the gap shrinks but the offsets do not, and the labels landed on top of
    # the numbers. Stacking all four rows in one axes' coordinates keeps them
    # separated at any canvas height.
    ax3.axis("off")
    ax3.set_ylim(0, 1)
    ROWS = [("", 0.88, 10.5, "bold", INK),          # family label
            ("", 0.63, 8.4, "normal", INK_MUTED),   # sublabel
            ("", 0.25, 9.2, "normal", INK_SECONDARY),
            ("", 0.04, 9.2, "normal", INK_SECONDARY)]
    for xi, r in enumerate(rows):
        vals = [r["label"], r["sublabel"], f"{r['pos_mean']:.2f}", f"{r['pos_median']:.0f}"]
        for v, (_lab, y, fs, w, c) in zip(vals, ROWS):
            ax3.text(xi, y, v, ha="center", va="center",
                     fontsize=fs, fontweight=w, color=c)

    ax.set_xlim(-0.68, len(rows) - 0.32)
    ax2.set_xticks(x)
    ax2.set_xticklabels([])

    fig.text(0.030, 0.950,
             "Detectability tracks spatial disruption, not attack sophistication",
             fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.030, 0.902, "UNSW-NB15  ·  OPTICS",
             fontsize=8.8, color=INK_MUTED, ha="left", va="top")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or (FIGURES_DIR / "figure_v3_four_families.pdf")
    # Raster only for visual QA; the poster artifact is always the vector PDF.
    fig.savefig(out, dpi=200, transparent=(out.suffix == ".pdf"))
    plt.close(fig)
    return out, fig.get_size_inches()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="parse + verify only")
    p.add_argument("--render", action="store_true", help="verify, then write the PDF")
    p.add_argument("--out", default=None,
                   help="override output path (e.g. a .png for visual QA)")
    args = p.parse_args()
    if not (args.check or args.render):
        p.error("pass --check or --render")

    rows, sg_row = parse()
    failures = check(rows, sg_row)

    print()
    if failures:
        print("FAIL — parsed values do not match CLAUDE.md §6:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("PASS — every parsed value matches CLAUDE.md §6 within 5e-3.")

    if args.render:
        from pathlib import Path
        out, size = render(rows, sg_row, Path(args.out) if args.out else None)
        print(f"\nWrote {out}")
        print(f"Figure size: {size[0]:.2f} x {size[1]:.2f} in "
              f"({size[0] * 25.4:.0f} x {size[1] * 25.4:.0f} mm), vector PDF, "
              f"transparent background")


if __name__ == "__main__":
    main()
