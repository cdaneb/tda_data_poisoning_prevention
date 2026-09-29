"""Generate publication figures from the preregistered downstream results.

This script is deliberately read-only with respect to experiment outcomes.  It
loads the committed merged JSON and cell-metrics CSV, validates the JSON content
hash, and writes the two paper figures in PDF and PNG form.

Usage:
    python programs/make_figure_downstream_classifier.py
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "downstream_classifier_results.json"
CELL_METRICS = ROOT / "results" / "downstream_classifier_cell_metrics.csv"
FIGURES = ROOT / "figures"

INK = "#111111"
MUTED = "#666666"
GRID = "#d9d9d9"
NAVY = "#2f5597"
BLUE = "#4e79a7"
ORANGE = "#e76f51"
LIGHT_BLUE = "#9ecae1"
LIGHT_ORANGE = "#f4a582"

EXPECTED_PREREGISTRATION_HASH = (
    "3451b9eeb5375434ad6ebaeb8273b32eb6a308fc40cdf5ff05f299de97105ade"
)
POPULATIONS = (
    ("unsw_matched", "Matched-size\nUNSW"),
    ("cicids_matched", "Matched-size\nCICIDS"),
    ("cicids_scale", "Large-scale\nCICIDS"),
)


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_results():
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    body = copy.deepcopy(result)
    recorded_hash = body.pop("content_hash", None)
    if recorded_hash != stable_hash(body):
        raise RuntimeError("downstream result content hash does not validate")
    if result.get("preregistration_hash") != EXPECTED_PREREGISTRATION_HASH:
        raise RuntimeError("unexpected downstream preregistration hash")
    if result.get("cell_count") != 45 or result.get("cell_metric_rows") != 495:
        raise RuntimeError("refusing to plot an incomplete downstream result")
    return result


def save_both(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def primary_figure(result):
    primary = result["primary"]
    effects = primary["effects"]
    rows = (
        ("Poisoning harm\n(clean-training recall - poisoned-training recall)", "attack_harm", 2.0, "mean >= 2.0"),
        ("Isolation Forest recovery\n(60-feature filter - poisoned recall)", "if60_recovery", 1.0, "mean >= 1.0"),
        ("Gain beyond random removal\n(60-feature filter - equal-cost random recall)", "if60_vs_random", 0.0, "mean > 0"),
        ("Benign false-positive-rate change\n(60-feature filter - clean-training FPR)", "if60_fpr_inflation", 1.0, "CI upper <= 1.0"),
    )
    means = np.array([100 * effects[key]["mean"] for _, key, _, _ in rows])
    lows = np.array([100 * effects[key]["ci95"][0] for _, key, _, _ in rows])
    highs = np.array([100 * effects[key]["ci95"][1] for _, key, _, _ in rows])
    passed = np.array([
        primary["gates"]["attack_harm"],
        primary["gates"]["if60_recovery"],
        primary["gates"]["if60_vs_random"],
        primary["gates"]["if60_fpr_guardrail"],
    ])

    fig, ax = plt.subplots(figsize=(7.15, 3.55), facecolor="white")
    fig.subplots_adjust(left=0.34, right=0.98, top=0.86, bottom=0.22)
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK, linewidth=1.0, zorder=1)
    for index, ypos in enumerate(y):
        color = NAVY if passed[index] else ORANGE
        ax.errorbar(
            means[index], ypos,
            xerr=[[means[index] - lows[index]], [highs[index] - means[index]]],
            fmt="o", markersize=7.5, color=color, ecolor=color,
            elinewidth=2.0, capsize=4, capthick=1.5, zorder=3,
        )
        threshold = rows[index][2]
        ax.scatter(threshold, ypos, marker="D", s=38, facecolor="white",
                   edgecolor=INK, linewidth=1.1, zorder=4)
        ax.text(
            4.08, ypos,
            f"{'PASS' if passed[index] else 'FAIL'}   "
            f"{means[index]:.2f} [{lows[index]:.2f}, {highs[index]:.2f}]",
            ha="right", va="center", fontsize=8.8, fontweight="bold", color=color,
        )
        ax.text(threshold, ypos - 0.31, rows[index][3], ha="center", va="top",
                fontsize=7.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _, _ in rows], fontsize=9.3)
    ax.set_xlim(-1.65, 4.15)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xlabel("Difference in percentage points\n(95% hierarchical bootstrap interval)",
                  fontsize=9.5, fontweight="bold")
    ax.set_title("Primary preregistered result: two of four criteria pass",
                 fontsize=11.3, fontweight="bold", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=7)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.text(0.0, -0.30, "Circle = observed mean and interval; diamond = prespecified threshold",
            transform=ax.transAxes, fontsize=8.0, color=MUTED)
    return save_both(fig, "figure_v10_downstream_primary")


def sanitizer_means():
    grouped = defaultdict(list)
    with CELL_METRICS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[(row["population"], row["arm"])].append(row)

    output = {}
    for population, _ in POPULATIONS:
        clean_rows = grouped[(population, "clean")]
        poisoned_rows = grouped[(population, "poisoned")]
        clean_count = np.mean([float(row["training_rows"]) for row in clean_rows])
        poison_count = np.mean([float(row["poison_retained"]) for row in poisoned_rows])
        output[population] = {}
        for arm in ("filter|control|0.05", "random|control|0.05"):
            rows = grouped[(population, arm)]
            training = np.mean([float(row["training_rows"]) for row in rows])
            retained = np.mean([float(row["poison_retained"]) for row in rows])
            poison_removed = poison_count - retained
            clean_removed = clean_count - (training - retained)
            output[population][arm] = {
                "poison_removed_pct": 100 * poison_removed / poison_count,
                "clean_removed_pct": 100 * clean_removed / clean_count,
            }
    return output


def generalization_figure(result):
    fig, (ax_effect, ax_remove) = plt.subplots(
        1, 2, figsize=(7.15, 3.25), facecolor="white",
        gridspec_kw={"width_ratios": [1.18, 0.92]},
    )
    # Reserve the far-right margin for the panel-b legend so it never covers
    # the bars or value labels.
    fig.subplots_adjust(left=0.10, right=0.79, top=0.84, bottom=0.24, wspace=0.34)

    # Panel A: classifier benefit beyond equally costly random removal.
    y_base = np.arange(len(POPULATIONS))[::-1]
    offsets = {"control": 0.12, "stack": -0.12}
    styles = {
        "control": (NAVY, "o", "Isolation Forest, 60 features"),
        "stack": (ORANGE, "s", "Isolation Forest, 540 features"),
    }
    for representation in ("control", "stack"):
        color, marker, label = styles[representation]
        for index, (population, _) in enumerate(POPULATIONS):
            effect = result["secondary_effects"][
                f"{population}|{representation}|0.05"
            ]["filter_minus_random_malicious_recall"]
            mean = 100 * effect["mean"]
            low, high = (100 * value for value in effect["ci95"])
            ypos = y_base[index] + offsets[representation]
            ax_effect.errorbar(
                mean, ypos, xerr=[[mean - low], [high - mean]], fmt=marker,
                markersize=5.5, color=color, ecolor=color, elinewidth=1.6,
                capsize=3, capthick=1.1, label=label if index == 0 else None,
            )
    ax_effect.axvline(0, color=INK, linewidth=1.0)
    ax_effect.set_yticks(y_base)
    ax_effect.set_yticklabels([label for _, label in POPULATIONS], fontsize=8.7)
    ax_effect.set_xlabel("Malicious-recall gain over\nequal-cost random removal (percentage points)",
                         fontsize=8.8, fontweight="bold")
    ax_effect.set_title("(a) Classifier benefit beyond\nrandom removal", loc="left",
                        fontsize=9.6, fontweight="bold")
    ax_effect.set_xlim(-0.38, 1.03)
    ax_effect.grid(axis="x", color=GRID, linewidth=0.8)

    # Panel B: why the cross-dataset effect changes sign.
    removal = sanitizer_means()
    x = np.arange(len(POPULATIONS))
    width = 0.34
    filter_values = [
        removal[population]["filter|control|0.05"]["poison_removed_pct"]
        for population, _ in POPULATIONS
    ]
    random_values = [
        removal[population]["random|control|0.05"]["poison_removed_pct"]
        for population, _ in POPULATIONS
    ]
    bars_filter = ax_remove.bar(x - width / 2, filter_values, width, color=BLUE,
                                label="Isolation Forest removal\n(60 features)")
    bars_random = ax_remove.bar(x + width / 2, random_values, width,
                                color=LIGHT_ORANGE, edgecolor=ORANGE,
                                linewidth=0.8, label="Equal-cost random\nremoval")
    ax_remove.bar_label(bars_filter, fmt="%.1f", padding=2, fontsize=7.5)
    ax_remove.bar_label(bars_random, fmt="%.1f", padding=2, fontsize=7.5)
    ax_remove.set_xticks(x)
    ax_remove.set_ylabel("Poison removed (%)", fontsize=8.8, fontweight="bold")
    ax_remove.set_title("(b) Poison-removal selectivity\nat 5% calibration target", loc="left",
                        fontsize=9.6, fontweight="bold")
    ax_remove.set_ylim(0, 16.5)
    ax_remove.grid(axis="y", color=GRID, linewidth=0.8)
    ax_remove.legend(
        frameon=False,
        fontsize=7.4,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    clean_costs = [
        removal[population]["filter|control|0.05"]["clean_removed_pct"]
        for population, _ in POPULATIONS
    ]
    for ax in (ax_effect, ax_remove):
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=MUTED, labelsize=8.0)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    ax_remove.set_xticklabels(
        ("UNSW\nmatched", "CICIDS\nmatched", "CICIDS\nlarge-scale"),
        fontsize=7.4,
    )
    ax_remove.text(
        1.02,
        0.49,
        "Unmodified rows removed:\n"
        f"Matched-size UNSW  {clean_costs[0]:.2f}%\n"
        f"Matched-size CICIDS  {clean_costs[1]:.2f}%\n"
        f"Large-scale CICIDS  {clean_costs[2]:.2f}%",
        transform=ax_remove.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=MUTED,
        linespacing=1.25,
    )
    effect_handles, effect_labels = ax_effect.get_legend_handles_labels()
    fig.legend(
        effect_handles,
        effect_labels,
        title="Panel (a)",
        frameon=False,
        fontsize=7.4,
        title_fontsize=7.4,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.28),
        borderaxespad=0.0,
    )
    return save_both(fig, "figure_v11_downstream_generalization")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    result = load_results()
    outputs = primary_figure(result) + generalization_figure(result)
    for output in outputs:
        print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
