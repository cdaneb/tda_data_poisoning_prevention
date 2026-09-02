"""Write the concise Markdown report from the complete merged artifact."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results/clean_novelty_confirmation.json"
OUT = ROOT / "docs/CLEAN_NOVELTY_CONFIRMATION_REPORT.md"


def pct(x): return f"{100*x:.2f}%"
def ci(x): return f"{100*x['mean']:.2f} pp (95% CI {100*x['ci95'][0]:.2f} to {100*x['ci95'][1]:.2f})"


def main():
    d = json.load(open(SOURCE))
    if not d["complete"] or len(d["records"]) != 90: raise RuntimeError("refusing to report partial populations")
    if not d.get("integrity", {}).get("passed"): raise RuntimeError("refusing to report without passed integrity gates")
    a = d["analysis"]
    def agg(pop, det, rep): return next(x for x in a["aggregates"] if x["population"] == pop and x["detector"] == det and x["representation"] == rep and x["budget"] == .05)
    lines = ["# Independent clean-novelty confirmation", "", "## Scope and integrity", "",
             "All 90 frozen cells completed: 40 unseen-seed UNSW matched-size cells, 40 matched-size CICIDS transfer cells, and 10 CICIDS scale cells. All integrity gates passed. Isolation Forest is primary; kNN distance is secondary. No outcome-based detector selection or retuning was performed.", "",
             "## Findings at the frozen 5% clean-removal operating point", "",
             "| Population | Detector | Features | Capture | Clean removal | Clean removed | Poison removed | Precision | AUROC | AUPRC |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pop in ("unsw_matched", "cicids_matched", "cicids_scale"):
        for det in ("isolation_forest", "knn_distance"):
            for rep, n in (("control", 60), ("stack", 540)):
                x=agg(pop,det,rep)
                lines.append(f"| {pop} | {det} | {n} | {pct(x['poison_capture']['mean'])} | {pct(x['clean_removal_rate']['mean'])} | {x['clean_removed']['mean']:.1f} | {x['poison_removed']['mean']:.1f} | {pct(x['precision']['mean'])} | {x['auroc']['mean']:.3f} | {x['auprc']['mean']:.3f} |")
    lines += ["", "## Paired 540−60 poison-capture effects", ""]
    for pop in ("unsw_matched", "cicids_matched", "cicids_scale"):
        for det in ("isolation_forest", "knn_distance"):
            e=a["paired_effects"][f"{pop}|{det}|0.05"]["metrics"]["poison_capture"]
            lines.append(f"- {pop}, {det}: {ci(e)}.")
    lines += ["", "Per-family consistency (positive/negative/zero seed-level paired effects):", ""]
    for x in a["per_family_consistency"]:
        if x["budget"] == .05:
            lines.append(f"- {x['population']}, {x['detector']}, {x['family']}: {x['positive_seeds']}/{x['negative_seeds']}/{x['zero_seeds']}; mean {ci(x['stack_minus_control_poison_capture'])}.")
    lines += ["", "## Frozen decision", "",
              "The primary representation-improvement claim **fails confirmation**. On unseen-seed UNSW, IF-540 met the clean-cost band and exceeded 12% capture, but its paired improvement over IF-60 did not have a 95% CI excluding zero. Matched-size CICIDS transfer was worse with 540 features on average. CICIDS scale behavior was mixed on capture and failed the frozen clean-cost requirement because mean held-out clean removal exceeded 6%.", "",
              "The secondary kNN result is mixed: 540 features improve UNSW mean capture, but degrade matched-size CICIDS and CICIDS-scale mean capture at the 5% operating point. Under the preregistration, the secondary detector cannot rescue the failed primary claim.", "",
              "Figures: `figures/clean_novelty_confirmation_capture.png` and `figures/clean_novelty_confirmation_paired_effects.png`. All values and confidence intervals come from `results/clean_novelty_confirmation.json`.", ""]
    OUT.write_text("\n".join(lines))
    print(OUT.relative_to(ROOT))


if __name__ == "__main__": main()
