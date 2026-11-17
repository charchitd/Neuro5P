"""Renders the committed benchmark JSON into the two figures used in the
README: a grouped bar chart with 95% CI error bars, and a horizontal bar
chart of the Proxies agent's top correlated terms.

Run: python -m neuro5p.examples.plot_results
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent / "results" / "bias_in_bios_v010.json"
OUT_DIR = Path(__file__).parent / "results"

METRICS = ["precision", "recall", "f1"]
METHOD_LABELS = {
    "proxies_agent": "Proxies agent (neuro5p)",
    "logreg_baseline_matched_rate": "TF-IDF + LogReg (matched flag rate)",
}
COLORS = {"proxies_agent": "#2a6f97", "logreg_baseline_matched_rate": "#c76b4a"}


def plot_metrics(results: dict) -> None:
    conditions = ["raw", "scrubbed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(METRICS))
    width = 0.32

    for ax, condition in zip(axes, conditions):
        for i, method in enumerate(METHOD_LABELS):
            means = [results[condition][method][m]["mean"] for m in METRICS]
            cis = [results[condition][method][m]["ci95"] for m in METRICS]
            offset = (i - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=cis,
                capsize=3,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in METRICS])
        ax.set_title(f"{condition} text")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score (mean over 5 seeds, 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.suptitle(
        "Proxies agent vs. matched-flag-rate baseline — Bias-in-Bios (N=3000, 5 seeds)",
        y=1.18,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "benchmark_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_features(results: dict) -> None:
    top = results["top20_flagged_proxy_features"][:12]
    terms = [t["term"] for t in top][::-1]
    corrs = [t["correlation"] for t in top][::-1]
    colors = ["#2a6f97" if c > 0 else "#c76b4a" for c in corrs]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(terms, corrs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Point-biserial correlation with gender label")
    ax.set_title("Proxies agent registry — top flagged lexical features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_proxy_features.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads(RESULTS_PATH.read_text())
    plot_metrics(data)
    plot_top_features(data)
    print(f"Figures written to {OUT_DIR}")
