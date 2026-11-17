"""Renders the committed benchmark JSON (and, for the ROC curves, a fresh
single-seed fit) into the figures used in the README.

Run: python -m neuro5p.examples.plot_results
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_curve

from neuro5p.examples.bias_in_bios_demo import (
    PROXY_THRESHOLD,
    SEEDS,
    signed_proxy_score,
    split_fit_eval,
)

RESULTS_PATH = Path(__file__).parent / "results" / "bias_in_bios_v010.json"
OUT_DIR = Path(__file__).parent / "results"

METHOD_LABELS = {
    "proxies_agent": "Proxies agent (neuro5p)",
    "logreg_baseline_matched_rate": "TF-IDF + LogReg (matched flag rate)",
}
COLORS = {"proxies_agent": "#2a6f97", "logreg_baseline_matched_rate": "#c76b4a"}


def plot_matched_rate_metrics(results: dict) -> None:
    metrics = ["precision", "recall", "f1"]
    conditions = ["raw", "scrubbed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(metrics))
    width = 0.32

    for ax, condition in zip(axes, conditions):
        for i, method in enumerate(METHOD_LABELS):
            means = [results[condition][method][m]["mean"] for m in metrics]
            cis = [results[condition][method][m]["ci95"] for m in metrics]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=cis, capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_title(f"{condition} text")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score at matched flag rate\n(mean over 5 seeds, 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    fig.suptitle("Matched operating point — Bias-in-Bios (N=3000, 5 seeds)", y=1.2, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "benchmark_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_free_metrics(results: dict) -> None:
    metrics = ["roc_auc", "pr_auc"]
    conditions = ["raw", "scrubbed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(metrics))
    width = 0.32

    for ax, condition in zip(axes, conditions):
        for i, method in enumerate(METHOD_LABELS):
            means = [results[condition][method][m]["mean"] for m in metrics]
            cis = [results[condition][method][m]["ci95"] for m in metrics]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=cis, capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUC=0.5)" if condition == "raw" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(["ROC-AUC", "PR-AUC"])
        ax.set_title(f"{condition} text")
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Threshold-free score, signed ranking\n(mean over 5 seeds, 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.suptitle("Threshold-free ranking quality — Bias-in-Bios (N=3000, 5 seeds)", y=1.22, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "benchmark_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(seed: int = SEEDS[0]) -> None:
    ds = load_dataset("LabHC/bias_in_bios", split="test")
    fitted = split_fit_eval(ds, seed)
    proxies, baseline = fitted["proxies"], fitted["baseline"]
    eval_genders = fitted["eval_genders"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, condition, eval_texts in zip(
        axes, ["raw", "scrubbed"], [fitted["eval_texts_raw"], fitted["eval_texts_scrubbed"]]
    ):
        proxies_signed = np.array([signed_proxy_score(proxies, t) for t in eval_texts])
        _, baseline_proba = baseline.deviation_scores(eval_texts)

        for scores, method in [(proxies_signed, "proxies_agent"), (baseline_proba, "logreg_baseline_matched_rate")]:
            fpr, tpr, _ = roc_curve(eval_genders, scores)
            ax.plot(fpr, tpr, color=COLORS[method], label=METHOD_LABELS[method], linewidth=2)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance" if condition == "raw" else None)
        ax.set_xlabel("False positive rate")
        ax.set_title(f"{condition} text (seed {seed})")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("True positive rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.suptitle("ROC curves — single representative seed", y=1.22, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=160, bbox_inches="tight")
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
    plot_matched_rate_metrics(data)
    plot_threshold_free_metrics(data)
    plot_roc_curves()
    plot_top_features(data)
    print(f"Figures written to {OUT_DIR}")
