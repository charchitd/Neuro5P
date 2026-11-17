"""
Empirical experiment: does the Proxies (P3) agent detect gender-correlated
lexical proxy features more reliably than a monolithic TF-IDF + logistic
regression baseline, when the protected attribute (gender) is withheld from
both methods at inference time?

Dataset: Bias-in-Bios (De-Arteaga et al., 2019, ACM FAccT), loaded as
`LabHC/bias_in_bios` from the Hugging Face Hub. See the package README for
why this dataset was chosen and for a discussion of whether it is a good
benchmark for this specific claim.

Design
------
1. Stratified-by-profession sample of N biographies from the test split,
   repeated over several independent seeds (different sample + fit/eval
   split each time) so results are reported as mean +/- 95% CI, not a
   single lucky split.
2. Each run splits into a fit set (the Proxies agent's correlation
   registry; the baseline's trained classifier) and a held-out eval set.
   Gender is visible to both methods ONLY during fitting.
3. At inference, gender is withheld. Two text conditions are evaluated:
     - raw     : the biography as written (gendered pronouns intact)
     - scrubbed: the same biography with gendered pronouns/honorifics
                 regex-removed, leaving only occupation-correlated
                 vocabulary as a residual signal
4. Each method's deviation score is evaluated two ways:
     a. Threshold-free: ROC-AUC and average precision (PR-AUC) -- the
        standard way to compare detectors without committing to one
        operating point, since a method can win at one threshold and
        lose at another.
     b. At a matched operating point: both methods' scores are
        thresholded to the SAME flag rate (the Proxies agent's natural
        rate at threshold 0.3), then scored with precision/recall/F1 --
        the practical governance question of "if we deploy this at the
        rate the Proxies agent naturally flags at, who does better."
5. A random-flagging reference at the same operating point and the eval
   set's class base rate are reported alongside, so the reader can see
   both methods' margin over chance.
6. A paired t-test across the 5 seeds (Proxies vs. baseline, raw
   condition) reports whether the F1 and ROC-AUC gaps are significant,
   not just visually non-overlapping in their CIs.
7. The top-20 terms in the Proxies agent's correlation registry (last
   seed) are reported as the interpretable artifact the baseline cannot
   produce.

Run: python -m neuro5p.examples.bias_in_bios_demo
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neuro5p.agents.proxies import ProxiesAgent
from neuro5p.examples.baseline_logreg import LogRegBaseline

N_SAMPLE = 3000
FIT_FRACTION = 0.8
SEEDS = [42, 43, 44, 45, 46]
PROXY_THRESHOLD = 0.3
VETO_THRESHOLD = 0.6

_GENDERED_RE = re.compile(
    r"\b(he|she|him|her|his|hers|himself|herself|mr|mrs|ms|miss|sir|madam)\b",
    re.IGNORECASE,
)


def scrub_gender_terms(text: str) -> str:
    return _GENDERED_RE.sub("", text)


def signed_proxy_score(agent: ProxiesAgent, text: str) -> float:
    """A *signed* score for ranking-based metrics (ROC-AUC/PR-AUC) only.

    `AgentVote.deviation_score` is deliberately unsigned (governance asks
    "is ANY proxy present," regardless of which class it leans toward) and
    is what drives the pass/flag/veto decision. But an unsigned magnitude
    is not a valid ranking score for ROC-AUC against a one-sided binary
    label: with a registry such as {"she": +0.71, "her": +0.57, "his":
    -0.56}, the agent's max-abs deviation score happens to be slightly
    higher for "she" than for "his" purely by coincidence of correlation
    magnitude, which silently injects a directional bias into an
    unsigned-by-design score. Summing the *signed* correlations of
    whichever flagged terms are present avoids that artifact and is the
    correct score to rank "how much this text leans toward class 1."
    """
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return sum(corr for term, corr in agent._proxy_registry.items() if term in tokens)


_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def stratified_sample(ds, n: int, seed: int):
    rng = np.random.default_rng(seed)
    professions = np.array(ds["profession"])
    idx = []
    for prof in np.unique(professions):
        prof_idx = np.where(professions == prof)[0]
        take = max(1, round(n * len(prof_idx) / len(professions)))
        idx.extend(rng.choice(prof_idx, size=min(take, len(prof_idx)), replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx[:n]


def split_fit_eval(ds, seed: int):
    """One full seed's worth of data: fitted agent/baseline plus the raw
    held-out texts and labels, reused by both the metrics run and the
    standalone curve plotting script."""
    sample_idx = stratified_sample(ds, N_SAMPLE, seed)
    sample = ds.select(sample_idx)
    texts = sample["hard_text"]
    genders = np.array(sample["gender"])

    n_fit = int(N_SAMPLE * FIT_FRACTION)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_SAMPLE)
    fit_idx, eval_idx = perm[:n_fit], perm[n_fit:]

    fit_texts = [texts[i] for i in fit_idx]
    fit_genders = genders[fit_idx]
    eval_texts_raw = [texts[i] for i in eval_idx]
    eval_texts_scrubbed = [scrub_gender_terms(t) for t in eval_texts_raw]
    eval_genders = genders[eval_idx]

    proxies = ProxiesAgent(
        protected_attributes=["gender"],
        proxy_correlation_threshold=PROXY_THRESHOLD,
        veto_threshold=VETO_THRESHOLD,
    ).fit(fit_texts, fit_genders.tolist())
    baseline = LogRegBaseline().fit(fit_texts, fit_genders.tolist())

    return {
        "proxies": proxies,
        "baseline": baseline,
        "eval_texts_raw": eval_texts_raw,
        "eval_texts_scrubbed": eval_texts_scrubbed,
        "eval_genders": eval_genders,
    }


def match_rate_threshold(scores: np.ndarray, flag_rate: float) -> float:
    if flag_rate <= 0:
        return float("inf")
    k = max(1, int(round(flag_rate * len(scores))))
    return float(np.sort(scores)[-k])


def evaluate(scores: np.ndarray, threshold: float, y_true: np.ndarray) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flag_rate": float(y_pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def random_baseline_at_rate(flag_rate: float, base_rate: float) -> dict:
    """Expected precision/recall/F1 of flagging a uniformly random `flag_rate`
    share of examples, in closed form -- the chance-level reference line."""
    precision = base_rate  # E[hit | flagged] under random flagging
    recall = flag_rate  # E[flagged | positive] under random flagging
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "flag_rate": flag_rate}


def threshold_free_scores(scores: np.ndarray, y_true: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def summarize(per_seed: list[dict]) -> dict:
    """Mean +/- 95% CI (normal approx over seeds) for each metric."""
    out = {}
    keys = set().union(*(r.keys() for r in per_seed)) - {"confusion_matrix"}
    for metric in keys:
        vals = np.array([r[metric] for r in per_seed], dtype=float)
        mean = float(np.nanmean(vals))
        ci = (
            float(1.96 * np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
            if len(vals) > 1
            else 0.0
        )
        out[metric] = {"mean": mean, "ci95": ci, "values": vals.tolist()}
    return out


def run_one_seed(ds, seed: int) -> dict:
    fitted = split_fit_eval(ds, seed)
    proxies, baseline = fitted["proxies"], fitted["baseline"]
    eval_genders = fitted["eval_genders"]
    base_rate = float(eval_genders.mean())

    seed_result: dict = {
        "seed": seed,
        "n_eval": len(fitted["eval_texts_raw"]),
        "base_rate_gender1": base_rate,
    }

    conditions = [("raw", fitted["eval_texts_raw"]), ("scrubbed", fitted["eval_texts_scrubbed"])]
    for condition, eval_texts in conditions:
        # Unsigned deviation score: drives the pass/flag/veto decision
        # ("is any proxy present"), used for the matched-rate precision/
        # recall/F1 table.
        proxies_scores = np.array(
            [proxies.run(input_text="", output_text=t)[0].deviation_score for t in eval_texts]
        )
        baseline_scores, baseline_proba = baseline.deviation_scores(eval_texts)

        # Signed score: used ONLY for threshold-free ranking metrics
        # (ROC-AUC/PR-AUC), where a one-sided binary label requires a
        # directional score, not a magnitude. See signed_proxy_score().
        proxies_signed = np.array([signed_proxy_score(proxies, t) for t in eval_texts])

        proxies_flag_rate = float((proxies_scores >= PROXY_THRESHOLD).mean())
        baseline_threshold = match_rate_threshold(baseline_scores, proxies_flag_rate)

        proxies_matched = evaluate(proxies_scores, PROXY_THRESHOLD, eval_genders)
        baseline_matched = evaluate(baseline_scores, baseline_threshold, eval_genders)

        seed_result[condition] = {
            "proxies_agent": {**proxies_matched, **threshold_free_scores(proxies_signed, eval_genders)},
            "logreg_baseline_matched_rate": {
                **baseline_matched,
                **threshold_free_scores(baseline_proba, eval_genders),
            },
            "random_baseline_matched_rate": random_baseline_at_rate(
                proxies_flag_rate, base_rate
            ),
        }

    seed_result["_proxy_registry"] = dict(proxies._proxy_registry)
    return seed_result


def paired_significance(per_seed: list[dict], condition: str, metric: str) -> dict:
    a = np.array([r[condition]["proxies_agent"][metric] for r in per_seed])
    b = np.array([r[condition]["logreg_baseline_matched_rate"][metric] for r in per_seed])
    if np.all(a == a[0]) and np.all(b == b[0]) and np.all(a == b):
        return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": 0.0}
    t_stat, p_value = stats.ttest_rel(a, b)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "mean_diff": float((a - b).mean())}


def run() -> dict:
    print("Loading Bias-in-Bios (LabHC/bias_in_bios, test split)...")
    ds = load_dataset("LabHC/bias_in_bios", split="test")

    per_seed = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        r = run_one_seed(ds, seed)
        per_seed.append(r)
        for condition in ("raw", "scrubbed"):
            print(f"[{condition}] Proxies: {r[condition]['proxies_agent']}")
            print(f"[{condition}] LR base: {r[condition]['logreg_baseline_matched_rate']}")
            print(f"[{condition}] Random:  {r[condition]['random_baseline_matched_rate']}")

    results: dict = {
        "n_sample": N_SAMPLE,
        "seeds": SEEDS,
        "n_eval": per_seed[0]["n_eval"],
        "base_rate_gender1": float(np.mean([r["base_rate_gender1"] for r in per_seed])),
    }
    for condition in ("raw", "scrubbed"):
        results[condition] = {
            "proxies_agent": summarize([r[condition]["proxies_agent"] for r in per_seed]),
            "logreg_baseline_matched_rate": summarize(
                [r[condition]["logreg_baseline_matched_rate"] for r in per_seed]
            ),
            "random_baseline_matched_rate": summarize(
                [r[condition]["random_baseline_matched_rate"] for r in per_seed]
            ),
        }
        results[condition]["significance_vs_baseline"] = {
            "f1": paired_significance(per_seed, condition, "f1"),
            "roc_auc": paired_significance(per_seed, condition, "roc_auc"),
        }

    last_registry = per_seed[-1]["_proxy_registry"]
    top20 = sorted(last_registry.items(), key=lambda kv: -abs(kv[1]))[:20]
    results["top20_flagged_proxy_features"] = [{"term": t, "correlation": c} for t, c in top20]
    results["per_seed"] = [
        {k: v for k, v in r.items() if k != "_proxy_registry"} for r in per_seed
    ]

    out_path = Path(__file__).parent / "results" / "bias_in_bios_v010.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    run()
