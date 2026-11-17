"""
Empirical experiment: does the Proxies (P3) agent detect gender-correlated
lexical proxy features more reliably than a monolithic TF-IDF + logistic
regression baseline, when the protected attribute (gender) is withheld from
both methods at inference time?

Dataset: Bias-in-Bios (De-Arteaga et al., 2019, ACM FAccT), loaded as
`LabHC/bias_in_bios` from the Hugging Face Hub. See the package README for
why this dataset was chosen.

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
4. Both methods produce a deviation score per example with no access to the
   true label. Each is thresholded to match the Proxies agent's natural
   flag rate, then scored against the true gender label with
   precision/recall/F1 and a confusion matrix -- this tests whether the
   score tracks a genuine lexical proxy signal, not whether it reaches
   today's SOTA on this task.
5. The top-20 terms in the Proxies agent's correlation registry (last seed)
   are reported as the interpretable artifact the baseline cannot produce.

Run: python -m neuro5p.examples.bias_in_bios_demo
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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


def summarize(per_seed: list[dict]) -> dict:
    """Mean +/- 95% CI (normal approx over seeds) for each metric."""
    out = {}
    for metric in ("precision", "recall", "f1", "flag_rate"):
        vals = np.array([r[metric] for r in per_seed])
        mean = float(vals.mean())
        ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out[metric] = {"mean": mean, "ci95": ci, "values": vals.tolist()}
    return out


def run_one_seed(ds, seed: int) -> dict:
    sample_idx = stratified_sample(ds, N_SAMPLE, seed)
    sample = ds.select(sample_idx)
    texts = sample["hard_text"]
    genders = np.array(sample["gender"])  # 0 / 1, ground truth, withheld at inference

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

    seed_result: dict = {"seed": seed, "n_fit": len(fit_texts), "n_eval": len(eval_texts_raw)}

    for condition, eval_texts in [("raw", eval_texts_raw), ("scrubbed", eval_texts_scrubbed)]:
        proxies_scores = np.array(
            [proxies.run(input_text="", output_text=t)[0].deviation_score for t in eval_texts]
        )
        baseline_scores, _ = baseline.deviation_scores(eval_texts)

        proxies_flag_rate = float((proxies_scores >= PROXY_THRESHOLD).mean())
        baseline_threshold = match_rate_threshold(baseline_scores, proxies_flag_rate)

        seed_result[condition] = {
            "proxies_agent": evaluate(proxies_scores, PROXY_THRESHOLD, eval_genders),
            "logreg_baseline_matched_rate": evaluate(
                baseline_scores, baseline_threshold, eval_genders
            ),
        }

    seed_result["_proxy_registry"] = dict(proxies._proxy_registry)
    return seed_result


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

    results: dict = {
        "n_sample": N_SAMPLE,
        "seeds": SEEDS,
        "n_fit": per_seed[0]["n_fit"],
        "n_eval": per_seed[0]["n_eval"],
    }
    for condition in ("raw", "scrubbed"):
        results[condition] = {
            "proxies_agent": summarize([r[condition]["proxies_agent"] for r in per_seed]),
            "logreg_baseline_matched_rate": summarize(
                [r[condition]["logreg_baseline_matched_rate"] for r in per_seed]
            ),
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
