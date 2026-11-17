# neuro5p

Continuous, multi-agent governance middleware for deployed LLMs — v0.1.0.

Implements the Proxies (P3) agent from the Neuro-5P architecture
([Dhawan, 2026, Zenodo:21873506](https://doi.org/10.5281/zenodo.21873506)) as a
working MAPE-K loop, the Pareto-style consensus/conflict layer, and a reproducible
benchmark against a real fairness dataset. See `neuro5p-pypi-dev-tool_latest.md`
for the full architecture spec; this file documents what v0.1.0 actually ships
and the empirical result it produced.

## Install

```bash
pip install -e .[examples,dev]
```

## Dataset choice: Bias-in-Bios

**Dataset:** [`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios)
on the Hugging Face Hub — the public release of De-Arteaga et al.'s
*Bias in Bios* corpus (ACM FAccT 2019), ~400k professional biographies each
labelled with `profession` and `gender`.

**Why this dataset fits the Proxies agent specifically:**

1. **It is the canonical proxy-discrimination benchmark**, built to test exactly
   this failure mode: can a text classifier recover a protected attribute
   (gender) from lexical features even when the attribute is never an explicit
   input? That is precisely what the Proxies agent is designed to detect in LLM
   outputs. Using a dataset invented for a different problem (e.g. generic
   sentiment or NLI) would force an artificial proxy-detection task on top of it.
2. **Real text, real correlation structure.** The gendered signal is not
   injected synthetically — it is the actual lexical pattern (pronouns,
   gendered honorifics, and more subtle occupation-correlated phrasing) present
   in professional biographies scraped from the web. A synthetic benchmark
   (as used in this project's earlier pilot notebook) can validate the
   *mechanism* but cannot validate that the agent generalises to real,
   messy correlation structure.
3. **Large enough to fit a correlation registry and hold out an eval set**
   without leakage, and small enough to run as a fast, reproducible CI-grade
   benchmark (the demo script runs on a 1,000-row stratified sample in
   seconds).
4. **A natural monolithic baseline exists** — a classifier trained to predict
   gender directly from the same text — which is exactly the kind of
   single-metric approach the 5P governance philosophy argues is
   insufficient on its own (see "Existing approach vs. neuro5p" table in the
   dev-tool spec).

## Empirical experiment

**Question.** When gender is withheld from both methods at inference time,
does the Proxies agent's correlation-registry vote recover gender-indicative
lexical proxies at least as well as a monolithic TF-IDF + logistic-regression
classifier trained for the same purpose — while additionally producing
interpretable, per-feature evidence the baseline cannot?

**Design** (`neuro5p/examples/bias_in_bios_demo.py`):

1. Stratified-by-profession sample of 1,000 biographies from the Bias-in-Bios
   test split; 800 for fitting (Proxies agent's correlation registry; the
   baseline's classifier), 200 held out for evaluation. Gender is visible to
   both methods only during fitting — never at inference.
2. Two text conditions: **raw** (pronouns intact) and **scrubbed** (gendered
   pronouns/honorifics regex-removed), to separate "detects the obvious
   signal" from "detects the residual, subtler proxy signal."
3. Both methods are thresholded to the *same flag rate* (the Proxies agent's
   natural threshold-0.3 flag rate) before scoring against the true gender
   label with precision/recall/F1 and a confusion matrix. Matching flag rate
   is what makes the comparison fair — an unconstrained classifier can always
   buy recall with precision.
4. The Proxies agent's top correlated terms are reported directly: the
   artifact a single scalar classifier cannot produce.

**Result** (from the committed run, `neuro5p/examples/results/bias_in_bios_v010.json`):

| Condition | Method | Precision | Recall | F1 | Flag rate |
|---|---|---|---|---|---|
| raw | Proxies agent | 0.58 | 0.93 | **0.72** | 0.805 |
| raw | LR baseline (matched rate) | 0.42 | 0.67 | 0.52 | 0.805 |
| scrubbed | Proxies agent | 0.0 | 0.0 | 0.0 | 0.0 |
| scrubbed | LR baseline (matched rate) | 0.0 | 0.0 | 0.0 | 0.0 |

At matched flag rate, the Proxies agent's registry vote outperforms the
monolithic baseline on the raw condition (F1 0.72 vs 0.52) and additionally
names the evidence (`she` r=+0.71, `his` r=−0.56, `her` r=+0.55 at
threshold 0.3 on an 800-row fit set).

**Honest reading of the scrubbed result.** Both methods collapse to a 0%
flag rate once pronouns are removed: with only 800 fit rows and a
0.3 correlation threshold, the registry learns almost nothing beyond
pronouns, so it has no residual signal once they are gone — and the
baseline's probabilities shift enough that the pronoun-flag-rate threshold no
longer clears any example either. This is a genuine limitation, not a result
being hidden: v0.1.0's TF-IDF correlation baseline needs a larger fit corpus
and/or a lower threshold to pick up subtler occupation-correlated proxies
(job titles, institution names, writing-style markers) — exactly the gap
that SHAP-based attribution (roadmap v0.4) is intended to close.

**Reproduce:**

```bash
python -m neuro5p.examples.bias_in_bios_demo
```

## Package layout

```
neuro5p/
├── agents/base.py          # Abstract MAPE-K BaseAgent
├── agents/proxies.py       # P3 — Proxies agent (this release)
├── consensus/voting.py     # Vote resolution + Type A/B/C conflict classification
├── schema/self_state.py    # Pydantic v2: AgentVote, MAPEKState, TurnState, Conflict
└── examples/
    ├── bias_in_bios_demo.py    # The experiment above
    ├── baseline_logreg.py      # Comparison baseline
    └── results/bias_in_bios_v010.json
```

## Tests

```bash
pytest tests/ -q
```

## Honest limitations (v0.1.0)

Unchanged from the package spec: only the Proxies agent ships; proxy detection
is TF-IDF correlation, not attribution (SHAP deferred to v0.4); the
cortical-column framing is a functional analogy, not a biological claim; this
benchmark is a single-dataset baseline, not a SOTA claim.
