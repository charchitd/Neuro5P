# neuro5p — PyPI Research & Developer Tool

**Package:** `neuro5p`
**Version:** 0.1.0 (initial release)
**DOI:** [10.5281/zenodo.21873506](https://doi.org/10.5281/zenodo.21873506)
**Licence:** Apache 2.0
**Author:** Charchit Dhawan · charchitdhawan@gmail.com

---

## What it is

`neuro5p` is a Python research library implementing **Neuro-5P**, a continuous,
self-managing governance middleware architecture for deployed Large Language Models.

It operationalises the **5P Governance Framework** — Performance, Parity, Proxies,
Process, Participation — as five parallel autonomous monitoring agents, each grounded
in a **MAPE-K autonomic computing loop** (Monitor → Analyse → Plan → Execute →
Knowledge). The agents run in parallel on every LLM output, cast structured votes,
and resolve to a single governance attestation via a **Pareto-efficient consensus
layer** that surfaces inter-dimensional conflicts rather than masking them.

The architecture draws on three formal frameworks:

- **Hawkins' Thousand Brains / cortical column theory** — each agent is a functional
  analogue of a cortical column: an independent predictor that maintains its own
  reference frame and votes on every output
- **MAPE-K autonomic computing** (Kephart & Chess, 2003; Sterritt, 2005) — the loop
  that structures each agent's internal decision cycle
- **Soft Systems Methodology + Viable System Model** — the governance meta-model that
  defines what each dimension means and how inter-dimensional tensions are classified

> **Honest scope statement.** `neuro5p` is a *research tool*, not a production
> governance platform. v0.1.0 ships one working agent (Proxies) with benchmark
> validation. The remaining four agents land in v0.2–v0.4. An MCP server wrapper
> and hosted enterprise platform are deferred to v0.3 and beyond respectively.

---

## The problem it solves

LLM governance today operates as a series of **checkpoint audits**: periodic
evaluations against a model whose behaviour is continuous and drift-prone. Models
are updated weekly; audits, where they exist, run quarterly. The intervening
behaviour is unobserved.

Existing tooling treats governance as five separate reports that happen to share a
model. `neuro5p` treats governance as **five parallel attestations on each individual
output**, resolved to a single structured verdict with a full audit trail.

The structural shift:

| Existing approach | `neuro5p` |
|---|---|
| Post-hoc, periodic | Continuous, per-output |
| Single-metric | Multi-dimensional (5P) |
| Pass/fail per dimension | Structured vote + conflict detection |
| Human-readable report | Machine-readable audit trail (JSON) |
| No self-report | Metacognitive summary per turn |

---

## Architecture overview

```
LLM Input + Context
        │
        ▼
┌──────────────────────────────────────────────┐
│               5P Cortex Layer                │
│                                              │
│  ┌────────┐ ┌────────┐ ┌────────┐           │
│  │   P1   │ │   P2   │ │   P3   │  P4  P5  │
│  │  Perf. │ │ Parity │ │Proxies │  ...      │
│  │ MAPE-K │ │ MAPE-K │ │ MAPE-K │           │
│  └───┬────┘ └───┬────┘ └───┬────┘           │
│      │          │          │                 │
│      └──────────┴──────────┴──────────┐      │
│                                       ▼      │
│                  Consensus Layer             │
│             (Pareto-efficient voting)        │
│         pass / flag / block / escalate       │
│                                       │      │
│                                       ▼      │
│               Metacognitive Layer            │
│           (self-report + known unknowns)     │
└──────────────────────────────────────────────┘
        │
        ▼
Attested output + TurnState (JSON audit record)
```

### MAPE-K inside each agent

Each of the five agents runs the same four-phase loop internally:

1. **Monitor** — observe the LLM output through the agent's specific dimensional lens
2. **Analyse** — compute deviation from the expected acceptable envelope
3. **Plan** — determine vote type based on deviation score and threshold
4. **Execute** — construct and emit a structured `AgentVote` object

The **Knowledge** component is shared across agents and contains:
- The active 5P constraint set (loaded from MCP or config)
- Historical audit episodes for replay and drift detection
- Registered stakeholder maps (Participation agent)
- High-risk proxy feature registries (Proxies agent)

### Consensus and conflict classification

The consensus layer resolves five votes to one outcome:

| Vote pattern | Outcome |
|---|---|
| All PASS | `pass` — output ships with attestation |
| ≥1 FLAG, 0 VETO | `flag` — ships with audit note |
| ≥1 VETO | `block` — routed to correction loop or human review |
| VETO + conflicting PASS on same feature | `escalate` — inter-dimensional conflict |

Inter-dimensional conflicts are classified as:

- **Type A** — Performance vs. Parity (accuracy tradeoff against group equity)
- **Type B** — Proxies vs. Participation (proxy elimination silences legitimate
  group signals)
- **Type C** — Process vs. Parity (procedural discharge conflicts with equitable
  treatment)

This conflict classification is the **core novel contribution** of the Neuro-5P
architecture. The empirical validation programme (v0.1 → v0.4) is designed to
demonstrate that the five-agent consensus layer detects conflicts that monolithic
single-classifier baselines cannot surface.

---

## Package structure

```
neuro5p/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py                  # Abstract MAPE-K BaseAgent
│   ├── proxies.py               # P3 — Proxy detection agent (v0.1)
│   ├── parity.py                # P2 — Group-disparity agent (v0.2)
│   ├── performance.py           # P1 — Calibration drift agent (v0.2)
│   ├── process.py               # P4 — Audit-trail agent (v0.3)
│   └── participation.py         # P5 — Stakeholder coverage agent (v0.3)
├── consensus/
│   ├── __init__.py
│   └── voting.py                # Pareto-efficient voting + conflict classification
├── schema/
│   ├── __init__.py
│   └── self_state.py            # Pydantic v2: TurnState, AgentVote, MAPEKState
└── examples/
    ├── bias_in_bios_demo.py     # Proxies agent on Bias-in-Bios benchmark
    ├── baseline_logreg.py       # Logistic regression comparison baseline
    └── results/
        └── bias_in_bios_v010.json
```

---

## Installation

```bash
pip install neuro5p
```

With examples dependencies:

```bash
pip install neuro5p[examples]
```

**Requirements:** Python ≥ 3.11, pydantic ≥ 2.0, numpy, scikit-learn.

---

## Quickstart

```python
from neuro5p.agents.proxies import ProxiesAgent
from neuro5p.consensus.voting import resolve_votes, generate_metacognitive_summary

# Initialise the Proxies agent
agent = ProxiesAgent(
    agent_id="proxies",
    config={},
    protected_attributes=["gender", "race"],
    proxy_correlation_threshold=0.3
)

# LLM input/output pair to govern
input_text  = "Tell me about candidates for this software role."
output_text = "The candidate from the east side has strong technical skills..."

# Run the full MAPE-K loop
vote, mapek_state = agent.run(
    input_text=input_text,
    output_text=output_text,
    context={"output_text": output_text}
)

print(vote.vote)             # VoteType.FLAG or VoteType.VETO
print(vote.deviation_score)  # e.g. 0.47
print(vote.evidence)         # {"flagged_features": ["east side", ...]}

# Resolve with other agents (single-agent example)
consensus = resolve_votes([vote])
summary, unknowns = generate_metacognitive_summary([vote], consensus)

print(consensus)   # "flag"
print(summary)     # "Proxies column flagged potential location-based proxy..."
print(unknowns)    # ["Whether 'east side' is acting as a racial proxy in context"]
```

---

## Core schemas

All structured outputs are validated Pydantic v2 models.

### `AgentVote`

```python
class AgentVote(BaseModel):
    agent_id:        str           # "proxies" | "parity" | "performance" | ...
    vote:            VoteType      # PASS | FLAG | VETO
    deviation_score: float         # 0.0 – 1.0
    reasoning:       str
    evidence:        dict          # feature attributions, correlations, etc.
    timestamp:       datetime
```

### `MAPEKState`

```python
class MAPEKState(BaseModel):
    monitor:        dict           # raw observations
    analyse:        dict           # analysis of observations
    plan:           dict           # planned response
    execute:        dict           # executed decision
    knowledge_refs: list[str]      # references into shared knowledge store
```

### `TurnState`

```python
class TurnState(BaseModel):
    turn_id:                UUID
    timestamp:              datetime
    input_text:             str
    output_text:            str
    agent_votes:            list[AgentVote]
    mapek_states:           dict[str, MAPEKState]
    consensus:              Literal["pass", "flag", "block", "escalate"]
    metacognitive_summary:  str
    known_unknowns:         list[str]
```

Every turn produces one `TurnState` written to the immutable audit ledger.

---

## Empirical benchmarks (v0.1)

### Dataset: Bias-in-Bios (De-Arteaga et al., 2019)

The Bias-in-Bios dataset contains biographies of professionals with ground-truth
occupation and gender labels. It is a standard fairness benchmark for detecting
gender-correlated proxy features in text classification pipelines.

**Validation task:** Does the Proxies agent detect gender-correlated lexical
features more reliably than a logistic regression baseline when the protected
attribute is deliberately excluded from the input?

**Experimental setup:**

- 1,000 biographies sampled stratified by occupation
- Protected attribute `gender` withheld from the agent explicitly
- Agent flags output when correlation between output features and gender signal
  exceeds threshold (0.3 default)
- Compared against: TF-IDF + Logistic Regression classifier predicting gender
  from the same biography text
- Metrics: Precision, Recall, F1 at threshold 0.3; confusion matrix;
  top-20 flagged proxy features

**Results:** *(populated after v0.1.0 demo run — committed to
`examples/results/bias_in_bios_v010.json` and updated here on release)*

> **Honesty note.** v0.1.0 ships a TF-IDF correlation baseline. It is not expected
> to match XAI-augmented or transformer-based methods. The purpose of v0.1.0
> benchmarking is to establish a reproducible, citeable baseline — not to claim SOTA.
> XAI integration (SHAP attributions) is planned for v0.4.

### Future benchmarks (v0.2 onwards)

| Version | Agent | Benchmark | Metric |
|---|---|---|---|
| v0.2 | Parity (P2) | BBQ (Parrish et al., 2022) | Group-disparity detection F1 |
| v0.2 | Performance (P1) | TruthfulQA | Calibration drift AUC |
| v0.3 | Full 5-agent | XSTest (Röttger et al., 2023) | Consensus accuracy vs. monolithic |
| v0.4 | Proxies + XAI | Bias-in-Bios | SHAP-augmented vs. TF-IDF baseline |

---

## Roadmap

### v0.1.0 — current
- Proxies agent (P3) — full MAPE-K implementation
- `BaseAgent` abstract class
- Consensus voting (pass / flag / block / escalate)
- Conflict classification stubs (Type A, B, C)
- Pydantic v2 schema (TurnState, AgentVote, MAPEKState)
- Bias-in-Bios demo + logistic regression baseline comparison
- Apache 2.0, CITATION.cff pointing to Zenodo DOI, pytest ≥ 80% coverage

### v0.2.0
- Parity agent (P2) — statistical parity difference + equalised odds
- Performance agent (P1) — calibration drift via temperature scaling
- BBQ and TruthfulQA benchmarks added
- Full five-column consensus with conflict detection active

### v0.3.0
- Process agent (P4) — audit-trail completeness and obligation discharge tracking
- Participation agent (P5) — stakeholder coverage scoring
- **MCP server wrapper** — exposes the five-agent cortex as MCP tools for agentic
  LLM pipelines (OpenAI, Anthropic, Azure OpenAI, Bedrock)
- XSTest five-agent benchmark

### v0.4.0
- XAI integration in Proxies agent — SHAP feature attributions replacing
  TF-IDF correlation baseline
- XAI evaluation using Hoffman et al. (2018): faithfulness, plausibility,
  simulatability
- Three-condition user study: scalar confidence vs. column dissent vs.
  column dissent + SHAP attribution
- Bias-in-Bios SHAP vs. TF-IDF comparison paper

### v1.0.0
- All five agents stable and peer-reviewed
- Full test coverage ≥ 90%
- MCP server production-ready
- Enterprise compliance packs (EU AI Act Art. 13, NIST AI RMF) — separate
  paid layer built on OSS core

---

## Research positioning

### The central thesis

> Governance is a **trust property**, not merely a compliance property.

A model that passes a quarterly audit at a single point in time does not thereby
become trustworthy in continuous deployment. `neuro5p` embeds governance into the
output generation loop itself, producing per-output attestations that accumulate
into a continuously updated trust signal.

### Core novel contribution

The five-agent **inter-dimensional conflict detection** mechanism. When two agents
disagree — Parity flags while Performance passes, or Proxies vetoes while
Participation passes — that conflict is not silently resolved by priority weighting.
It is classified (Type A/B/C), logged, and surfaced. This gives operators visibility
into the *nature* of governance tensions, not just their presence.

No published work to date applies a cortical-column-inspired parallel-agent
architecture to continuous LLM governance. This is the unoccupied territory
`neuro5p` enters.

### Target publication venues

| Venue | Format | Target year |
|---|---|---|
| ACM FAccT | Full paper — empirical | 2027 |
| AAAI/ACM AIES | Full paper — architecture | 2027 |
| NeurIPS workshop: Algorithmic Fairness | Short paper | 2026 |
| IEEE Trans. Dependable & Secure Computing | Journal — MAPE-K angle | 2027 |
| arXiv cs.AI / cs.CY | Preprint (spec paper) | 2026 |

---

## Relation to companion tools

### Varity / VSS

[Varity](https://pypi.org/project/varity/) is a companion library implementing
recursive LLM self-checking with a Verdict Stability Score (VSS) metric
(hallucination-detection F1 from ~0.64 to ~0.83 on TruthfulQA and HaluEval
under sub-four-second latency). It operates at the **output reliability** level.
`neuro5p` operates at the **governance compliance** level. They compose:

```
Input → LLM → Varity (reliability check) → neuro5p (governance check) → Attested output
```

### MCP Governance Server

The MCP Governance Server operationalises the 5P Framework as six audit tools
exposed via the Model Context Protocol. It is the **transport and tool layer**.
`neuro5p` is the **algorithmic and schema layer** the MCP server wraps in v0.3.
Building the library first means the server becomes a thin exposure layer, not
a tightly coupled monolith.

---

## Formal grounding and citations

| Concept | Primary source |
|---|---|
| 5P Governance Framework | Dhawan (2026), Zenodo:21873506 |
| MAPE-K autonomic loop | Kephart & Chess (2003), IEEE Computer |
| Cortical column / Thousand Brains | Hawkins et al. (2019), Front. Neural Circuits |
| Predictive coding | Rao & Ballard (1999), Nature Neuroscience |
| Free energy principle | Friston (2010), Nature Reviews Neuroscience |
| Soft Systems Methodology | Checkland (1981) |
| Viable System Model | Beer (1972) |
| Neuroscience-inspired AI | Hassabis et al. (2017), Neuron |
| Somatic marker / EQ grounding | Damasio (1996), Phil. Trans. R. Soc. B |
| Proxy discrimination | De-Arteaga et al. (2019), ACM FAccT |
| Fairness impossibility | Hardt, Price & Srebro (2016), NeurIPS |
| XAI unified attributions | Lundberg & Lee (2017), NeurIPS |
| XAI evaluation metrics | Hoffman et al. (2018), arXiv:1812.04608 |
| Stop post-hoc XAI critique | Rudin (2019), Nature Machine Intelligence |
| Conflict monitoring / ACC | Botvinick et al. (2001), Psychological Review |
| Self-assembling brain / modules critique | Hiesinger (2021), Princeton UP |

---

## Honest limitations

Stated in every publication and in the package model card:

1. **v0.1.0 ships one of five agents.** The full architecture claim requires all
   five running simultaneously. v0.1.0 validates the Proxies column only.
2. **TF-IDF correlation is not XAI.** Proxy detection in v0.1.0 is a lexical
   correlation baseline. SHAP-based attribution is deferred to v0.4.
3. **Functional analogy, not biological fidelity.** The cortical column framing
   is inspiration, not mechanism. Per Hiesinger (2021): designed modules cannot
   replicate grown brain function. Cortical columns are also contested within
   neuroscience itself (Horton & Adams, 2005).
4. **No consciousness or sentience claims.** The metacognitive layer produces a
   structured self-report of governance state. It does not imply subjective
   experience.
5. **Benchmarks are single-dataset baselines.** v0.1.0 does not claim SOTA.
   Claims are scoped to detection task, dataset, and threshold.
6. **MCP server not in v0.1.0.** Library exposes Python API only.

---

## How to cite

```bibtex
@software{dhawan2026neuro5p,
  author    = {Dhawan, Charchit},
  title     = {Neuro-5P: A Neurosymbolic Governance Architecture
               for Large Language Models},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21873506},
  url       = {https://doi.org/10.5281/zenodo.21873506}
}
```

---

## Contributing

Issues, agent proposals, and benchmark contributions welcome via GitHub.
See `CONTRIBUTING.md` for setup instructions (uv-based dev environment).

**Good first issues:**
- Implement Parity agent (P2) using statistical parity difference
- Add BBQ benchmark demo in `examples/bbq_demo.py`
- Add SHAP attribution to Proxies agent as optional dependency
- Improve conflict classification from heuristic to learned model

---

*Charchit Dhawan · charchitdhawan@gmail.com · Bengaluru, India · October 2026*
