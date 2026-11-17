# neuro5p — LLM Context Brief

`neuro5p` is a Python research library (Apache 2.0, DOI: 10.5281/zenodo.21873506)
implementing **Neuro-5P**: a continuous, multi-agent governance middleware for
deployed Large Language Models.

## Core idea

Instead of periodic compliance audits, `neuro5p` attaches five parallel autonomous
agents — one per governance dimension — to every LLM output. Each agent runs an
internal **MAPE-K loop** (Monitor → Analyse → Plan → Execute → Knowledge), casts a
structured vote, and the votes resolve to a single attestation:
`pass / flag / block / escalate`.

## The five agents (5P)

| Agent | Dimension | Detects |
|---|---|---|
| Performance | P1 | Calibration drift, confidence errors |
| Parity | P2 | Group-level statistical disparities |
| Proxies | P3 | Correlated proxy features for protected attributes |
| Process | P4 | Audit-trail gaps, obligation discharge |
| Participation | P5 | Missing stakeholder representation |

## Key mechanism

When two agents disagree, the conflict is **classified** (not silenced):
- **Type A** — Performance vs. Parity
- **Type B** — Proxies vs. Participation
- **Type C** — Process vs. Parity

This conflict taxonomy is the novel contribution. Each turn produces a
`TurnState` JSON audit record including all votes, MAPE-K traces, a consensus
verdict, and a metacognitive self-report of known unknowns.

## Grounding

Architecture inspired by Hawkins' Thousand Brains / cortical column theory.
Formal loop from Kephart & Chess (2003) MAPE-K. Governance meta-model from
Checkland's SSM + Beer's VSM. v0.1.0 ships the Proxies agent validated on
the Bias-in-Bios fairness benchmark.

**Thesis:** governance is a trust property, not merely a compliance property.
