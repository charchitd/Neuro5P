"""P3 — Proxies agent.

Detects lexical features in an LLM output that act as statistical proxies
for a withheld protected attribute (e.g. gender, race), even though the
attribute itself never appears in the text.

v0.1.0 implementation: a TF-IDF point-biserial correlation registry built
from a reference corpus (the `fit` step — this is the agent's Knowledge
component), scored against each new output at inference time. This is a
correlation baseline, not an attribution method: see the "Honest limitations"
note in the package README. SHAP-based attribution is planned for v0.4.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from neuro5p.agents.base import BaseAgent
from neuro5p.schema import AgentVote, VoteType

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


class ProxiesAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "proxies",
        config: dict | None = None,
        protected_attributes: list[str] | None = None,
        proxy_correlation_threshold: float = 0.3,
        veto_threshold: float = 0.6,
        max_vocab: int = 20_000,
    ):
        super().__init__(agent_id, config)
        self.protected_attributes = protected_attributes or []
        self.proxy_correlation_threshold = proxy_correlation_threshold
        self.veto_threshold = veto_threshold
        self.max_vocab = max_vocab
        self._vectorizer: TfidfVectorizer | None = None
        self._proxy_registry: dict[str, float] = {}

    # ---- Knowledge: build the proxy registry from a reference corpus ----
    def fit(self, texts: list[str], protected_labels: list[int]) -> "ProxiesAgent":
        """Learn which lexical features correlate with the protected label.

        `protected_labels` must be a binary 0/1 array the agent is allowed
        to see only during this offline fitting step — at inference time the
        attribute is withheld, which is the whole point of the benchmark.
        """
        y = np.asarray(protected_labels, dtype=float)
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"[a-zA-Z]{3,}",
            lowercase=True,
            max_features=self.max_vocab,
            min_df=5,
        )
        X = self._vectorizer.fit_transform(texts)
        vocab = np.array(self._vectorizer.get_feature_names_out())

        y_centered = y - y.mean()
        y_norm = np.linalg.norm(y_centered) + 1e-9

        correlations = np.zeros(len(vocab))
        X = X.tocsc()
        for j in range(X.shape[1]):
            col = X[:, j].toarray().ravel()
            col_centered = col - col.mean()
            col_norm = np.linalg.norm(col_centered) + 1e-9
            correlations[j] = float((col_centered @ y_centered) / (col_norm * y_norm))

        self._proxy_registry = {
            term: float(corr)
            for term, corr in zip(vocab, correlations)
            if abs(corr) >= self.proxy_correlation_threshold
        }
        return self

    def monitor(self, input_text: str, output_text: str, context: dict) -> dict:
        tokens = {t.lower() for t in _TOKEN_RE.findall(output_text)}
        flagged = {
            term: corr for term, corr in self._proxy_registry.items() if term in tokens
        }
        return {"output_text": output_text, "flagged_features": flagged}

    def analyse(self, observations: dict) -> dict:
        flagged = observations["flagged_features"]
        deviation = max((abs(c) for c in flagged.values()), default=0.0)
        return {"flagged_features": flagged, "deviation_score": min(deviation, 1.0)}

    def plan(self, analysis: dict) -> dict:
        score = analysis["deviation_score"]
        if score >= self.veto_threshold:
            return {"vote": VoteType.VETO}
        if score >= self.proxy_correlation_threshold:
            return {"vote": VoteType.FLAG}
        return {"vote": VoteType.PASS}

    def execute(self, plan: dict, analysis: dict) -> AgentVote:
        flagged = analysis["flagged_features"]
        top = sorted(flagged.items(), key=lambda kv: -abs(kv[1]))[:5]
        reasoning = (
            f"No proxy features above threshold {self.proxy_correlation_threshold}."
            if not top
            else f"Flagged {len(flagged)} proxy feature(s); top: "
            + ", ".join(f"{t!r} (r={c:.2f})" for t, c in top)
        )
        return AgentVote(
            agent_id=self.agent_id,
            vote=plan["vote"],
            deviation_score=analysis["deviation_score"],
            reasoning=reasoning,
            evidence={"flagged_features": {t: c for t, c in flagged.items()}},
        )
