"""TF-IDF + logistic-regression baseline used as the comparison point for
the Proxies agent benchmark. A classifier trained to predict the withheld
protected attribute directly from text is the natural monolithic baseline:
if it can recover gender from lexical features alone, those features are
proxies, but the model gives no per-feature interpretable signal and no
governance vote/evidence trail — only a scalar probability.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogRegBaseline:
    def __init__(self, max_features: int = 20_000):
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[a-zA-Z]{3,}", lowercase=True, max_features=max_features, min_df=5
        )
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, texts: list[str], gender_labels: list[int]) -> "LogRegBaseline":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, gender_labels)
        return self

    def deviation_scores(self, texts: list[str]):
        """abs(p - 0.5) * 2, rescaled to [0, 1]: confidence that gender is
        lexically recoverable from the text, used as this baseline's
        equivalent of the Proxies agent's deviation_score."""
        proba = self.model.predict_proba(self.vectorizer.transform(texts))[:, 1]
        return abs(proba - 0.5) * 2, proba
