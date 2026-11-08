"""Question router: 5 classifiers (GridSearchCV, StratifiedKFold 5, macro-F1) vs the LLM -> models/router.joblib + router_meta.json. Run: python -m src.train.train_router

Owner: Francesca.
"""
from __future__ import annotations


def train(cfg) -> dict:
    """Train, compare and save the best router; return the scores."""
    raise NotImplementedError("train: not built yet")
