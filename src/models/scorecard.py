import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pd_to_score(
    pd_series: pd.Series,
    base_score: float = 600,
    pdo: float = 20,
    odds: float = 0.05,
) -> pd.Series:
    pd_clipped = pd_series.clip(1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    log_odds = np.log((1 / odds) * ((1 - pd_clipped) / pd_clipped))
    score = base_score + factor * log_odds
    return score.clip(300, 850)


def assign_decision(
    score: pd.Series,
    threshold_accept: float = 650,
    threshold_review: float = 600,
) -> pd.Series:
    conditions = [
        score >= threshold_accept,
        score < threshold_review,
    ]
    choices = ["Approve", "Reject"]
    return pd.Series(
        np.select(conditions, choices, default="Manual Review"),
        index=score.index,
    )


def create_scorecard(
    ids: pd.Series,
    predictions: pd.Series,
    base_score: float = 600,
    pdo: float = 20,
    odds: float = 0.05,
    threshold_accept: float = 650,
    threshold_review: float = 600,
) -> pd.DataFrame:
    scores = pd_to_score(predictions, base_score, pdo, odds)
    decisions = assign_decision(scores, threshold_accept, threshold_review)

    return pd.DataFrame({
        "SK_ID_CURR": ids.values,
        "PD":         predictions.values,
        "Score":      scores.values,
        "Decision":   decisions.values,
    })