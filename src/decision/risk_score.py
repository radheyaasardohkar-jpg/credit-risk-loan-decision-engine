import numpy as np


def calculate_risk_score(pd_probability):
    """Convert PD into a project-defined score between 300 and 850."""
    probability = np.asarray(pd_probability, dtype=float)

    if not np.all(np.isfinite(probability)):
        raise ValueError("PD probabilities must be finite.")

    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("PD probabilities must be between 0 and 1.")

    score = 850 - 550 * probability
    score = np.clip(score, 300, 850)

    if score.ndim == 0:
        return int(round(float(score)))

    return np.rint(score).astype(int)