# app/utils/prediction.py

import numpy as np


def calculate_risk_score(pd_probability):
    """Map probability of default to a project-defined 300–850 score."""
    probability = np.asarray(pd_probability, dtype=float)

    if not np.all(np.isfinite(probability)):
        raise ValueError("Probability must be finite.")

    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probability must be between 0 and 1.")

    score = np.clip(850 - 550 * probability, 300, 850)

    if score.ndim == 0:
        return int(round(float(score)))

    return np.rint(score).astype(int)


def categorize_risk(pd_probability):
    if pd_probability < 0.05:
        return "Low Risk"
    elif pd_probability < 0.15:
        return "Moderate Risk"
    elif pd_probability < 0.30:
        return "High Risk"
    return "Very High Risk"


def make_loan_decision(
    pd_probability,
    review_threshold=0.10,
    reject_threshold=0.25
):
    if not np.isfinite(pd_probability) or not 0 <= pd_probability <= 1:
        raise ValueError("PD must be a finite value between 0 and 1.")

    if not 0 <= review_threshold < reject_threshold <= 1:
        raise ValueError("Invalid decision thresholds.")

    if pd_probability < review_threshold:
        return "Approve"
    elif pd_probability < reject_threshold:
        return "Manual Review"
    return "Reject"


def predict_applicant(
    model,
    applicant_features,
    review_threshold=0.10,
    reject_threshold=0.25
):
    """Return model PD separately from the illustrative policy decision."""
    pd_probability = float(
        model.predict_proba(applicant_features)[:, 1][0]
    )

    return {
        "probability_of_default": pd_probability,
        "risk_score": calculate_risk_score(pd_probability),
        "risk_category": categorize_risk(pd_probability),
        "decision": make_loan_decision(
            pd_probability,
            review_threshold,
            reject_threshold
        )
    }