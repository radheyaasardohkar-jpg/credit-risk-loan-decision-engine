def make_loan_decision(
    pd_probability: float,
    review_threshold: float = 0.10,
    reject_threshold: float = 0.25
) -> str:
    """
    Apply a configurable triage policy to a predicted default probability.

    Thresholds are illustrative and must be validated before operational use.
    """
    if not 0 <= pd_probability <= 1:
        raise ValueError("PD probability must be between 0 and 1.")

    if not 0 <= review_threshold < reject_threshold <= 1:
        raise ValueError(
            "Require 0 <= review_threshold < reject_threshold <= 1."
        )

    if pd_probability < review_threshold:
        return "Approve"

    if pd_probability < reject_threshold:
        return "Manual Review"

    return "Reject"