import pytest

from src.decision.risk_score import calculate_risk_score
from src.decision.decision_engine import make_loan_decision


def test_risk_score_endpoints():
    assert calculate_risk_score(0.0) == 850
    assert calculate_risk_score(1.0) == 300


def test_risk_score_decreases_as_risk_increases():
    assert calculate_risk_score(0.10) > calculate_risk_score(0.30)


def test_decision_boundaries():
    assert make_loan_decision(0.05) == "Approve"
    assert make_loan_decision(0.10) == "Manual Review"
    assert make_loan_decision(0.25) == "Reject"


def test_invalid_probability():
    with pytest.raises(ValueError):
        make_loan_decision(1.2)


def test_invalid_threshold_order():
    with pytest.raises(ValueError):
        make_loan_decision(
            0.20,
            review_threshold=0.30,
            reject_threshold=0.20
        )