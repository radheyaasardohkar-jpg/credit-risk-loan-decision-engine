import pytest

from app.utils.prediction import (
    calculate_risk_score,
    categorize_risk,
    make_loan_decision,
)


def test_risk_score_endpoints():
    assert calculate_risk_score(0.0) == 850
    assert calculate_risk_score(1.0) == 300


def test_risk_score_decreases_with_default_probability():
    assert calculate_risk_score(0.10) > calculate_risk_score(0.30)


def test_risk_categories():
    assert categorize_risk(0.02) == "Low Risk"
    assert categorize_risk(0.10) == "Moderate Risk"
    assert categorize_risk(0.20) == "High Risk"
    assert categorize_risk(0.40) == "Very High Risk"


def test_decision_policy():
    assert make_loan_decision(0.05) == "Approve"
    assert make_loan_decision(0.15) == "Manual Review"
    assert make_loan_decision(0.30) == "Reject"


def test_invalid_probability():
    with pytest.raises(ValueError):
        make_loan_decision(1.5)