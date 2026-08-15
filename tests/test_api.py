"""FastAPI service tests. Uses the module-level app instance (trained once
at import time, same pattern as the other two repos in this portfolio)."""

from fastapi.testclient import TestClient

from creditrisk.api.main import app

client = TestClient(app)

_LOW_RISK_APPLICATION = {
    "application_id": "api-test-low-risk",
    "protected_group": "A",
    "age": 45,
    "income_monthly": 4000,
    "employment_years": 15,
    "existing_debt_ratio": 0.10,
    "credit_history_score": 780,
    "loan_amount_requested": 8000,
}

_HIGH_RISK_APPLICATION = {
    "application_id": "api-test-high-risk",
    "protected_group": "B",
    "age": 25,
    "income_monthly": 1100,
    "employment_years": 0.5,
    "existing_debt_ratio": 0.85,
    "credit_history_score": 380,
    "loan_amount_requested": 40000,
}


class TestHealthz:
    def test_reports_a_model_in_production(self):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["production_model_stage"] == "prod"
        assert body["production_model_version"]


class TestScoreEndpoint:
    def test_low_risk_application_is_approved_with_reason_codes(self):
        response = client.post("/score", json=_LOW_RISK_APPLICATION)
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "approve"
        assert body["reason_codes"]

    def test_high_risk_application_is_denied(self):
        response = client.post("/score", json=_HIGH_RISK_APPLICATION)
        body = response.json()
        assert body["decision"] == "deny"

    def test_response_includes_a_decision_id_for_later_override(self):
        response = client.post("/score", json=_LOW_RISK_APPLICATION)
        assert response.json()["decision_id"]

    def test_missing_required_field_is_a_validation_error(self):
        incomplete = {k: v for k, v in _LOW_RISK_APPLICATION.items() if k != "income_monthly"}
        response = client.post("/score", json=incomplete)
        assert response.status_code == 422


class TestOverrideEndpoint:
    def test_override_with_valid_reason_succeeds(self):
        scored = client.post("/score", json=_HIGH_RISK_APPLICATION).json()
        response = client.post(
            f"/override/{scored['decision_id']}",
            json={
                "overridden_decision": "approve",
                "reason": "Garante verificato, reddito familiare sopra soglia.",
                "overridden_by": "analista.credito@banca.example",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["original_decision"] == "deny"
        assert body["overridden_decision"] == "approve"

    def test_override_without_reason_is_rejected(self):
        scored = client.post("/score", json=_HIGH_RISK_APPLICATION).json()
        response = client.post(
            f"/override/{scored['decision_id']}",
            json={"overridden_decision": "approve", "reason": "  ", "overridden_by": "reviewer"},
        )
        assert response.status_code == 400
        assert "reason" in response.json()["detail"].lower()

    def test_override_of_unknown_decision_id_fails(self):
        response = client.post(
            "/override/does-not-exist",
            json={"overridden_decision": "approve", "reason": "valid", "overridden_by": "reviewer"},
        )
        assert response.status_code == 400


class TestFairnessEndpoint:
    def test_returns_group_metrics_and_gap_summaries(self):
        response = client.get("/fairness")
        assert response.status_code == 200
        body = response.json()
        assert len(body["groups"]) == 2
        assert "demographic_parity_difference" in body


class TestDossierEndpoint:
    def test_returns_markdown_with_required_sections(self):
        response = client.get("/dossier")
        assert response.status_code == 200
        for section in ("## Dataset", "## Performance", "## Fairness", "## Limiti noti"):
            assert section in response.text
