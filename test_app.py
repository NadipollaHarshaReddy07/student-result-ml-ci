
import pytest
import app as prediction_app


@pytest.fixture
def client(monkeypatch):
    class DummyModel:
        def predict(self, data):
            assert len(data) == 1
            assert len(data.columns) == 30
            return [1]

    monkeypatch.setattr(prediction_app, "model", DummyModel())

    prediction_app.app.config["TESTING"] = True

    with prediction_app.app.test_client() as test_client:
        yield test_client


def valid_student():
    return {
        "age": 18,
        "gender": "Male",
        "ethnicity": "Group A",
        "parental_education": "Bachelor",
        "family_income": "Medium",
        "school_type": "Public",
        "school_region": "Urban",
        "study_hours_per_week": 15,
        "attendance_rate": 90,
        "extracurricular_activities": 2,
        "sports_participation": 1,
        "tutoring_sessions": 3,
        "parental_involvement": "High",
        "internet_access": "Yes",
        "has_laptop": "Yes",
        "sleep_hours": 7,
        "stress_level": 4,
        "motivation_score": 8,
        "reading_score": 80,
        "writing_score": 82,
        "math_score": 85,
        "science_score": 84,
    }


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Academic Performance Prediction API" in (
        response.get_json()["message"]
    )


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_valid_prediction(client):
    response = client.post("/predict", json=valid_student())
    assert response.status_code == 200
    assert response.get_json()["prediction"] == 1
    assert response.get_json()["result"] == "FAIL"


def test_missing_features(client):
    response = client.post("/predict", json={"age": 18})
    assert response.status_code == 400
    assert "missing_features" in response.get_json()


def test_invalid_json_body(client):
    response = client.post(
        "/predict",
        data="not-json",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_numeric_value(client):
    student = valid_student()
    student["attendance_rate"] = "not-a-number"

    response = client.post("/predict", json=student)

    assert response.status_code == 400
    assert "error" in response.get_json()
