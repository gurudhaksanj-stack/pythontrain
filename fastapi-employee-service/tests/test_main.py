import pytest
from fastapi.testclient import TestClient

from src import main

client = TestClient(main.app)


@pytest.fixture
def employee_data(tmp_path, monkeypatch):
    file = tmp_path / "employees.json"
    file.write_text('[{"name": "Ravi", "salary": 25000}]', encoding="utf-8")
    monkeypatch.setattr(main, "DATA_FILE", file)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_employees(employee_data):
    response = client.get("/employees")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_employee_success(employee_data):
    response = client.get("/employees/Ravi")
    assert response.status_code == 200
    assert response.json()["name"] == "Ravi"


def test_employee_not_found(employee_data):
    response = client.get("/employees/Arun")
    assert response.status_code == 404


def test_invalid_salary(employee_data):
    response = client.post("/employees", json={"name": "Arun", "salary": -5})
    assert response.status_code == 422


def test_empty_name(employee_data):
    response = client.post("/employees", json={"name": "", "salary": 25000})
    assert response.status_code == 422


def test_create_employee(employee_data):
    response = client.post("/employees", json={"name": "Arun", "salary": 40000})
    assert response.status_code == 201
