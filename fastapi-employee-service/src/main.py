import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Employee Management API", version="1.0.0")

DATA_FILE = Path(__file__).parent / "employees.json"


class Employee(BaseModel):
    name: str = Field(min_length=1)
    salary: float = Field(gt=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if not value.strip():
            raise ValueError("Name cannot be empty")
        return value.strip()


def read_employees():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_employees(employees):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(employees, file, indent=4)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/employees")
def get_employees():
    return read_employees()


@app.get("/employees/{name}")
def get_employee(name: str):
    employees = read_employees()

    for employee in employees:
        if employee["name"].lower() == name.lower():
            return employee

    raise HTTPException(status_code=404, detail="Employee not found")


@app.post("/employees", status_code=201)
def create_employee(employee: Employee):
    employees = read_employees()

    employees.append(employee.model_dump())

    save_employees(employees)

    return {"message": "Employee created successfully", "employee": employee}
