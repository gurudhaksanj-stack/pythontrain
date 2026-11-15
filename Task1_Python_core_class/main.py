import csv


def calculate_bonus(salary, rate=0.1, *extras, **options):
    bonus = salary * rate + sum(extras)

    if "cap" in options:
        bonus = min(bonus, options["cap"])

    return bonus


class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_details(self):
        return f"{self.name} - {self.salary}"

    @property
    def annual_salary(self):
        return self.salary * 12


def log_call(func):
    def wrapper(*args, **kwargs):
        print("Calling get_details")
        return func(*args, **kwargs)
    return wrapper


class Manager(Employee):
    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size

    @log_call
    def get_details(self):
        return f"{self.name} - {self.salary} - Team Size: {self.team_size}"


print("Bonus:", calculate_bonus(50000))
print("Bonus with extras:", calculate_bonus(50000, 0.1, 1000, 2000))
print("Bonus with cap:", calculate_bonus(50000, 0.1, 1000, cap=4000))


emp = Employee("Arun", 30000)

print(emp.get_details())
print(emp.annual_salary)


manager = Manager("Divya", 75000, 5)

print(manager.get_details())

# Read Employee Data and Print Salary Categories

with open("employees.csv", "r") as file:
    reader = csv.DictReader(file)

    for employee in reader:
        name = employee["name"]
        salary = int(employee["salary"])

        if salary < 50000:
            category = "Low Salary"
        elif salary < 70000:
            category = "Medium Salary"
        else:
            category = "High Salary"

        print(f"{name} - {category}")

        