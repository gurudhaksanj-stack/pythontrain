import pandas as pd
import numpy as np

# 1. Load CSV file
df = pd.read_csv("employees.csv")

print("EMPLOYEE DATA")
print(df)

# 2. head(), info(), describe()

print("\nHEAD")
print(df.head())

print("\nINFO")
df.info()

print("\nDESCRIBE")
print(df.describe())

# 3. Select columns

print("\nNAME COLUMN")
print(df["name"])

print("\nNAME AND SALARY")
print(df[["name", "salary"]])

# 4. Using loc

print("\nCHENNAI EMPLOYEES")
chennai = df.loc[df["city"] == "Chennai", "name"]
print(", ".join(chennai))

# 5. Using iloc

print("\nILOC SLICE")
print(df.iloc[0:3, 1:3])

# 6. Salary above 50000

print("\nSALARY ABOVE 50000")
high_salary = df[df["salary"] > 50000]
print(high_salary)
print("Salary above 50000:", len(high_salary), "employees")

# 7. IT department employees

print("\nIT DEPARTMENT")
it_employees = df[df["department"] == "IT"]
print(", ".join(it_employees["name"]))
print("IT department:", len(it_employees), "employees")

# 8. Bonus - NumPy reshape

print("\nBONUS CHALLENGE")

arr = np.arange(1, 25).reshape(2, 3, 4)

print("Array:")
print(arr)

print("Overall shape:", arr.shape)

for i, slice_data in enumerate(arr):
    print(f"Slice {i + 1} shape:", slice_data.shape)

    