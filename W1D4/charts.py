import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("employees.csv")

print(df)

avg_salary = df.groupby("department")["salary"].mean()

plt.figure(figsize=(8, 5))

plt.bar(avg_salary.index, avg_salary.values)

plt.title("Average Salary per Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.bar_label(plt.gca().containers[0])

plt.savefig("bar_chart.png")

plt.show()
plt.close()

plt.figure(figsize=(8, 5))

plt.hist(df["salary"], bins=5)

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.savefig("histogram.png")

plt.show()
plt.close()


plt.figure(figsize=(8, 5))

for dept in df["department"].unique():
    dept_data = df[df["department"] == dept]

    plt.scatter(
        dept_data["id"],
        dept_data["salary"],
        label=dept
    )

plt.title("Employee ID vs Salary")
plt.xlabel("Employee ID")
plt.ylabel("Salary")

plt.legend()

plt.savefig("scatter_plot.png")

plt.show()
plt.close()

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

sales = [12, 15, 14, 18, 22, 25]

plt.figure(figsize=(8, 5))

plt.plot(months, sales, marker="o")

plt.title("Monthly Sales")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.savefig("line_chart.png")

plt.show()
plt.close()

