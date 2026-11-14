# 1. Variables and Data Types

a = 100
b = 99.50
c = "Guru"
d = [10, 20, 30]
e = {"name": "Guru", "age": 21}
f = {10, 20, 30}
g = (10, 20, 30)

print(a, type(a))
print(b, type(b))
print(c, type(c))
print(d, type(d))
print(e, type(e))
print(f, type(f))
print(g, type(g))


# 2. Sales Report

sales = [12000, 0, 8500, 15000, -200, 9800]

total = 0

for amount in sales:

    if amount <= 0:
        print("Skipped invalid value", amount)
        continue

    total += amount

    print("Added", amount, "total", total)

    if total > 30000:
        break


# 3. Target Checking

if total > 30000:
    print("Target reached. Stopped at", total)

elif total == 30000:
    print("Target reached. Total is", total)

else:
    print("Target not reached. Total is", total)