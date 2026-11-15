import pandas as pd

# 1. Load orders and customers
orders = pd.read_csv("orders.csv")
customers = pd.read_csv("customers.csv")

# 2. Clean orders data

# Remove duplicate rows
orders = orders.drop_duplicates()

# Fill missing amount with median
median_value = orders["amount"].median()
orders["amount"] = orders["amount"].fillna(median_value)

# Convert order_id to string
orders["order_id"] = orders["order_id"].astype(str)

print("Cleaned Orders:")
print(orders)

# 3. INNER JOIN

inner_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="inner"
)

print("\nINNER JOIN")
print(inner_result)
print("Inner join rows:", len(inner_result))

# 4. LEFT JOIN

left_result = pd.merge(
    customers,
    orders,
    on="customer_id",
    how="left",
    indicator=True
)

print("\nLEFT JOIN")
print(left_result)

# Find customers without orders
no_orders = left_result[left_result["_merge"] == "left_only"]

print("\nCustomers with no orders:")
print(no_orders["name"].to_list())

# 5. OUTER JOIN

outer_result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="outer",
    indicator=True
)

print("\nOUTER JOIN")
print(outer_result)

# Find orders without customer records
unmatched_orders = outer_result[
    outer_result["_merge"] == "left_only"
]

print("\nOrder without customer:")
print(unmatched_orders[["customer_id", "order_id"]])

# BONUS - Concatenate monthly orders

jan = pd.read_csv("orders_jan.csv")
feb = pd.read_csv("orders_feb.csv")

monthly_orders = pd.concat(
    [jan, feb],
    ignore_index=True
)

print("\nCombined Monthly Orders:")
print(monthly_orders)

# GroupBy
monthly_summary = monthly_orders.groupby("customer_id").agg(
    total=("amount", "sum"),
    orders=("order_id", "count")
)

print("\nMonthly Sales Summary:")
print(monthly_summary)

