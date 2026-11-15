import pandas as pd

# 1. Load CSV file
df = pd.read_csv("orders.csv")

print("Original Data:")
print(df)

# 2. Find missing values
print("\nMissing Values:")
print(df.isna().sum())

missing_count = df["amount"].isna().sum()
print("Missing amount:", missing_count)

# 3. Find duplicates
print("\nDuplicate Rows:")
print(df.duplicated())

duplicate_count = df.duplicated().sum()
print("Duplicates:", duplicate_count)

# 4. Remove duplicate rows
df = df.drop_duplicates()

# 5. Make amount numeric
df["amount"] = pd.to_numeric(df["amount"])

# 6. Fill missing amount with median
median_value = df["amount"].median()

df["amount"] = df["amount"].fillna(median_value)

print("\nMedian used:", median_value)

# 7. Convert order_id into string
df["order_id"] = df["order_id"].astype(str)

print("\nCleaned Data:")
print(df)

# 8. GroupBy and Aggregation
result = df.groupby("customer_id").agg(
    total=("amount", "sum"),
    orders=("order_id", "count")
)

print("\nCustomer Order Summary:")
print(result)

