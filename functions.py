prices = [500, 1200, 800, 2500, 300]

above_1000 = [price for price in prices if price > 1000]

print("Above 1000:", above_1000)

with_gst = list(map(lambda price: round(price * 1.18, 2), prices))

print("With GST:", with_gst)

below_1000 = list(filter(lambda price: price < 1000, prices))

print("Below 1000:", below_1000)

prices = [500, 1200, 800, 2500, 300]

total_sales = 0

for price in prices:
    total_sales += price

print("Total Sales using for loop:", total_sales)

total_sales = 0

i = 0

while i < len(prices):
    total_sales += prices[i]
    i += 1

print("Total Sales using while loop:", total_sales)

