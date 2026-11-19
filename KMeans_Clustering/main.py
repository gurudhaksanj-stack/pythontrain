import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Generate 300 data points with 4 centers

X, y = make_blobs(
n_samples=300,
centers=4,
random_state=7
)

print("Dataset Shape:", X.shape)
print("Number of Clusters:", len(np.unique(y)))

# Elbow Method

inertia_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

print("Inertia Values:", inertia_values)

# Plot Elbow Curve

plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 9))
plt.grid(True)
plt.show()

# Silhouette Score for K = 2 to 8

silhouette_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    score = silhouette_score(X, cluster_labels)
    silhouette_values.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

best_k = range(2, 9)[np.argmax(silhouette_values)]

print("\nBest k by silhouette:", best_k)
print("Best Silhouette Score:", round(max(silhouette_values), 4))

# Silhouette Score

silhouette_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    silhouette_values.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

best_k = list(range(2, 9))[np.argmax(silhouette_values)]

print("\nBest k by silhouette:", best_k)
print("Best Silhouette Score:", round(max(silhouette_values), 4))

# Final K-Means Clustering

final_kmeans = KMeans(n_clusters=best_k, random_state=7, n_init=10)

final_labels = final_kmeans.fit_predict(X)

centroids = final_kmeans.cluster_centers_

# Plot Final Clusters

plt.figure(figsize=(8, 6))

plt.scatter(
X[:, 0],
X[:, 1],
c=final_labels,
cmap='viridis',
s=50
)

plt.scatter(
centroids[:, 0],
centroids[:, 1],
marker='X',
s=250,
c='red',
label='Centroids'
)

plt.title("Final K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("Centroids:")
print(centroids)

# Mall Customers Style Dataset

customer_data = {
'Annual_Income': [
15, 18, 20, 22, 25,
30, 32, 35, 38, 40,
45, 48, 50, 55, 58,
60, 65, 68, 70, 75,
80, 85, 88, 90, 95,
100, 105, 110, 115, 120
],

'Spending_Score': [
    85, 80, 90, 78, 88,
    75, 82, 70, 85, 78,
    65, 72, 68, 75, 60,
    55, 62, 58, 65, 50,
    45, 52, 48, 55, 40,
    35, 42, 38, 45, 30
]

}

customers = pd.DataFrame(customer_data)

print("\nMall Customers Dataset:")
print(customers.head())

# Scale Customer Data

customer_scaler = StandardScaler()

customer_scaled = customer_scaler.fit_transform(customers)

print("\nScaled Customer Data:")
print(customer_scaled[:5])

# K-Means on Customer Data

customer_kmeans = KMeans(
    n_clusters=4,
    random_state=7,
    n_init=10
)

customers['Cluster'] = customer_kmeans.fit_predict(customer_scaled)

print("\nCustomer Segments:")
print(customers)

# Customer Segment Analysis

segment_summary = customers.groupby('Cluster')[
    ['Annual_Income', 'Spending_Score']
].mean()

print("\nCustomer Segment Summary:")
print(segment_summary)


# Plot Customer Segments

plt.figure(figsize=(8, 6))

plt.scatter(
    customers['Annual_Income'],
    customers['Spending_Score'],
    c=customers['Cluster'],
    cmap='viridis',
    s=80
)

plt.title("Mall Customers Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()


# Customer Segment Descriptions

for cluster, row in segment_summary.iterrows():

    income = row['Annual_Income']
    spending = row['Spending_Score']

    if income >= 60 and spending >= 60:
        description = "High income, high spending customers."

    elif income >= 60 and spending < 60:
        description = "High income, low spending customers."

    elif income < 60 and spending >= 60:
        description = "Low income, high spending customers."

    else:
        description = "Low income, low spending customers."

    print(f"Cluster {cluster}: {description}")

    