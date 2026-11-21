# ==========================================
# W2D3 - Task 2: Scaling
# ==========================================

# 1. Import Libraries

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


# ==========================================
# 2. Load Breast Cancer Dataset
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", X.shape)
print("\nFirst 5 Rows:")
print(X.head())

print("\nTarget Names:")
print(data.target_names)


# ==========================================
# 3. StandardScaler - Age and Income
# ==========================================

sample_data = pd.DataFrame({
    'age': [22, 25, 30, 35, 40, 45, 50, 55],
    'income': [20000, 25000, 30000, 35000,
               40000, 45000, 50000, 55000]
})

sample_train, sample_test = train_test_split(
    sample_data,
    test_size=0.25,
    random_state=42
)

standard_scaler = StandardScaler()

scaled_train = standard_scaler.fit_transform(sample_train)
scaled_test = standard_scaler.transform(sample_test)

scaled_train_df = pd.DataFrame(
    scaled_train,
    columns=sample_data.columns
)

print("\n========== StandardScaler ==========")

print("\nScaled Training Data:")
print(scaled_train_df)

print("\nMean:")
print(scaled_train_df.mean())

print("\nStandard Deviation:")
print(scaled_train_df.std(ddof=0))


# ==========================================
# 4. MinMaxScaler - Age and Income
# ==========================================

minmax_scaler = MinMaxScaler()

minmax_train = minmax_scaler.fit_transform(sample_train)
minmax_test = minmax_scaler.transform(sample_test)

minmax_train_df = pd.DataFrame(
    minmax_train,
    columns=sample_data.columns
)

print("\n========== MinMaxScaler ==========")

print("\nScaled Training Data:")
print(minmax_train_df)

print("\nMinimum Values:")
print(minmax_train_df.min())

print("\nMaximum Values:")
print(minmax_train_df.max())


# ==========================================
# 5. Split Breast Cancer Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. KNN Without Scaling
# ==========================================

knn_without_scaling = KNeighborsClassifier(n_neighbors=5)

knn_without_scaling.fit(X_train, y_train)

y_pred_without = knn_without_scaling.predict(X_test)

accuracy_without = accuracy_score(y_test, y_pred_without)

print("\n========== KNN Without Scaling ==========")

print("Accuracy:", accuracy_without)


# ==========================================
# 7. KNN With StandardScaler
# ==========================================

knn_scaler = StandardScaler()

X_train_scaled = knn_scaler.fit_transform(X_train)

X_test_scaled = knn_scaler.transform(X_test)

knn_with_scaling = KNeighborsClassifier(n_neighbors=5)

knn_with_scaling.fit(X_train_scaled, y_train)

y_pred_scaled = knn_with_scaling.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print("\n========== KNN With Scaling ==========")

print("Accuracy:", accuracy_scaled)


# ==========================================
# 8. Decision Tree Without Scaling
# ==========================================

tree_without_scaling = DecisionTreeClassifier(random_state=42)

tree_without_scaling.fit(X_train, y_train)

tree_pred_without = tree_without_scaling.predict(X_test)

tree_accuracy_without = accuracy_score(
    y_test, tree_pred_without
)

print("\n========== Decision Tree Without Scaling ==========")

print("Accuracy:", tree_accuracy_without)


# ==========================================
# 9. Decision Tree With Scaling
# ==========================================

tree_with_scaling = DecisionTreeClassifier(random_state=42)

tree_with_scaling.fit(X_train_scaled, y_train)

tree_pred_scaled = tree_with_scaling.predict(X_test_scaled)

tree_accuracy_scaled = accuracy_score(
    y_test, tree_pred_scaled
)

print("\n========== Decision Tree With Scaling ==========")

print("Accuracy:", tree_accuracy_scaled)


# ==========================================
# 10. Accuracy Comparison
# ==========================================

comparison = pd.DataFrame({
    'Model': [
        'KNN Without Scaling',
        'KNN With Scaling',
        'Decision Tree Without Scaling',
        'Decision Tree With Scaling'
    ],

    'Accuracy': [
        accuracy_without,
        accuracy_scaled,
        tree_accuracy_without,
        tree_accuracy_scaled
    ]
})

print("\n========== Accuracy Comparison ==========")

print(comparison)


# ==========================================
# 11. RobustScaler - Bonus Challenge
# ==========================================

outlier_data = pd.DataFrame({
    'income': [20000, 25000, 30000, 35000,
               40000, 45000, 50000, 500000]
})

robust_scaler = RobustScaler()

robust_scaled = robust_scaler.fit_transform(outlier_data)

print("\n========== RobustScaler ==========")

print("\nOriginal Data:")
print(outlier_data)

print("\nRobust Scaled Data:")
print(robust_scaled)

print("\nRobustScaler Median:")
print(robust_scaler.center_)

print("\nRobustScaler IQR:")
print(robust_scaler.scale_)


# ==========================================
# 12. Concepts and Explanation
# ==========================================

print("\n========== Concepts ==========")

print("""
1. StandardScaler:
   Transforms data to mean 0 and standard deviation 1.

2. MinMaxScaler:
   Scales data between 0 and 1.

3. KNN:
   Scaling matters because KNN uses distance calculations.

4. Decision Tree:
   Scaling does not significantly affect decision tree
   predictions because it uses feature-based split thresholds.

5. RobustScaler:
   Uses median and interquartile range (IQR).
   It is less affected by outliers.

6. Data Leakage:
   Always fit scalers on training data only.
   Use transform() on test data.
""")

print("\nScaling Task Completed Successfully!")

