# ==========================================
# W2D4 - Task 2: Metrics on Imbalanced Data
# ==========================================


# 1. Import Libraries

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 2. Create Imbalanced Dataset
# ==========================================

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

print("Dataset Shape:", X.shape)

print("\nClass Distribution:")
print(pd.Series(y).value_counts())

print("\nClass Percentages:")
print(pd.Series(y).value_counts(normalize=True) * 100)


# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ==========================================
# 4. Majority Class Model
# ==========================================

majority_model = DummyClassifier(strategy="most_frequent")

majority_model.fit(X_train, y_train)

y_pred_majority = majority_model.predict(X_test)


# Calculate Metrics

majority_accuracy = accuracy_score(y_test, y_pred_majority)

majority_precision = precision_score(
    y_test, y_pred_majority, zero_division=0
)

majority_recall = recall_score(
    y_test, y_pred_majority, zero_division=0
)

majority_f1 = f1_score(
    y_test, y_pred_majority, zero_division=0
)


print("\n========== Majority Class Model ==========")

print("Accuracy:", majority_accuracy)
print("Precision:", majority_precision)
print("Recall:", majority_recall)
print("F1 Score:", majority_f1)


# ==========================================
# 5. Logistic Regression
# ==========================================

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)


print("\n========== Logistic Regression ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_logistic))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_logistic,
        zero_division=0
    )
)


# ==========================================
# 6. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred_logistic)

print("\n========== Confusion Matrix ==========")

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ==========================================
# 7. Fraud Detection Example
# ==========================================

print("\n========== Fraud Detection Explanation ==========")

print("""
True Positive (TP):
Actual fraud transaction correctly predicted as fraud.

True Negative (TN):
Genuine transaction correctly predicted as genuine.

False Positive (FP):
Genuine transaction incorrectly predicted as fraud.

False Negative (FN):
Fraud transaction incorrectly predicted as genuine.

In fraud detection, False Negatives are dangerous
because actual fraud transactions are missed.
""")


# ==========================================
# 8. Logistic Regression With Balanced Weights
# ==========================================

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train, y_train)

y_pred_balanced = balanced_model.predict(X_test)


print("\n========== Balanced Logistic Regression ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))

print("Precision:", precision_score(
    y_test, y_pred_balanced, zero_division=0
))

print("Recall:", recall_score(
    y_test, y_pred_balanced, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, y_pred_balanced, zero_division=0
))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_balanced,
        zero_division=0
    )
)


# ==========================================
# 9. Compare Model Performance
# ==========================================

comparison = pd.DataFrame({

    "Model": [
        "Majority Class Model",
        "Logistic Regression",
        "Logistic Regression Balanced"
    ],

    "Accuracy": [
        majority_accuracy,
        accuracy_score(y_test, y_pred_logistic),
        accuracy_score(y_test, y_pred_balanced)
    ],

    "Precision": [
        majority_precision,
        precision_score(y_test, y_pred_logistic, zero_division=0),
        precision_score(y_test, y_pred_balanced, zero_division=0)
    ],

    "Recall": [
        majority_recall,
        recall_score(y_test, y_pred_logistic, zero_division=0),
        recall_score(y_test, y_pred_balanced, zero_division=0)
    ],

    "F1 Score": [
        majority_f1,
        f1_score(y_test, y_pred_logistic, zero_division=0),
        f1_score(y_test, y_pred_balanced, zero_division=0)
    ]

})


print("\n========== Model Comparison ==========")

print(comparison.to_string(index=False))


# ==========================================
# 10. Concepts and Explanation
# ==========================================

print("\n========== Concepts ==========")

print("""
1. Accuracy:
   Overall percentage of correct predictions.

2. Precision:
   Out of predicted positive cases, how many are correct.

3. Recall:
   Out of actual positive cases, how many are identified.

4. F1 Score:
   Harmonic mean of precision and recall.

5. Confusion Matrix:
   Shows correct and incorrect predictions.

6. Class Imbalance:
   One class has significantly more samples than another.

7. Class Weight Balanced:
   Gives more importance to the minority class during training.

8. Majority Class Model:
   Always predicts the most frequent class.
""")


print("\nImbalanced Data Task Completed Successfully!")

