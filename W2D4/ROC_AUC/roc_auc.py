# ==========================================
# ROC CURVE AND AUC
# ==========================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")


# 2. Create Imbalanced Dataset

X, y = make_classification(
    n_samples=3000,
    n_features=20,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)

print("Dataset created successfully!")
print("Dataset shape:", X.shape)

print("\nClass Distribution:")
print(pd.Series(y).value_counts())


# 3. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Define Three Models

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=9,
        eval_metric="logloss",
        random_state=42
    )
}


# 5. Train Models and Get Probabilities

probabilities = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    probabilities[name] = proba

    print(f"\n{name} trained successfully!")


# 6. Plot ROC Curves and Calculate AUC

plt.figure(figsize=(9, 6))

auc_results = {}

for name, proba in probabilities.items():

    fpr, tpr, thresholds = roc_curve(y_test, proba)

    auc = roc_auc_score(y_test, proba)

    auc_results[name] = auc

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

    print(f"{name} AUC: {auc:.4f}")


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid()

plt.savefig("roc_curve.png", dpi=300)

plt.show()


# 7. Best Threshold Function

def best_threshold(y_true, proba, min_recall):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        proba
    )

    best_precision = -1
    best_thresh = 0.0

    for i, threshold in enumerate(thresholds):

        if recall[i] >= min_recall:

            if precision[i] > best_precision:

                best_precision = precision[i]
                best_thresh = threshold

    return best_thresh


# 8. Find Threshold with At Least 80% Recall

threshold_results = {}

print("\n===== THRESHOLD ANALYSIS =====")

for name, proba in probabilities.items():

    threshold = best_threshold(
        y_test,
        proba,
        min_recall=0.80
    )

    predictions = (proba >= threshold).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions
    )

    threshold_results[name] = threshold

    print(f"\nModel: {name}")
    print(f"Best Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


# 9. Create Model Comparison Table

results = []

for name, proba in probabilities.items():

    predictions = (proba >= 0.5).astype(int)

    f1 = f1_score(y_test, predictions)

    results.append({
        "Model": name,
        "AUC": auc_results[name],
        "F1": f1
    })


results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")

print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# 10. Plot Precision-Recall Curves

plt.figure(figsize=(9, 6))

for name, proba in probabilities.items():

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        proba
    )

    plt.plot(
        recall,
        precision,
        label=name
    )


plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve Comparison")

plt.legend()
plt.grid()

plt.savefig(
    "precision_recall_curve.png",
    dpi=300
)

plt.show()


# 11. F1 Score vs AUC Explanation

print("\n===== F1 vs AUC =====")

print("1. F1 Score is useful when precision and recall are important.")
print("2. F1 is suitable when false positives and false negatives matter.")
print("3. AUC measures how well a model separates positive and negative classes.")
print("4. AUC is useful for comparing models across different thresholds.")

print("\nFor imbalanced data, Precision-Recall Curve is generally more informative than ROC Curve.")


# 12. Final Output

print("\n===== TASK COMPLETED SUCCESSFULLY =====")

print("ROC Curve saved as: roc_curve.png")
print("PR Curve saved as: precision_recall_curve.png")
print("Comparison table saved as: model_comparison.csv")

