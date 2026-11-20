import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Load Wine dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Split dataset (same split as Decision Tree task)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train Single Decision Tree
dt_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\n--- Single Decision Tree ---")
print("Test Accuracy:", dt_accuracy)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n--- Random Forest ---")
print("OOB Score:", rf_model.oob_score_)
print("Test Accuracy:", rf_accuracy)

print("\n--- Accuracy Comparison ---")
print("Decision Tree:", dt_accuracy)
print("Random Forest:", rf_accuracy)

# Compare different numbers of trees
tree_counts = [10, 50, 100, 300]
accuracies = []

for n in tree_counts:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)

    print(f"Trees: {n} | Accuracy: {accuracy:.4f}")

    # Plot accuracy comparison
plt.figure(figsize=(8, 5))

plt.plot(tree_counts, accuracies, marker='o')

plt.title("Random Forest Accuracy vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Test Accuracy")
plt.xticks(tree_counts)
plt.grid(True)

plt.savefig("accuracy_comparison.png")
plt.show()

# Plot Feature Importances
importances = rf_model.feature_importances_

feature_importance = pd.Series(
    importances,
    index=X.columns
).sort_values()

plt.figure(figsize=(10, 6))

feature_importance.plot(kind="barh")

plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

# Bonus: Extra Trees Classifier
extra_model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

extra_model.fit(X_train, y_train)

extra_predictions = extra_model.predict(X_test)
extra_accuracy = accuracy_score(y_test, extra_predictions)

print("\n--- Extra Trees Classifier ---")
print("Test Accuracy:", extra_accuracy)

print("\n--- Final Model Comparison ---")
print("Decision Tree:", dt_accuracy)
print("Random Forest:", rf_accuracy)
print("Extra Trees:", extra_accuracy)

