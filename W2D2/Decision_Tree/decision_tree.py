import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load Wine dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("Dataset Shape:", X.shape)
print("\nTarget Classes:", wine.target_names)
print("\nFirst 5 Rows:")
print(X.head())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Decision Tree using Gini
gini_model = DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

gini_model.fit(X_train, y_train)

gini_predictions = gini_model.predict(X_test)

gini_accuracy = accuracy_score(y_test, gini_predictions)


# Train Decision Tree using Entropy
entropy_model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

entropy_model.fit(X_train, y_train)

entropy_predictions = entropy_model.predict(X_test)

entropy_accuracy = accuracy_score(y_test, entropy_predictions)


# Compare Accuracy
print("\nDecision Tree Accuracy Comparison")

print("Gini Accuracy:", gini_accuracy)
print("Entropy Accuracy:", entropy_accuracy)

# Hyperparameter Tuning

best_accuracy = 0
best_params = {}

for depth in [2, 3, 4, 5, 6, None]:
    for leaf in [1, 2, 3, 4, 5]:

        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'max_depth': depth,
                'min_samples_leaf': leaf
            }

print("\nBest Hyperparameters:")
print(best_params)

print("Best Test Accuracy:", best_accuracy)

# Plot Decision Tree

plt.figure(figsize=(20, 10))

plot_tree(
    gini_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.title("Decision Tree Visualization (Gini)")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")

plt.show()

# Feature Importance

importance = gini_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': wine.feature_names,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 5 Important Features:")
print(feature_importance.head(5).to_string(index=False))

# Save Feature Importance Table
feature_importance.head(5).to_csv(
    "feature_importance.csv",
    index=False
)

# Explain First Two Splits

tree = gini_model.tree_

feature_names = wine.feature_names

print("\nFirst Two Splits:")

for i in range(3):
    feature_index = tree.feature[i]

    if feature_index >= 0:
        feature = feature_names[feature_index]
        threshold = tree.threshold[i]

        print(f"Node {i}: {feature} <= {threshold:.2f}")

        # Gini Impurity Explanation

print("\nGini Impurity:")
print("1. Gini impurity measures how mixed the classes are in a dataset.")
print("2. A value of 0 means all samples belong to the same class.")
print("3. Lower Gini impurity indicates a better split in a decision tree.")

