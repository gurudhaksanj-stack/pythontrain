W2D2 - Random Forest

Objective

To understand Random Forest, Bootstrap Aggregation (Bagging), OOB Score, and Feature Importance.

Dataset

- Dataset: Wine Classification Dataset
- Source: Scikit-learn
- Total samples: 178
- Features: 13
- Classes: 3
- Train-test split: 80% Training / 20% Testing

Models Implemented

1. Decision Tree Classifier
2. Random Forest Classifier (200 trees)
3. Extra Trees Classifier (Bonus)

Requirements Completed

- Trained RandomForestClassifier with 200 estimators.
- Calculated Out-of-Bag (OOB) score.
- Compared Decision Tree and Random Forest test accuracy.
- Tested 10, 50, 100, and 300 trees.
- Plotted accuracy comparison.
- Visualized feature importances.
- Compared Random Forest with Extra Trees Classifier.

Concepts

Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and stability.

Bootstrap Aggregation (Bagging)

Bagging trains multiple models using different bootstrap samples of the training data. Bootstrap sampling selects data points randomly with replacement.

For classification, the forest combines predictions using majority voting.

OOB Score

Out-of-Bag score evaluates the model using training samples that were not selected in each tree's bootstrap sample.

Why Random Forest Overfits Less

A single decision tree can overfit by learning noise in the training data. Random Forest combines predictions from multiple diverse trees, reducing variance and improving generalization.

However, Random Forest can still overfit in some situations.

Extra Trees Classifier

Extra Trees is another ensemble method. It introduces additional randomness when selecting split thresholds, which can help improve model diversity.

Output Files

- "random_forest.py"
- "accuracy_comparison.png"
- "feature_importance.png"
- "README.md"

