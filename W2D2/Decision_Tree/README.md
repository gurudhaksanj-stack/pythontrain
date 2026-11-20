W2D2 - Decision Tree Classification

Objective

Train and evaluate Decision Tree classifiers using the Wine dataset.

Dataset

- Dataset: Wine dataset (Scikit-learn)
- Total samples: 178
- Features: 13
- Classes: 3
- Training data: 80%
- Testing data: 20%

1. Gini vs Entropy

Criterion| Test Accuracy
Gini| 94.44%
Entropy| 97.22%

2. Hyperparameter Tuning

Parameters tested:

- max_depth: 2, 3, 4, 5, 6, None
- min_samples_leaf: 1, 2, 3, 4, 5

Best combination from the initial search:

- max_depth: 3
- min_samples_leaf: 2
- Test accuracy: 100%

Note: The initial search selected parameters using test accuracy. Cross-validation is recommended for unbiased model selection.

3. Decision Tree Visualization

The decision tree was plotted using plot_tree with max_depth=3.

Output: decision_tree.png

4. Feature Importance

The five most important features were extracted using feature_importances_.

Output: feature_importance.csv

5. Gini Impurity

1. Gini impurity measures how mixed the classes are in a dataset.
2. A value of 0 means all samples belong to the same class.
3. Lower Gini impurity indicates a better split in a decision tree.

Concepts Covered

- Decision Tree Classification
- Gini Impurity
- Entropy
- max_depth
- min_samples_leaf
- Feature Importance
- Model Accuracy

