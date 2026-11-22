import time
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


# ==========================================
# 1. CREATE IMBALANCED DATASET
# ==========================================

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ==========================================
# 2. RANDOM FOREST MODEL
# ==========================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 3. GRID SEARCH CV
# ==========================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

start_time = time.time()

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time

print("\n========== GRID SEARCH RESULTS ==========")

print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", round(grid_search.best_score_, 4))
print("Execution Time:", round(grid_time, 2), "seconds")


# ==========================================
# 4. RANDOMIZED SEARCH CV
# ==========================================

param_distributions = {
    "n_estimators": [100, 150, 200, 250, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20, 25, 30, 40],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 10]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()

random_search.fit(X_train, y_train)

random_time = time.time() - start_time

print("\n========== RANDOMIZED SEARCH RESULTS ==========")

print("Best Parameters:", random_search.best_params_)
print("Best F1 Score:", round(random_search.best_score_, 4))
print("Execution Time:", round(random_time, 2), "seconds")


# ==========================================
# 5. COMPARE BOTH SEARCH METHODS
# ==========================================

comparison = pd.DataFrame({
    "Search Method": ["Grid Search", "Randomized Search"],
    "Best F1 Score": [
        grid_search.best_score_,
        random_search.best_score_
    ],
    "Execution Time (seconds)": [
        grid_time,
        random_time
    ]
})

print("\n========== SEARCH COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv("search_comparison.csv", index=False)


# ==========================================
# 6. CROSS-VALIDATION RESULTS AS DATAFRAME
# ==========================================

grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results["Search Method"] = "Grid Search"

random_results = pd.DataFrame(random_search.cv_results_)
random_results["Search Method"] = "Randomized Search"

cv_results = pd.concat(
    [grid_results, random_results],
    ignore_index=True
)

selected_columns = [
    "Search Method",
    "params",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]

cv_results = cv_results[selected_columns]

print("\n========== CROSS VALIDATION RESULTS ==========")
print(cv_results.to_string(index=False))

cv_results.to_csv("cross_validation_results.csv", index=False)


# ==========================================
# 7. SELECT BEST MODEL USING CV SCORE
# ==========================================

if grid_search.best_score_ >= random_search.best_score_:
    best_model = grid_search.best_estimator_
    best_method = "Grid Search"
else:
    best_model = random_search.best_estimator_
    best_method = "Randomized Search"

print("\nSelected Method:", best_method)


# ==========================================
# 8. FINAL TEST SET EVALUATION
# ==========================================

# Test set is evaluated only once
y_pred = best_model.predict(X_test)

test_f1 = f1_score(y_test, y_pred)

print("\n========== FINAL TEST RESULTS ==========")
print("Selected Model:", best_method)
print("Test F1 Score:", round(test_f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. WHY NOT TUNE ON THE TEST SET?
# ==========================================

print("\n========== TUNING DISCIPLINE ==========")

print("1. The test set must remain unseen during model tuning.")
print("2. Tuning on test data causes data leakage and overfitting.")
print("3. Use cross-validation for tuning and the test set for final evaluation.")


