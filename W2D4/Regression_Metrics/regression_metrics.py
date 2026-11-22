# ==========================================
# REGRESSION METRICS
# ==========================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# 2. Load California Housing Dataset

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Loaded Successfully!")

print("Dataset Shape:", X.shape)

print("\nFirst Five Rows:")
print(X.head())


# 3. Train Test Split (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 4. Define Models

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

}


# 5. Train Models and Make Predictions

predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    print(f"\n{name} trained successfully!")


# 6. Mean Baseline Prediction

baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

predictions["Mean Baseline"] = baseline_prediction


# 7. Calculate Regression Metrics

results = []

for name, y_pred in predictions.items():

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# 8. Display Comparison Table

results_df = pd.DataFrame(results)

print("\n===== REGRESSION MODEL COMPARISON =====")

print(results_df.to_string(index=False))

results_df.to_csv(
    "regression_metrics.csv",
    index=False
)


# 9. Predicted vs Actual Plot

for name, y_pred in predictions.items():

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.4
    )

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.title(f"Predicted vs Actual - {name}")

    plt.grid()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"predicted_vs_actual_{filename}.png",
        dpi=300
    )

    plt.show()


# 10. Residual Analysis

for name, y_pred in predictions.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.4
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    plt.title(f"Residual Plot - {name}")

    plt.grid()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"residuals_{filename}.png",
        dpi=300
    )

    plt.show()


# 11. MAE vs RMSE and R-squared Explanation

print("\n===== CONCEPT EXPLANATION =====")

print("1. MAE measures the average absolute difference between actual and predicted values.")
print("2. RMSE penalizes larger prediction errors more heavily than MAE.")
print("3. R-squared measures how much variation in the target is explained by the model.")
print("4. R-squared of 0.6 means the model explains 60% of the target variance.")


# 12. Final Output

print("\n===== TASK COMPLETED SUCCESSFULLY =====")

print("Comparison Table: regression_metrics.csv")
print("Predicted vs Actual plots generated.")
print("Residual plots generated.")


