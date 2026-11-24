Task 2: Preprocessing, Features and Pipeline

1. Objective

The objective of this task is to build a complete Machine Learning preprocessing pipeline using a synthetic loan dataset.

The task covers missing value handling, categorical encoding, feature engineering, preprocessing pipelines, cross-validation, model saving and predictions.

2. Dataset

- Dataset: Synthetic Loan Dataset
- Number of Records: 600
- Problem Type: Binary Classification
- Target Column: "loan_default"
- Features include:
  - Application Date
  - Income
  - Loan Amount
  - Credit Score
  - Employment Type
  - Education Level
  - City

3. Missing Value Handling

Used "SimpleImputer" to handle missing values.

- Numerical Columns: Median strategy
- Categorical Columns: Most Frequent strategy

Also experimented with "KNNImputer" on numerical columns using 5 nearest neighbors.

4. Categorical Encoding

Implemented two encoding techniques:

OneHotEncoder

Used for nominal categorical columns:

- Employment Type
- City

OrdinalEncoder

Used for ordered categorical column:

- Education Level

Education order:

1. High School
2. Diploma
3. Bachelor
4. Master
5. PhD

5. Feature Engineering

Created three new features:

1. Application Month

Extracted the month from the application date.

2. Income-to-Loan Ratio

Calculated using:

"income / loan_amount"

3. Log Loan Amount

Applied "np.log1p()" to transform the loan amount.

6. Preprocessing Pipeline

Used "ColumnTransformer" to combine numerical, categorical and ordinal preprocessing.

Numerical Pipeline

- SimpleImputer (median)
- StandardScaler

Categorical Pipeline

- SimpleImputer (most frequent)
- OneHotEncoder

Ordinal Pipeline

- SimpleImputer (most frequent)
- OrdinalEncoder

7. Machine Learning Model

- Algorithm: Logistic Regression
- Maximum Iterations: 2000
- Evaluation: 5-Fold Cross-Validation
- Scoring Metric: Accuracy

All preprocessing steps are included inside the Scikit-learn Pipeline to prevent data leakage during cross-validation.

8. Model Saving and Loading

Used Joblib to save and load the fitted pipeline.

Model artifact:

"loan_pipeline.joblib"

9. Prediction

Loaded the saved pipeline and predicted loan default outcomes for three new raw data records.

The pipeline automatically performs feature engineering and preprocessing before generating predictions.

10. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

11. Final Outcome

Successfully implemented a complete Machine Learning preprocessing workflow covering:

- Missing Value Imputation
- Numerical and Categorical Encoding
- Feature Engineering
- ColumnTransformer
- Machine Learning Pipeline
- Cross-Validation
- Model Serialization
- Predictions on New Data

12. Project Files

- "task2_preprocessing_pipeline.ipynb" – Complete implementation.
- "README.md" – Task documentation.
- "loan_pipeline.joblib" – Saved fitted ML pipeline.

