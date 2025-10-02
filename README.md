## End-to-End EDA and ML Pipeline of the "UCI Adult Income" dataset

Predicting whether an individual earns >$50K/year using the UCI Adult dataset. The notebook walks through a complete, reproducible workflow:
- Data loading and cleaning
- Exploratory Data Analysis (EDA) with visuals
- Encoding, scaling, and class-imbalance handling (SMOTE)
- Training and evaluating multiple models
- Hyperparameter tuning (Grid Search, Randomized Search, Optuna)
- Final comparison and conclusion

## Dataset
- Source: UCI Machine Learning Repository (`adult.data`, `adult.test`)
- Target: `income` (binary: `<=50K` vs `>50K`)

## Highlights of the Pipeline
- Cleaning:
  - Harmonized test labels (`>50K.` → `>50K`)
  - Removed duplicates
  - Imputed categorical missing values with training-mode
- Encoding:
  - Label encoding for binary (`sex`, `income`)
  - Ordinal encoding for `education` (ordered levels)
  - One-Hot encoding for nominal features (`workclass`, `marital-status`, `occupation`, `relationship`, `race`, `native-country`)
- Scaling:
  - `StandardScaler` on numerical features
- Class imbalance:
  - `SMOTE` on the training set
- Models evaluated:
  - KNN, Decision Tree, Random Forest, XGBoost, LightGBM, Logistic Regression, SVM (linear/RBF), MLP
- Evaluation:
  - Accuracy, Precision, Recall, F1
  - Confusion Matrix, ROC-AUC, Precision–Recall curve
- Hyperparameter tuning:
  - Grid Search (KNN)
  - Randomized Search (Random Forest)
  - Bayesian Optimization with Optuna (MLP)
- Cross-validation:
  - Example with Logistic Regression

## Key Results (test set)
- KNN: F1 ≈ 0.355 (Acc ≈ 0.603)
- Decision Tree: F1 ≈ 0.597 (Acc ≈ 0.805)
- Random Forest: F1 ≈ 0.661 (Acc ≈ 0.849)
- XGBoost: F1 ≈ 0.706 (Acc ≈ 0.870)
- LightGBM: F1 ≈ 0.703 (Acc ≈ 0.869)
- Logistic Regression: F1 ≈ 0.672 (Acc ≈ 0.814)
- MLP: F1 ≈ 0.612 (Acc ≈ 0.823)

Conclusion: XGBoost delivered the best F1 (≈ 0.706), balancing precision and recall most effectively in this setup.


*The notebook downloads the dataset directly from UCI and reproduces all steps end-to-end.*
