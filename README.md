# Week 2 ML Assessment - Loan Default Project

Predict loan default (classification), estimate loan amount (regression) and segment customers (K-Means).

## Run
```bash
pip install -r requirements.txt
python loan_ml_project.py                 # built-in synthetic loan dataset
python loan_ml_project.py trainer_data.csv   # trainer's dataset (see "Using the trainer's dataset")
```
Outputs: `outputs/` (plots, results table, full report), `models/` (saved pipelines). Run from this folder,
because the saved pipeline imports `ml_utils.py`.

## Files
| File | Purpose |
|---|---|
| `loan_ml_project.py` | Tasks 1-4 in order (cells split with `# %%`, works in VS Code / Colab) |
| `ml_utils.py` | Data loader, `FeatureEngineer`, ColumnTransformer + Pipeline builders |
| `outputs/report.md` | Complete printed output of the last run |
| `outputs/results_table.csv` | Model \| Validation score table |
| `outputs/task*.png` | Overfitting, elbow/silhouette, ROC, confusion matrices, precision-recall |
| `models/loan_pipeline.joblib` | Task 2 pipeline (raw rows in, prediction out) |
| `models/best_model.joblib` | Best tuned model + chosen threshold |

## Key decisions
- **Split:** 60/20/20, `random_state=42`, stratified. Test set untouched until the final evaluation (used once).
- **No leakage:** imputing, feature engineering, encoding and scaling all live inside one `Pipeline`, so they are re-fitted on the training part of every CV fold.
- **Features:** date -> year/month/weekday, `income / loan_amount`, `log1p(income)`.
- **Encoding:** `OneHotEncoder` for home_ownership, loan_purpose; `OrdinalEncoder` for education (High School < Bachelors < Masters < PhD).
- **XGBoost early stopping** uses the validation set; the preprocessor for that set is fitted on train only.
- **Tuning** is done with CV on the training set only (ROC-AUC); the validation set picks the best model.
- **Regression** predicts `loan_amount` without the `loan_amount`-based ratio feature (to avoid target leakage).
- **Threshold:** the highest threshold on validation that keeps recall >= 75%, to get the best precision at that recall.

## Results of the included run (synthetic data, 13% defaults)
- Best model: **XGBoost (tuned)**, validation ROC-AUC 0.778; test ROC-AUC 0.794.
- At threshold 0.5 recall is only ~9% (accuracy 87% is misleading because 87% of customers do not default).
- At the validation-chosen threshold 0.125: test recall 0.72, precision 0.26. Recall dipped just under 75% on the test set, which is normal sampling variation.
- Regression: Ridge MAE 42 / RMSE 78 / R2 0.86 vs mean baseline MAE 152 / RMSE 211 / R2 0.00. Lasso (alpha=3) zeroes 7 of 19 coefficients.
- K-Means: best k=2 by silhouette (0.41): high-income/high-loan vs lower-income/lower-loan customers.
- Leakage demo: the effect of scaling before the split is tiny with StandardScaler here, but the principle (and the different mean used) is shown.

## Using the trainer's dataset
1. Make sure the target column is named in `ml_utils.TARGET` (default `default`, values 0/1).
2. Edit the column lists at the top of `ml_utils.py` (`NUMERIC_BASE`, `ORDINAL_COL`, `EDU_ORDER`, `NOMINAL_COLS`, `DATE_COL`).
3. If the dataset has no date column, or no `income` / `loan_amount`, adjust `FeatureEngineer` (e.g. use `tenure` / `monthly_charges` for Telco churn).
4. Re-run, then update the numbers in the section above.

## Git
```bash
git init && git add . && git commit -m "Week 2 ML assessment: loan default project"
```
`.env` and key files are already in `.gitignore`.
