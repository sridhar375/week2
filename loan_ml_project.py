"""
Week 2 - Machine Learning Weekly Assessment : Loan default project
Run:  python loan_ml_project.py                 (uses built-in synthetic loan data)
      python loan_ml_project.py my_data.csv     (uses the trainer's CSV - see README)
Everything is split in '# %%' cells, so it also runs cell-by-cell in VS Code / Colab.
"""
# %% Imports
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import ml_utils as U

warnings.filterwarnings("ignore")
RS = 42
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
REPORT = []          # lines that end up in outputs/report.md
RESULTS = []         # rows for the "Model | Validation score" table


def log(text=""):
    print(text)
    REPORT.append(text)


def header(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# %% Load data
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else None
df = U.load_data(DATA_PATH)
df.to_csv("outputs/loan_data_used.csv", index=False)
log(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Default rate: {df[U.TARGET].mean():.3f}")
log("Missing values per column:\n" + df.isna().sum()[df.isna().sum() > 0].to_string())

X = df.drop(columns=[U.TARGET])
y = df[U.TARGET]

# =============================================================================
header("TASK 1 - Concepts, splits and leakage")
# =============================================================================
# 60/20/20 stratified split: first cut off 20% test, then 25% of the rest (=20% overall) for validation
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25,
                                                  stratify=y_tmp, random_state=RS)
log(f"Train {len(X_train)} | Validation {len(X_val)} | Test {len(X_test)}  (test set put aside)")
log(f"Default rate  train {y_train.mean():.3f} | val {y_val.mean():.3f} | test {y_test.mean():.3f}")

# --- 1a. Overfitting with DecisionTree depth ---------------------------------
num_raw = X.select_dtypes("number").columns.tolist()


def quick(model, scale=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("model", model)])


depths = [1, 3, 5, 10, None]
tr_acc, va_acc = [], []
log("\nOverfitting demo (DecisionTreeClassifier on numeric columns):")
for d in depths:
    m = quick(DecisionTreeClassifier(max_depth=d, random_state=RS)).fit(X_train[num_raw], y_train)
    tr_acc.append(m.score(X_train[num_raw], y_train))
    va_acc.append(m.score(X_val[num_raw], y_val))
    log(f"  Depth {str(d):>4}: train {tr_acc[-1]:.2f} val {va_acc[-1]:.2f}")
plt.figure(figsize=(6, 4))
labels = [str(d) for d in depths]
plt.plot(labels, tr_acc, "o-", label="Train")
plt.plot(labels, va_acc, "s-", label="Validation")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.title("Overfitting: train vs validation")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("outputs/task1_overfitting.png", dpi=150); plt.close()

# --- 1b. 5-fold CV for logistic regression ------------------------------------
cv_scores = cross_val_score(quick(LogisticRegression(max_iter=1000), scale=True),
                            X_train[num_raw], y_train, cv=5, scoring="accuracy")
log(f"\nCV (LogisticRegression, 5-fold): {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

# --- 1c. Data leakage demo -----------------------------------------------------
med = X_train[num_raw].median()                       # same fill values for both ways
fill = lambda d: d[num_raw].fillna(med)
# WRONG: scaler sees the whole dataset (incl. validation + test) before the split
leaky_sc = StandardScaler().fit(fill(X))
lr_leak = LogisticRegression(max_iter=1000).fit(leaky_sc.transform(fill(X_train)), y_train)
acc_leak = lr_leak.score(leaky_sc.transform(fill(X_val)), y_val)
# RIGHT: scaler fitted on the training part only
good_sc = StandardScaler().fit(fill(X_train))
lr_good = LogisticRegression(max_iter=1000).fit(good_sc.transform(fill(X_train)), y_train)
acc_good = lr_good.score(good_sc.transform(fill(X_val)), y_val)
i = num_raw.index("income")
log("\nData leakage demo (scaling):")
log(f"  income mean used for scaling -> leaky (all data): {leaky_sc.mean_[i]:.2f} | correct (train only): {good_sc.mean_[i]:.2f}")
log(f"  Validation accuracy -> leaky: {acc_leak:.4f} | correct: {acc_good:.4f}")
log("  Explanation (4 lines):")
log("  1) Scaling before the split lets mean/std be computed from validation and test rows too.")
log("  2) The model therefore 'sees' information about unseen data, so scores look slightly optimistic.")
log("  3) The correct way: split first, fit the scaler on train only, then only transform val/test.")
log("  4) Putting the scaler inside a Pipeline does this automatically in every CV fold.")

# --- 1d. Supervised vs unsupervised + workflow --------------------------------
log("\nClassify the 4 problems:")
log("  Spam detection         -> SUPERVISED (classification, labelled spam / not spam)")
log("  Customer segmentation  -> UNSUPERVISED (clustering, no labels)")
log("  House price prediction -> SUPERVISED (regression, numeric target)")
log("  Anomaly detection      -> UNSUPERVISED (mostly; no/rare labels, finds unusual points)")
log("\nML workflow in 7 steps:")
for s in ["1. Define the problem and the success metric (e.g. recall / ROC-AUC for default).",
          "2. Collect the data and explore it (EDA, missing values, class balance).",
          "3. Split into train / validation / test BEFORE any fitting; keep test aside.",
          "4. Preprocess and engineer features inside a Pipeline (impute, encode, scale).",
          "5. Train several models and compare them on the validation set / cross-validation.",
          "6. Tune hyperparameters, choose the threshold, avoid overfitting and leakage.",
          "7. Evaluate the single best model once on the test set, save it, deploy and monitor."]:
    log("  " + s)

# =============================================================================
header("TASK 2 - Preprocessing, features and pipeline")
# =============================================================================
cv5 = StratifiedKFold(5, shuffle=True, random_state=RS)
lr_params = dict(max_iter=2000, random_state=RS)
acc_median = cross_val_score(U.make_pipeline(LogisticRegression(**lr_params), numeric_imputer="median"),
                             X_train, y_train, cv=cv5, scoring="accuracy").mean()
acc_knn = cross_val_score(U.make_pipeline(LogisticRegression(**lr_params), numeric_imputer="knn"),
                          X_train, y_train, cv=cv5, scoring="accuracy").mean()
log("Imputation comparison (5-fold CV accuracy, whole pipeline):")
log(f"  SimpleImputer(median) : {acc_median:.4f}")
log(f"  KNNImputer(k=5)       : {acc_knn:.4f}")
log("Encoding: OneHotEncoder for home_ownership + loan_purpose, OrdinalEncoder for education")
log("          (High School < Bachelors < Masters < PhD)")
log("New features: app_year / app_month / app_dayofweek (date), income_to_loan (interaction), log_income (log)")

pipe2 = U.make_pipeline(LogisticRegression(**lr_params))
cv_acc = cross_val_score(pipe2, X_train, y_train, cv=cv5, scoring="accuracy")
log(f"\nPipeline CV accuracy: {cv_acc.mean():.2f} (+/- {cv_acc.std():.2f})")
pipe2.fit(X_train, y_train)
joblib.dump(pipe2, "models/loan_pipeline.joblib")

new_rows = pd.DataFrame([
    dict(age=45, income=1200, employment_years=20, credit_score=780, num_credit_lines=6,
         debt_to_income=0.15, loan_amount=250, education="Masters", home_ownership="OWN",
         loan_purpose="Home", application_date="2024-06-15"),
    dict(age=24, income=250, employment_years=1, credit_score=520, num_credit_lines=2,
         debt_to_income=0.65, loan_amount=400, education="High School", home_ownership="RENT",
         loan_purpose="Business", application_date="2024-11-03"),
    dict(age=33, income=np.nan, employment_years=6, credit_score=np.nan, num_credit_lines=4,
         debt_to_income=0.32, loan_amount=300, education=np.nan, home_ownership="MORTGAGE",
         loan_purpose="Personal", application_date="2024-02-20"),
])
loaded = joblib.load("models/loan_pipeline.joblib")
log(f"Predictions for 3 new rows: {loaded.predict(new_rows)}   "
    f"(default probabilities: {np.round(loaded.predict_proba(new_rows)[:, 1], 3)})")
log("Saved: models/loan_pipeline.joblib (raw rows go in; imputing/encoding/scaling happen inside)")

# =============================================================================
header("TASK 3 - Models: supervised and unsupervised")
# =============================================================================
models = {
    "LogisticRegression": U.make_pipeline(LogisticRegression(max_iter=2000, random_state=RS)),
    "DecisionTree": U.make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                           random_state=RS)),
    "RandomForest": U.make_pipeline(RandomForestClassifier(n_estimators=300, oob_score=True,
                                                           min_samples_leaf=3, n_jobs=-1,
                                                           random_state=RS)),
    "XGBoost": U.make_pipeline(XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                                             subsample=0.8, colsample_bytree=0.8,
                                             eval_metric="logloss", early_stopping_rounds=30,
                                             random_state=RS, n_jobs=1)),
}
for name, pipe in models.items():
    if name == "XGBoost":
        # early stopping needs a transformed validation set -> fit the preprocessing on TRAIN only
        pre = Pipeline(pipe.steps[:-1]).fit(X_train, y_train)
        pipe.fit(X_train, y_train, model__eval_set=[(pre.transform(X_val), y_val)],
                 model__verbose=False)
        log(f"XGBoost stopped at best iteration {pipe.named_steps['model'].best_iteration}")
    else:
        pipe.fit(X_train, y_train)
    if name == "RandomForest":
        log(f"RandomForest OOB score: {pipe.named_steps['model'].oob_score_:.4f}")

# predict_proba vs predict
lr = models["LogisticRegression"]
proba = lr.predict_proba(X_val.iloc[:5])
cmp_df = pd.DataFrame({"P(no default)": proba[:, 0].round(3), "P(default)": proba[:, 1].round(3),
                       "predict (thr 0.5)": lr.predict(X_val.iloc[:5]),
                       "actual": y_val.iloc[:5].values})
log("\nLogisticRegression predict_proba vs predict (5 validation rows):\n" + cmp_df.to_string())
log("predict() is just 'P(default) >= 0.5'; predict_proba lets us move the threshold later (Task 4).")

# feature importances
names = U.feature_names(models["RandomForest"])
for name in ["RandomForest", "XGBoost"]:
    imp = models[name].named_steps["model"].feature_importances_
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(5)
    log(f"\nTop 5 features - {name}:\n" + top.round(4).to_string())

# validation scores of the classifiers
val_proba = {n: m.predict_proba(X_val)[:, 1] for n, m in models.items()}
for n, m in models.items():
    RESULTS.append({"Model": n, "Task": "Classification",
                    "Validation score": f"ROC-AUC {roc_auc_score(y_val, val_proba[n]):.3f} | "
                                        f"Acc {m.score(X_val, y_val):.3f}"})

# --- Regression: predict loan_amount ------------------------------------------
header("Regression: predict loan_amount (Linear / Ridge / Lasso)")
Xr_train, Xr_val, Xr_test = (d.drop(columns=["loan_amount"]) for d in (X_train, X_val, X_test))
yr_train, yr_val, yr_test = X_train["loan_amount"], X_val["loan_amount"], X_test["loan_amount"]
reg_models = {
    "LinearRegression": U.make_pipeline(LinearRegression(), task="reg"),
    "Ridge(alpha=10)": U.make_pipeline(Ridge(alpha=10), task="reg"),
    "Lasso(alpha=3)": U.make_pipeline(Lasso(alpha=3, max_iter=20000), task="reg"),
}
coef_table = {}
reg_val_r2 = {}
for n, m in reg_models.items():
    m.fit(Xr_train, yr_train)
    coef_table[n] = m.named_steps["model"].coef_
    reg_val_r2[n] = r2_score(yr_val, m.predict(Xr_val))
    RESULTS.append({"Model": n, "Task": "Regression (loan_amount)",
                    "Validation score": f"R2 {reg_val_r2[n]:.3f}"})
coefs = pd.DataFrame(coef_table, index=U.feature_names(reg_models["LinearRegression"])).round(2)
log("Coefficients (features are standardised, so they are comparable):\n" + coefs.to_string())
n_zero = int((np.abs(coef_table["Lasso(alpha=3)"]) < 1e-8).sum())
log(f"\nLasso zero coefficients: {n_zero} of {len(coefs)}  -> Lasso removes weak features")
log("Ridge shrinks coefficients towards 0 but keeps all of them; Lasso can set them exactly to 0.")
log("Validation R2: " + " | ".join(f"{k} {v:.3f}" for k, v in reg_val_r2.items()))

# --- K-Means --------------------------------------------------------------------
header("K-Means customer segmentation (income, loan_amount, credit_score)")
km_cols = ["income", "loan_amount", "credit_score"]
km_data = X_train[km_cols].fillna(X_train[km_cols].median())
km_scaled = StandardScaler().fit_transform(km_data)
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(km_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(km_scaled, km.labels_))
    log(f"  k={k}: inertia {inertias[-1]:.1f} | silhouette {sils[-1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertias, "o-"); ax[0].set_title("Elbow curve")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sils, "s-", color="tab:orange"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/task3_elbow_silhouette.png", dpi=150); plt.close()
best_k = list(ks)[int(np.argmax(sils))]
log(f"\nBest k by silhouette: {best_k}")
km = KMeans(n_clusters=best_k, n_init=10, random_state=RS).fit(km_scaled)
seg = km_data.assign(cluster=km.labels_, default=y_train.values)
summary = seg.groupby("cluster").agg(customers=("income", "size"), income=("income", "mean"),
                                     loan_amount=("loan_amount", "mean"),
                                     credit_score=("credit_score", "mean"),
                                     default_rate=("default", "mean")).round(2)
log("Segment profile (original units):\n" + summary.to_string())
overall = km_data.mean()
log("Segment description:")
for c, row in summary.iterrows():
    tags = []
    for col in km_cols:
        tags.append(f"{'high' if row[col] > overall[col] * 1.1 else 'low' if row[col] < overall[col] * 0.9 else 'average'} {col}")
    log(f"  Cluster {c}: {', '.join(tags)}; default rate {row['default_rate']:.1%} ({int(row['customers'])} customers)")
RESULTS.append({"Model": f"K-Means (k={best_k})", "Task": "Clustering",
                "Validation score": f"Silhouette {max(sils):.3f}"})

# =============================================================================
header("TASK 4 - Evaluation and tuning")
# =============================================================================
log("Metrics on the VALIDATION set (threshold 0.5):")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
rows = []
for ax, (n, m) in zip(axes, models.items()):
    pred = m.predict(X_val)
    rows.append({"Model": n, "Accuracy": accuracy_score(y_val, pred),
                 "Precision": precision_score(y_val, pred), "Recall": recall_score(y_val, pred),
                 "F1": f1_score(y_val, pred), "ROC-AUC": roc_auc_score(y_val, val_proba[n])})
    log(f"\n{n}  confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_val, pred)}")
    ConfusionMatrixDisplay.from_predictions(y_val, pred, ax=ax, colorbar=False)
    ax.set_title(n)
plt.tight_layout(); plt.savefig("outputs/task4_confusion_matrices.png", dpi=130); plt.close()
metrics = pd.DataFrame(rows).set_index("Model").round(3)
log("\n" + metrics.to_string())

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
d_pred = dummy.predict(X_val)
log("\nWhy accuracy misleads with imbalanced classes:")
log(f"  Only {y_val.mean():.0%} of customers default. A 'model' that ALWAYS predicts 'no default' gets "
    f"accuracy {accuracy_score(y_val, d_pred):.2f}")
log(f"  but recall {recall_score(y_val, d_pred):.2f} - it never finds a single defaulter. "
    "So we also look at precision, recall, F1 and ROC-AUC.")

# --- Hyperparameter tuning (on TRAIN only, CV inside) ----------------------------
log("\nGridSearchCV - RandomForest (3 parameters, 3-fold CV, scoring ROC-AUC):")
rf_grid = GridSearchCV(
    U.make_pipeline(RandomForestClassifier(min_samples_leaf=3, n_jobs=-1, random_state=RS)),
    {"model__n_estimators": [200, 400], "model__max_depth": [6, 12, None],
     "model__min_samples_leaf": [1, 5, 10]},
    cv=3, scoring="roc_auc", n_jobs=1).fit(X_train, y_train)
log(f"  Best params: {rf_grid.best_params_}")
log(f"  Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

log("\nRandomizedSearchCV - XGBoost (n_iter=15, 3-fold CV, scoring ROC-AUC):")
xgb_search = RandomizedSearchCV(
    U.make_pipeline(XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=1)),
    {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [2, 3, 4, 5, 6],
     "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
     "model__subsample": [0.6, 0.8, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0],
     "model__min_child_weight": [1, 3, 5]},
    n_iter=15, cv=3, scoring="roc_auc", random_state=RS, n_jobs=1).fit(X_train, y_train)
log(f"  Best params: {xgb_search.best_params_}")
log(f"  Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

candidates = dict(models)
candidates["RandomForest (tuned)"] = rf_grid.best_estimator_
candidates["XGBoost (tuned)"] = xgb_search.best_estimator_
cand_auc = {}
for n, m in candidates.items():
    cand_auc[n] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
log("\nValidation ROC-AUC of all candidates:")
for n, a in sorted(cand_auc.items(), key=lambda kv: -kv[1]):
    log(f"  {n:<22} {a:.4f}")
    if "tuned" in n:
        RESULTS.append({"Model": n, "Task": "Classification",
                        "Validation score": f"ROC-AUC {a:.3f}"})
best_name = max(cand_auc, key=cand_auc.get)
best_model = candidates[best_name]
p_val = best_model.predict_proba(X_val)[:, 1]
best_pred = (p_val >= 0.5).astype(int)
log(f"\nBest model: {best_name} | ROC-AUC {cand_auc[best_name]:.2f} | F1 {f1_score(y_val, best_pred):.2f}")

# --- ROC curves of 3 models in one chart -------------------------------------------
plt.figure(figsize=(6.5, 5.5))
for n in ["LogisticRegression", "RandomForest (tuned)", "XGBoost (tuned)"]:
    fpr, tpr, _ = roc_curve(y_val, candidates[n].predict_proba(X_val)[:, 1])
    plt.plot(fpr, tpr, label=f"{n} (AUC = {cand_auc[n]:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves (validation set)"); plt.legend(loc="lower right"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/task4_roc_curves.png", dpi=150); plt.close()

# --- Threshold with recall >= 75% -----------------------------------------------------
prec, rec, thr = precision_recall_curve(y_val, p_val)
ok = np.where(rec[:-1] >= 0.75)[0]
t_idx = ok[-1]                       # highest threshold that still keeps recall >= 75%
threshold = float(thr[t_idx])
pred_t = (p_val >= threshold).astype(int)
log(f"\nThreshold for >= 75% recall: {threshold:.3f}")
log(f"  At this threshold: recall {recall_score(y_val, pred_t):.3f} | precision "
    f"{precision_score(y_val, pred_t):.3f} | F1 {f1_score(y_val, pred_t):.3f}")
log(f"  (at 0.5: recall {recall_score(y_val, best_pred):.3f} | precision {precision_score(y_val, best_pred):.3f})")
log("  Lower threshold = catch more defaulters (higher recall) but more false alarms (lower precision).")
plt.figure(figsize=(6, 4.5))
plt.plot(rec[:-1], prec[:-1]); plt.axvline(0.75, color="r", ls="--", label="recall = 0.75")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curve")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("outputs/task4_precision_recall.png", dpi=150); plt.close()

# --- Regression metrics vs mean baseline ---------------------------------------------------
header("Regression metrics (validation) vs mean baseline")
best_reg_name = max(reg_val_r2, key=reg_val_r2.get)
best_reg = reg_models[best_reg_name]
base = DummyRegressor(strategy="mean").fit(Xr_train, yr_train)
reg_rows = []
for n, m in [("Mean baseline", base), (best_reg_name, best_reg)]:
    p = m.predict(Xr_val)
    reg_rows.append({"Model": n, "MAE": mean_absolute_error(yr_val, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yr_val, p))), "R2": r2_score(yr_val, p)})
log(pd.DataFrame(reg_rows).set_index("Model").round(3).to_string())
log("(loan_amount is in Rs. thousands in the synthetic data)")

# --- Final evaluation: single best model, ONCE on the test set ---------------------------------
header("FINAL TEST-SET EVALUATION (best model, evaluated once)")
p_test = best_model.predict_proba(X_test)[:, 1]
pred_test_05 = (p_test >= 0.5).astype(int)
pred_test_thr = (p_test >= threshold).astype(int)
log(f"Best model: {best_name}")
log(f"Test ROC-AUC: {roc_auc_score(y_test, p_test):.3f}")
log(f"At 0.5       -> acc {accuracy_score(y_test, pred_test_05):.3f} | precision "
    f"{precision_score(y_test, pred_test_05):.3f} | recall {recall_score(y_test, pred_test_05):.3f} | "
    f"F1 {f1_score(y_test, pred_test_05):.3f}")
log(f"At {threshold:.3f}   -> acc {accuracy_score(y_test, pred_test_thr):.3f} | precision "
    f"{precision_score(y_test, pred_test_thr):.3f} | recall {recall_score(y_test, pred_test_thr):.3f} | "
    f"F1 {f1_score(y_test, pred_test_thr):.3f}  (threshold chosen on validation)")
log(f"Confusion matrix at tuned threshold:\n{confusion_matrix(y_test, pred_test_thr)}")
joblib.dump({"pipeline": best_model, "threshold": threshold}, "models/best_model.joblib")
log("Saved: models/best_model.joblib (pipeline + decision threshold)")

# --- Results table ---------------------------------------------------------------------------------
res = pd.DataFrame(RESULTS)
res.to_csv("outputs/results_table.csv", index=False)
header("RESULTS TABLE: Model | Validation score")
log(res.to_markdown(index=False) if hasattr(res, "to_markdown") else res.to_string(index=False))
with open("outputs/report.md", "w") as f:
    f.write("# Auto-generated run report\n\n```\n" + "\n".join(REPORT) + "\n```\n")
print("\nDone. Plots + report are in ./outputs, models in ./models")
