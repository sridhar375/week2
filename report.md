# Auto-generated run report

```
Dataset: 6000 rows x 12 columns
Default rate: 0.134
Missing values per column:
income              364
employment_years    229
credit_score        308
education           240
home_ownership      185

==============================================================================
TASK 1 - Concepts, splits and leakage
==============================================================================
Train 3600 | Validation 1200 | Test 1200  (test set put aside)
Default rate  train 0.134 | val 0.134 | test 0.134

Overfitting demo (DecisionTreeClassifier on numeric columns):
  Depth    1: train 0.87 val 0.87
  Depth    3: train 0.87 val 0.87
  Depth    5: train 0.88 val 0.87
  Depth   10: train 0.93 val 0.84
  Depth None: train 1.00 val 0.80

CV (LogisticRegression, 5-fold): 0.87 +/- 0.01

Data leakage demo (scaling):
  income mean used for scaling -> leaky (all data): 624.21 | correct (train only): 620.56
  Validation accuracy -> leaky: 0.8683 | correct: 0.8683
  Explanation (4 lines):
  1) Scaling before the split lets mean/std be computed from validation and test rows too.
  2) The model therefore 'sees' information about unseen data, so scores look slightly optimistic.
  3) The correct way: split first, fit the scaler on train only, then only transform val/test.
  4) Putting the scaler inside a Pipeline does this automatically in every CV fold.

Classify the 4 problems:
  Spam detection         -> SUPERVISED (classification, labelled spam / not spam)
  Customer segmentation  -> UNSUPERVISED (clustering, no labels)
  House price prediction -> SUPERVISED (regression, numeric target)
  Anomaly detection      -> UNSUPERVISED (mostly; no/rare labels, finds unusual points)

ML workflow in 7 steps:
  1. Define the problem and the success metric (e.g. recall / ROC-AUC for default).
  2. Collect the data and explore it (EDA, missing values, class balance).
  3. Split into train / validation / test BEFORE any fitting; keep test aside.
  4. Preprocess and engineer features inside a Pipeline (impute, encode, scale).
  5. Train several models and compare them on the validation set / cross-validation.
  6. Tune hyperparameters, choose the threshold, avoid overfitting and leakage.
  7. Evaluate the single best model once on the test set, save it, deploy and monitor.

==============================================================================
TASK 2 - Preprocessing, features and pipeline
==============================================================================
Imputation comparison (5-fold CV accuracy, whole pipeline):
  SimpleImputer(median) : 0.8658
  KNNImputer(k=5)       : 0.8647
Encoding: OneHotEncoder for home_ownership + loan_purpose, OrdinalEncoder for education
          (High School < Bachelors < Masters < PhD)
New features: app_year / app_month / app_dayofweek (date), income_to_loan (interaction), log_income (log)

Pipeline CV accuracy: 0.87 (+/- 0.01)
Predictions for 3 new rows: [0 1 0]   (default probabilities: [0.006 0.903 0.106])
Saved: models/loan_pipeline.joblib (raw rows go in; imputing/encoding/scaling happen inside)

==============================================================================
TASK 3 - Models: supervised and unsupervised
==============================================================================
RandomForest OOB score: 0.8650
XGBoost stopped at best iteration 65

LogisticRegression predict_proba vs predict (5 validation rows):
   P(no default)  P(default)  predict (thr 0.5)  actual
0          0.972       0.028                  0       0
1          0.950       0.050                  0       0
2          0.838       0.162                  0       0
3          0.972       0.028                  0       0
4          0.963       0.037                  0       0
predict() is just 'P(default) >= 0.5'; predict_proba lets us move the threshold later (Task 4).

Top 5 features - RandomForest:
credit_score        0.2234
debt_to_income      0.1178
income_to_loan      0.0878
loan_amount         0.0763
employment_years    0.0694

Top 5 features - XGBoost:
credit_score              0.1604
debt_to_income            0.0832
loan_purpose_Education    0.0469
loan_purpose_Medical      0.0469
education                 0.0462

==============================================================================
Regression: predict loan_amount (Linear / Ridge / Lasso)
==============================================================================
Coefficients (features are standardised, so they are comparable):
                         LinearRegression  Ridge(alpha=10)  Lasso(alpha=3)
age                                  9.28             9.80            7.31
income                             162.01           159.25          161.31
employment_years                    47.78            47.16           46.49
credit_score                        23.18            23.13           20.22
num_credit_lines                     0.49             0.45            0.00
debt_to_income                      -1.19            -1.20           -0.00
app_year                            -0.53            -0.51           -0.00
app_month                            1.00             1.02            0.00
app_dayofweek                        0.33             0.32            0.00
log_income                           2.75             5.28            0.60
education                           12.60            12.61            9.15
home_ownership_MORTGAGE             -1.88            -1.82           -0.00
home_ownership_OWN                  26.02            25.80           17.08
home_ownership_RENT                -24.15           -23.98          -15.07
loan_purpose_Business               33.12            32.55           28.84
loan_purpose_Education             -12.85           -12.77            0.00
loan_purpose_Home                   77.05            76.12           78.59
loan_purpose_Medical               -54.16           -53.01          -14.69
loan_purpose_Personal              -43.16           -42.88          -20.83

Lasso zero coefficients: 7 of 19  -> Lasso removes weak features
Ridge shrinks coefficients towards 0 but keeps all of them; Lasso can set them exactly to 0.
Validation R2: LinearRegression 0.861 | Ridge(alpha=10) 0.861 | Lasso(alpha=3) 0.856

==============================================================================
K-Means customer segmentation (income, loan_amount, credit_score)
==============================================================================
  k=2: inertia 6797.5 | silhouette 0.409
  k=3: inertia 5021.6 | silhouette 0.300
  k=4: inertia 4070.6 | silhouette 0.305
  k=5: inertia 3426.8 | silhouette 0.293
  k=6: inertia 3001.8 | silhouette 0.291
  k=7: inertia 2650.0 | silhouette 0.291
  k=8: inertia 2350.1 | silhouette 0.288

Best k by silhouette: 2
Segment profile (original units):
         customers   income  loan_amount  credit_score  default_rate
cluster                                                             
0              865  1111.74       692.76        689.47          0.11
1             2735   465.21       359.15        679.61          0.14
Segment description:
  Cluster 0: high income, high loan_amount, average credit_score; default rate 11.0% (865 customers)
  Cluster 1: low income, low loan_amount, average credit_score; default rate 14.0% (2735 customers)

==============================================================================
TASK 4 - Evaluation and tuning
==============================================================================
Metrics on the VALIDATION set (threshold 0.5):

LogisticRegression  confusion matrix [[TN FP] [FN TP]]:
[[1026   13]
 [ 144   17]]

DecisionTree  confusion matrix [[TN FP] [FN TP]]:
[[1016   23]
 [ 138   23]]

RandomForest  confusion matrix [[TN FP] [FN TP]]:
[[1032    7]
 [ 154    7]]

XGBoost  confusion matrix [[TN FP] [FN TP]]:
[[1030    9]
 [ 149   12]]

                    Accuracy  Precision  Recall     F1  ROC-AUC
Model                                                          
LogisticRegression     0.869      0.567   0.106  0.178    0.776
DecisionTree           0.866      0.500   0.143  0.222    0.738
RandomForest           0.866      0.500   0.043  0.080    0.761
XGBoost                0.868      0.571   0.075  0.132    0.774

Why accuracy misleads with imbalanced classes:
  Only 13% of customers default. A 'model' that ALWAYS predicts 'no default' gets accuracy 0.87
  but recall 0.00 - it never finds a single defaulter. So we also look at precision, recall, F1 and ROC-AUC.

GridSearchCV - RandomForest (3 parameters, 3-fold CV, scoring ROC-AUC):
  Best params: {'model__max_depth': 6, 'model__min_samples_leaf': 1, 'model__n_estimators': 400}
  Best CV ROC-AUC: 0.7470

RandomizedSearchCV - XGBoost (n_iter=15, 3-fold CV, scoring ROC-AUC):
  Best params: {'model__subsample': 0.6, 'model__n_estimators': 200, 'model__min_child_weight': 3, 'model__max_depth': 2, 'model__learning_rate': 0.03, 'model__colsample_bytree': 0.8}
  Best CV ROC-AUC: 0.7597

Validation ROC-AUC of all candidates:
  XGBoost (tuned)        0.7780
  LogisticRegression     0.7758
  XGBoost                0.7736
  RandomForest (tuned)   0.7642
  RandomForest           0.7609
  DecisionTree           0.7380

Best model: XGBoost (tuned) | ROC-AUC 0.78 | F1 0.15

Threshold for >= 75% recall: 0.125
  At this threshold: recall 0.752 | precision 0.262 | F1 0.388
  (at 0.5: recall 0.087 | precision 0.583)
  Lower threshold = catch more defaulters (higher recall) but more false alarms (lower precision).

==============================================================================
Regression metrics (validation) vs mean baseline
==============================================================================
                     MAE     RMSE     R2
Model                                   
Mean baseline    152.128  210.625 -0.004
Ridge(alpha=10)   42.150   78.412  0.861
(loan_amount is in Rs. thousands in the synthetic data)

==============================================================================
FINAL TEST-SET EVALUATION (best model, evaluated once)
==============================================================================
Best model: XGBoost (tuned)
Test ROC-AUC: 0.794
At 0.5       -> acc 0.871 | precision 0.636 | recall 0.087 | F1 0.153
At 0.125   -> acc 0.688 | precision 0.261 | recall 0.720 | F1 0.383  (threshold chosen on validation)
Confusion matrix at tuned threshold:
[[710 329]
 [ 45 116]]
Saved: models/best_model.joblib (pipeline + decision threshold)

==============================================================================
RESULTS TABLE: Model | Validation score
==============================================================================
| Model                | Task                     | Validation score          |
|:---------------------|:-------------------------|:--------------------------|
| LogisticRegression   | Classification           | ROC-AUC 0.776 | Acc 0.869 |
| DecisionTree         | Classification           | ROC-AUC 0.738 | Acc 0.866 |
| RandomForest         | Classification           | ROC-AUC 0.761 | Acc 0.866 |
| XGBoost              | Classification           | ROC-AUC 0.774 | Acc 0.868 |
| LinearRegression     | Regression (loan_amount) | R2 0.861                  |
| Ridge(alpha=10)      | Regression (loan_amount) | R2 0.861                  |
| Lasso(alpha=3)       | Regression (loan_amount) | R2 0.856                  |
| K-Means (k=2)        | Clustering               | Silhouette 0.409          |
| XGBoost (tuned)      | Classification           | ROC-AUC 0.778             |
| RandomForest (tuned) | Classification           | ROC-AUC 0.764             |
```
