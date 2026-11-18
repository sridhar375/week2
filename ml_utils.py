"""Helper module for the Week 2 ML assessment (loan default project).

Kept in a separate importable module so that the saved joblib pipeline can be
loaded later (joblib needs to import FeatureEngineer by name).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Column configuration -- edit these if the trainer's dataset has other names
# ----------------------------------------------------------------------------
TARGET = "default"
DATE_COL = "application_date"
NUMERIC_BASE = ["age", "income", "employment_years", "credit_score",
                "num_credit_lines", "debt_to_income", "loan_amount"]
FE_NUMERIC = ["app_year", "app_month", "app_dayofweek",   # date decomposition
              "income_to_loan", "log_income"]             # interaction + log
ORDINAL_COL = "education"
EDU_ORDER = ["High School", "Bachelors", "Masters", "PhD"]
NOMINAL_COLS = ["home_ownership", "loan_purpose"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_synthetic_loan_data(n=6000, seed=42):
    """Create a realistic loan dataset (income and loan_amount are in Rs. thousands)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38, 10, n), 21, 70).round()
    income = rng.lognormal(mean=6.3, sigma=0.55, size=n).round(1)           # skewed
    employment_years = np.clip(age - 21 - rng.exponential(4, n), 0, 40).round(1)
    credit_score = np.clip(rng.normal(680, 70, n), 300, 850).round()
    num_credit_lines = rng.poisson(5, n)
    debt_to_income = np.clip(rng.normal(0.30, 0.12, n), 0.02, 0.9).round(3)
    education = rng.choice(EDU_ORDER, n, p=[0.30, 0.40, 0.22, 0.08])
    home = rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.40, 0.25, 0.35])
    purpose = rng.choice(["Personal", "Home", "Education", "Business", "Medical"],
                         n, p=[0.30, 0.25, 0.15, 0.20, 0.10])
    start = pd.Timestamp("2022-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")

    edu_rank = pd.Series(education).map({e: i for i, e in enumerate(EDU_ORDER)}).to_numpy()
    purpose_eff = pd.Series(purpose).map({"Personal": 0, "Home": 120, "Education": 30,
                                          "Business": 80, "Medical": -10}).to_numpy()
    home_eff = pd.Series(home).map({"RENT": -20, "OWN": 30, "MORTGAGE": 0}).to_numpy()
    loan_amount = (0.45 * income + 6 * employment_years + 0.35 * (credit_score - 650)
                   + 15 * edu_rank + purpose_eff + home_eff
                   + rng.normal(0, 40, n))
    loan_amount = np.clip(loan_amount, 20, None).round(1)

    z = (-2.55 + 5.0 * (debt_to_income - 0.30) - 0.017 * (credit_score - 680)
         + 0.9 * (loan_amount / income - 0.5) + 0.35 * (home == "RENT")
         + 0.30 * (purpose == "Business") - 0.12 * edu_rank
         - 0.02 * employment_years + rng.normal(0, 0.4, n))
    p = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, p)

    df = pd.DataFrame({
        "age": age, "income": income, "employment_years": employment_years,
        "credit_score": credit_score, "num_credit_lines": num_credit_lines,
        "debt_to_income": debt_to_income, "loan_amount": loan_amount,
        "education": education, "home_ownership": home, "loan_purpose": purpose,
        "application_date": dates.strftime("%Y-%m-%d"), TARGET: default,
    })
    # inject realistic missing values
    for col, frac in [("income", 0.06), ("credit_score", 0.05),
                      ("employment_years", 0.04), ("education", 0.04),
                      ("home_ownership", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def load_data(path=None):
    """Load the trainer's CSV if a path is given, otherwise build the synthetic one."""
    if path:
        return pd.read_csv(path)
    return make_synthetic_loan_data()


# ----------------------------------------------------------------------------
# Feature engineering (lives INSIDE the pipeline -> no leakage, saved with joblib)
# ----------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds 5 new features: year/month/weekday from the date, income/loan_amount, log(income)."""

    def __init__(self, date_col=DATE_COL, add_ratio=True):
        self.date_col = date_col
        self.add_ratio = add_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        d = pd.to_datetime(X[self.date_col], errors="coerce")
        X["app_year"] = d.dt.year
        X["app_month"] = d.dt.month
        X["app_dayofweek"] = d.dt.dayofweek
        if self.add_ratio:                                   # interaction feature
            X["income_to_loan"] = X["income"] / X["loan_amount"].replace(0, np.nan)
        X["log_income"] = np.log1p(X["income"])              # log of a skewed column
        return X.drop(columns=[self.date_col])


def make_preprocessor(task="clf", numeric_imputer="median"):
    num_cols = NUMERIC_BASE + FE_NUMERIC
    if task == "reg":   # loan_amount is the target there, so ratio feature must go too
        num_cols = [c for c in num_cols if c not in ("loan_amount", "income_to_loan")]

    if numeric_imputer == "median":
        num_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    else:  # KNN works on distances, so scale first
        num_steps = [("scaler", StandardScaler()), ("imputer", KNNImputer(n_neighbors=5))]

    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[EDU_ORDER], handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])
    nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("ord", ordinal, [ORDINAL_COL]),
        ("nom", nominal, NOMINAL_COLS),
    ], sparse_threshold=0)


def make_pipeline(model, task="clf", numeric_imputer="median"):
    """FeatureEngineer -> ColumnTransformer -> model, all in one Pipeline."""
    return Pipeline([
        ("fe", FeatureEngineer(add_ratio=(task == "clf"))),
        ("prep", make_preprocessor(task, numeric_imputer)),
        ("model", model),
    ])


def feature_names(pipe):
    """Readable feature names after preprocessing."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]
