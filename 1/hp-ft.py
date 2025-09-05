import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# Load
train = pd.read_csv("train.csv")
X = train.drop(columns=["id", "y"])
y = train["y"].astype(int)

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

# Stratified CV setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": 3000,  # rely on early stopping
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 6.0, 8.0),
        "n_jobs": -1,
        "random_state": 42,
    }

    oof_pred = np.zeros(len(X))
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = val_proba

    return roc_auc_score(y, oof_pred)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)   # adjust n_trials upward if you have compute budget

print("Best params:", study.best_params)
print("Best CV AUC:", study.best_value)
