import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

# 1) Load data
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

X = train.drop(columns=["id", "y"])
y = train["y"].astype(int)
X_test = test.drop(columns=["id"])

# 2) Identify categorical columns and cast
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# 3) Best params from Optuna
best_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "learning_rate": 0.010077234222694983,
    "num_leaves": 254,
    "max_depth": 16,
    "min_child_samples": 24,
    "subsample": 0.977803155400075,
    "colsample_bytree": 0.6005341109936488,
    "reg_alpha": 4.407277688987944,
    "reg_lambda": 4.365598988101468,
    "scale_pos_weight": 6.013651543136506,
    "n_estimators": 3000,   # rely on early stopping
    "n_jobs": -1,
    "random_state": 42
}

# 4) Stratified 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_aucs = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(100)]
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    oof_pred[val_idx] = val_proba
    test_pred += model.predict_proba(X_test)[:, 1] / kf.n_splits

    fold_auc = roc_auc_score(y_val, val_proba)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold}: AUC = {fold_auc:.6f}")

# 5) Out-of-fold AUC
oof_auc = roc_auc_score(y, oof_pred)
print(f"\nOOF AUC: {oof_auc:.6f}")
print("Fold AUCs:", [f"{a:.6f}" for a in fold_aucs])
print(f"Mean/Std AUC: {np.mean(fold_aucs):.6f} / {np.std(fold_aucs):.6f}")

# 6) Save submission
submission = pd.DataFrame({"id": test["id"], "y": test_pred})
submission.to_csv("submission.csv", index=False, float_format="%.6f")
print("✅ submission.csv saved with probabilities")
