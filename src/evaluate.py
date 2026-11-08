"""Evaluation harness for the MK8 build-prediction models.

Runs repeated K-fold CV (since LOOCV on 45-52 rows is noisy per-fold) plus a
pooled leave-one-track-out pass for a genuine single-track holdout demo.
Reports per-target R^2 and MAE (mean +/- std) against a DummyRegressor(mean)
baseline. Designed to be called from the notebook, not just as a script.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_SEED = 42
TARGET_COLS = ["CBC_Speed", "CBC_Acceleration", "CBC_Handling", "CBC_Mini Turbo", "CBC_Traction"]


def get_models():
    """Fresh model instances -- call per CV split, sklearn models carry state."""
    return {
        "Baseline (mean)": MultiOutputRegressor(DummyRegressor(strategy="mean")),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(random_state=RANDOM_SEED)),
        "SVR (rbf)": MultiOutputRegressor(SVR(kernel="rbf")),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=RANDOM_SEED)
        ),
    }


def repeated_kfold_cv(X, y, n_splits=5, n_repeats=10, random_state=RANDOM_SEED):
    """Repeated K-fold CV. Returns a tidy DataFrame: model x target x fold metrics."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for model_name in get_models():
        for fold_i, (train_idx, test_idx) in enumerate(rkf.split(X)):
            model = get_models()[model_name]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for t_i, target in enumerate(TARGET_COLS):
                y_true_t = y_test[target].values
                y_pred_t = y_pred[:, t_i]
                mae = mean_absolute_error(y_true_t, y_pred_t)
                # R2 is unstable on tiny folds (can be 1 row); guard it.
                r2 = r2_score(y_true_t, y_pred_t) if len(y_true_t) > 1 else np.nan
                rows.append({
                    "model": model_name, "target": target, "fold": fold_i,
                    "mae": mae, "r2": r2,
                })
    return pd.DataFrame(rows)


def summarize_cv(cv_df):
    """Mean +/- std of MAE/R2 per model x target."""
    summary = cv_df.groupby(["model", "target"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    ).reset_index()
    return summary


def pooled_loto(X, y, track_names):
    """Leave-one-track-out, pooled out-of-fold predictions -> one R2/MAE per
    model x target computed over all 52 (or 45) held-out predictions at once,
    not averaged per-fold (folds of size 1 can't have a meaningful R2).
    """
    loo = LeaveOneOut()
    preds = {name: np.zeros((len(X), len(TARGET_COLS))) for name in get_models()}

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        for name in get_models():
            model = get_models()[name]
            model.fit(X_train, y_train)
            preds[name][test_idx[0], :] = model.predict(X_test)[0]

    rows = []
    for name, pred_arr in preds.items():
        for t_i, target in enumerate(TARGET_COLS):
            y_true_t = y[target].values
            y_pred_t = pred_arr[:, t_i]
            rows.append({
                "model": name, "target": target,
                "mae": mean_absolute_error(y_true_t, y_pred_t),
                "r2": r2_score(y_true_t, y_pred_t),
            })
    pred_df = pd.DataFrame({"Track Name": track_names})
    for name, pred_arr in preds.items():
        for t_i, target in enumerate(TARGET_COLS):
            pred_df[f"{name}_{target}_pred"] = pred_arr[:, t_i]
    return pd.DataFrame(rows), pred_df
