"""Model training with stratified K-fold CV. Every model returns out-of-fold (OOF) predictions on the
training split and the fold-averaged prediction on the untouched holdout split."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    holdout: np.ndarray
    fold_auc: list[float]
    models: list = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self, y: np.ndarray) -> dict:
        return {
            "model": self.name,
            "cv_auc_mean": float(np.mean(self.fold_auc)),
            "cv_auc_std": float(np.std(self.fold_auc)),
            "oof_auc": float(roc_auc_score(y, self.oof)),
            "minutes": round(self.seconds / 60, 2),
        }


def has_gpu() -> bool:
    try:
        import subprocess

        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _folds(y, n_folds, seed):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


# ---------------------------------------------------------------------------------- baselines
def logistic_pipeline(X: pd.DataFrame, C: float = 0.1) -> Pipeline:
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                              ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50))]), cat),
        ]
    )
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=2000))])


def _as_object_cats(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(object).where(X[c].notna(), "MISSING")
    return X


def cv_logistic(X, y, X_ho, n_folds=5, seed=42, C=0.1) -> CVResult:
    t0 = time.time()
    Xo, Xho = _as_object_cats(X), _as_object_cats(X_ho)
    oof, ho, aucs, models = np.zeros(len(y)), np.zeros(len(X_ho)), [], []
    for k, (tr, va) in enumerate(_folds(y, n_folds, seed)):
        m = logistic_pipeline(Xo, C).fit(Xo.iloc[tr], y[tr])
        oof[va] = m.predict_proba(Xo.iloc[va])[:, 1]
        ho += m.predict_proba(Xho)[:, 1] / n_folds
        aucs.append(roc_auc_score(y[va], oof[va]))
        models.append(m)
        print(f"  [logreg] fold {k}: AUC={aucs[-1]:.4f}")
    return CVResult("LogisticRegression", oof, ho, aucs, models, seconds=time.time() - t0)


# ---------------------------------------------------------------------------------- LightGBM
LGB_DEFAULTS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    feature_fraction=0.4,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=255,
    verbose=-1,
    n_jobs=-1,
)


def cv_lightgbm(X, y, X_ho, params=None, n_folds=5, seed=42, num_boost_round=10000, early_stopping=200,
                name="LightGBM", verbose=True) -> CVResult:
    t0 = time.time()
    p = {**LGB_DEFAULTS, **(params or {}), "seed": seed}
    oof, ho, aucs, models, iters = np.zeros(len(y)), np.zeros(len(X_ho)), [], [], []
    for k, (tr, va) in enumerate(_folds(y, n_folds, seed)):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        booster = lgb.train(p, dtr, num_boost_round, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        ho += booster.predict(X_ho, num_iteration=booster.best_iteration) / n_folds
        aucs.append(roc_auc_score(y[va], oof[va]))
        models.append(booster)
        iters.append(booster.best_iteration)
        if verbose:
            print(f"  [{name}] fold {k}: AUC={aucs[-1]:.4f}  best_iter={booster.best_iteration}")
    return CVResult(name, oof, ho, aucs, models, iters, time.time() - t0)


# ---------------------------------------------------------------------------------- XGBoost
def cv_xgboost(X, y, X_ho, params=None, n_folds=5, seed=42, num_boost_round=10000, early_stopping=200) -> CVResult:
    import xgboost as xgb

    t0 = time.time()
    p = dict(objective="binary:logistic", eval_metric="auc", eta=0.03, max_depth=6, min_child_weight=30,
             subsample=0.8, colsample_bytree=0.4, reg_lambda=5.0, tree_method="hist", max_cat_to_onehot=1,
             device="cuda" if has_gpu() else "cpu", seed=seed)
    p.update(params or {})
    dho = xgb.DMatrix(X_ho, enable_categorical=True)
    oof, ho, aucs, models, iters = np.zeros(len(y)), np.zeros(len(X_ho)), [], [], []
    for k, (tr, va) in enumerate(_folds(y, n_folds, seed)):
        dtr = xgb.DMatrix(X.iloc[tr], y[tr], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va], y[va], enable_categorical=True)
        bst = xgb.train(p, dtr, num_boost_round, evals=[(dva, "valid")], early_stopping_rounds=early_stopping,
                        verbose_eval=False)
        rng = (0, bst.best_iteration + 1)
        oof[va] = bst.predict(dva, iteration_range=rng)
        ho += bst.predict(dho, iteration_range=rng) / n_folds
        aucs.append(roc_auc_score(y[va], oof[va]))
        models.append(bst)
        iters.append(bst.best_iteration)
        print(f"  [xgboost] fold {k}: AUC={aucs[-1]:.4f}  best_iter={bst.best_iteration}")
    return CVResult("XGBoost", oof, ho, aucs, models, iters, time.time() - t0)


# ---------------------------------------------------------------------------------- CatBoost
def cv_catboost(X, y, X_ho, params=None, n_folds=5, seed=42, iterations=10000, early_stopping=200) -> CVResult:
    from catboost import CatBoostClassifier, Pool

    t0 = time.time()
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]

    def prep(D):
        D = D.copy()
        for c in cat:
            D[c] = D[c].astype(object).where(D[c].notna(), "MISSING").astype(str)
        return D

    Xc, Xhoc = prep(X), prep(X_ho)
    gpu = has_gpu()
    p = dict(iterations=iterations, learning_rate=0.05, depth=6, l2_leaf_reg=5, eval_metric="AUC",
             od_type="Iter", od_wait=early_stopping, random_seed=seed, verbose=0,
             task_type="GPU" if gpu else "CPU")
    p.update(params or {})
    oof, ho, aucs, models, iters = np.zeros(len(y)), np.zeros(len(X_ho)), [], [], []
    for k, (tr, va) in enumerate(_folds(y, n_folds, seed)):
        m = CatBoostClassifier(**p)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat), eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat),
              use_best_model=True)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        ho += m.predict_proba(Xhoc)[:, 1] / n_folds
        aucs.append(roc_auc_score(y[va], oof[va]))
        models.append(m)
        iters.append(m.get_best_iteration())
        print(f"  [catboost] fold {k}: AUC={aucs[-1]:.4f}  best_iter={m.get_best_iteration()}")
    return CVResult("CatBoost", oof, ho, aucs, models, iters, time.time() - t0)


# ---------------------------------------------------------------------------------- tuning
def tune_lightgbm(X, y, n_trials=25, timeout_minutes=40, sample_frac=0.5, n_folds=3, seed=42):
    """Optuna TPE search on a stratified subsample with a high learning rate (fast proxy objective).
    The final model re-uses the found tree-shape/regularisation params with a smaller learning rate."""
    import optuna

    rng = np.random.RandomState(seed)
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], int((y == c).sum() * sample_frac), replace=False) for c in (0, 1)
    ])
    Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]

    def objective(trial):
        params = dict(
            learning_rate=0.08,
            num_leaves=trial.suggest_int("num_leaves", 8, 96, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.15, 0.8),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 20, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 50, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            max_depth=trial.suggest_int("max_depth", 4, 12),
        )
        res = cv_lightgbm(Xs, ys, Xs.iloc[:1], params, n_folds=n_folds, seed=seed, early_stopping=100,
                          name=f"trial{trial.number}", verbose=False)
        return float(np.mean(res.fold_auc))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: v for k, v in LGB_DEFAULTS.items() if k in
                         ("num_leaves", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l2")}
                        | {"lambda_l1": 1e-3, "min_split_gain": 0.0, "max_depth": 8})
    study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"  trial {t.number:3d}: AUC={t.value:.4f}  best={s.best_value:.4f}")])
    return study


# ---------------------------------------------------------------------------------- blending
def blend_weights(oofs: dict[str, np.ndarray], y: np.ndarray, step: float = 0.05):
    """Grid search over convex weights of rank-normalised predictions (maximises OOF AUC).
    Uses only OOF predictions, so the holdout stays clean."""
    from itertools import product
    from scipy.stats import rankdata

    names = list(oofs)
    R = np.column_stack([rankdata(oofs[n]) / len(y) for n in names])
    grid = np.arange(0, 1 + 1e-9, step)
    best = (-1, None)
    for w in product(grid, repeat=len(names) - 1):
        last = 1 - sum(w)
        if last < -1e-9:
            continue
        wv = np.array([*w, last])
        auc = roc_auc_score(y, R @ wv)
        if auc > best[0]:
            best = (auc, wv)
    return dict(zip(names, best[1].round(3))), best[0]


def apply_blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    from scipy.stats import rankdata

    return sum(weights[n] * rankdata(preds[n]) / len(preds[n]) for n in weights)
