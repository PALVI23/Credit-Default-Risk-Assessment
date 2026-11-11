"""Group fairness audit and post-processing mitigation for an approve/reject decision.

Conventions: y=1 means the applicant defaulted. A decision of "approve" means PD < threshold.
  * approval_rate         - P(approve | group)                       -> demographic parity
  * good_reject_rate      - P(reject | repaid, group)   (a 'false positive' for the lender's
                            default detector: a creditworthy person who is denied credit)
  * bad_approve_rate      - P(approve | defaulted, group)
Equal opportunity for creditworthy applicants = equal good_reject_rate across groups.
Equalized odds = equal good_reject_rate AND equal bad_approve_rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import expected_calibration_error

AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def age_group(age_years) -> pd.Series:
    return pd.cut(pd.Series(age_years), AGE_BINS, labels=AGE_LABELS, right=False).astype(str)


def group_report(y, pd_hat, approve, group, credit=None, margin=0.10, lgd=0.50) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat), "approve": np.asarray(approve).astype(bool),
                       "g": np.asarray(group)})
    if credit is not None:
        df["pnl"] = np.where(df.y == 0, margin, -lgd) * np.asarray(credit, dtype=float) * df.approve
    rows = []
    for g, d in df.groupby("g"):
        good, bad = d[d.y == 0], d[d.y == 1]
        row = {
            "group": g,
            "n": len(d),
            "share": len(d) / len(df),
            "default_rate": d.y.mean(),
            "mean_pd": d.p.mean(),
            "auc": roc_auc_score(d.y, d.p) if d.y.nunique() == 2 else np.nan,
            "ece": expected_calibration_error(d.y, d.p, n_bins=10) if len(d) > 200 else np.nan,
            "approval_rate": d.approve.mean(),
            "good_reject_rate": (~good.approve).mean() if len(good) else np.nan,
            "bad_approve_rate": bad.approve.mean() if len(bad) else np.nan,
        }
        if "pnl" in d:
            row["profit"] = d.pnl.sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def disparity_summary(report: pd.DataFrame, min_share: float = 0.01) -> dict:
    r = report[report["share"] >= min_share]
    return {
        "approval_rate_gap": float(r.approval_rate.max() - r.approval_rate.min()),
        "approval_rate_ratio": float(r.approval_rate.min() / r.approval_rate.max()),  # "four-fifths rule" < 0.8
        "good_reject_rate_gap": float(r.good_reject_rate.max() - r.good_reject_rate.min()),
        "bad_approve_rate_gap": float(r.bad_approve_rate.max() - r.bad_approve_rate.min()),
        "auc_gap": float(r.auc.max() - r.auc.min()),
        "calibration_gap": float((r.mean_pd - r.default_rate).abs().max()),
    }


# ------------------------------------------------------------------------------ mitigation
def group_thresholds(y, pd_hat, group, criterion: str, target: float) -> dict:
    """Per-group PD thresholds via quantiles on the *fitting* data.
    criterion='approval_rate'    -> every group approved at rate `target` (demographic parity)
    criterion='good_reject_rate' -> every group's creditworthy applicants rejected at rate `target`
                                    (equal opportunity)"""
    y, p, g = np.asarray(y), np.asarray(pd_hat), np.asarray(group)
    th = {}
    for grp in np.unique(g):
        m = g == grp
        if criterion == "approval_rate":
            th[grp] = float(np.quantile(p[m], target))
        elif criterion == "good_reject_rate":
            th[grp] = float(np.quantile(p[m & (y == 0)], 1 - target))
        else:
            raise ValueError(criterion)
    return th


def apply_group_thresholds(pd_hat, group, thresholds: dict, default: float) -> np.ndarray:
    p, g = np.asarray(pd_hat), np.asarray(group)
    t = np.array([thresholds.get(x, default) for x in g])
    return p < t


def proxy_auc(X: pd.DataFrame, sensitive: np.ndarray, seed: int = 42, n_estimators: int = 300) -> float:
    """How well can the *other* features predict the sensitive attribute? If AUC >> 0.5, simply dropping
    the attribute ('fairness through unawareness') does not remove the information."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict

    s = pd.Series(sensitive).astype("category").cat.codes.values
    clf = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=seed)
    pred = cross_val_predict(clf, X, s, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(s, pred))
