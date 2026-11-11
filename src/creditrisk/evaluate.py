"""Discrimination, calibration, and business (profit) evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov separation between score distributions of defaulters vs non-defaulters
    (a standard metric in credit scoring)."""
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE with equal-frequency bins (more stable than equal-width bins when PDs are skewed)."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_summary(y, p, is_probability: bool = True) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    out = {
        "auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y, p),
        "ks": ks_statistic(y, p),
    }
    if is_probability:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        out.update({"brier": brier_score_loss(y, pc), "log_loss": log_loss(y, pc), "ece": expected_calibration_error(y, pc)})
    return {k: float(v) for k, v in out.items()}


class Calibrator:
    """Maps raw model scores to calibrated probabilities of default (PD).
    Fit on OOF predictions of the training split only."""

    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "platt")
        self.method = method

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        else:
            self.model_ = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return self.model_.predict(p)
        return expit(self.model_.decision_function(logit(p).reshape(-1, 1)))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def plot_reliability(ax, y, curves: dict[str, np.ndarray], n_bins: int = 10):
    for name, p in curves.items():
        t = reliability_table(y, p, n_bins)
        ax.plot(t["mean_pred"], t["observed"], marker="o", label=f"{name} (ECE={expected_calibration_error(y, p):.4f})")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.legend()


# ------------------------------------------------------------------------------ business view
def profit_curve(y, pd_hat, credit, margin: float = 0.10, lgd: float = 0.50, thresholds=None) -> pd.DataFrame:
    """Simple lending P&L. Approve if PD < threshold.
    A repaid loan earns `margin * credit`; a defaulted loan loses `lgd * credit` (loss given default).
    Both numbers are stated assumptions, not data, and are varied in a sensitivity analysis."""
    y, p, credit = np.asarray(y), np.asarray(pd_hat), np.asarray(credit, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(p, np.linspace(0.01, 1.0, 200)))
    pnl = np.where(y == 0, margin * credit, -lgd * credit)
    order = np.argsort(p)
    p_sorted, pnl_sorted, y_sorted = p[order], pnl[order], y[order]
    cum_pnl, cum_bad = np.cumsum(pnl_sorted), np.cumsum(y_sorted)
    rows = []
    for t in thresholds:
        k = np.searchsorted(p_sorted, t, side="left")  # number approved (PD < t)
        rows.append({
            "threshold": float(t),
            "approval_rate": k / len(p),
            "bad_rate_approved": cum_bad[k - 1] / k if k > 0 else np.nan,
            "profit": cum_pnl[k - 1] if k > 0 else 0.0,
        })
    out = pd.DataFrame(rows)
    out["profit_vs_approve_all"] = out["profit"] - pnl.sum()
    return out


def best_threshold(curve: pd.DataFrame) -> dict:
    r = curve.loc[curve["profit"].idxmax()]
    return {k: float(v) for k, v in r.items()}


def breakeven_pd(margin: float, lgd: float) -> float:
    """Approve iff expected profit > 0: (1-PD)*margin - PD*lgd > 0  <=>  PD < margin / (margin + lgd).
    Only valid if PDs are *calibrated* - which is why calibration matters for decisions."""
    return margin / (margin + lgd)
