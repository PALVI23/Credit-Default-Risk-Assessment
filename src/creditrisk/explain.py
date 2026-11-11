"""SHAP explanations and plain-language 'reason codes' (as required for adverse-action notices
under e.g. the US ECOA / EU consumer-credit rules)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import describe


def shap_values(booster, X: pd.DataFrame):
    import shap

    explainer = shap.TreeExplainer(booster)
    sv = explainer(X)
    if sv.values.ndim == 3:  # some versions return both classes
        sv = sv[:, :, 1]
    return explainer, sv


def global_importance(sv, feature_names) -> pd.DataFrame:
    imp = np.abs(sv.values).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def reason_codes(sv_row, x_row: pd.Series, k: int = 4) -> list[dict]:
    """Top-k features pushing this applicant's risk UP (positive SHAP in log-odds)."""
    vals = np.asarray(sv_row.values)
    order = np.argsort(-vals)
    out = []
    for j in order[:k]:
        if vals[j] <= 0:
            break
        f = x_row.index[j]
        out.append({"feature": f, "description": describe(f), "value": x_row.iloc[j], "shap_logodds": float(vals[j])})
    return out


def format_reasons(reasons: list[dict]) -> str:
    if not reasons:
        return "No factors materially increased the risk estimate."
    lines = []
    for r in reasons:
        v = r["value"]
        v = f"{v:,.3f}" if isinstance(v, (float, np.floating)) and pd.notna(v) else str(v)
        lines.append(f"- {r['description']} (value: {v}; +{r['shap_logodds']:.3f} log-odds)")
    return "\n".join(lines)
