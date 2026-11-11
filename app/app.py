"""Gradio demo: pick an applicant, tweak key inputs, get calibrated PD + decision + reason codes."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from creditrisk.explain import format_reasons, reason_codes, shap_values  # noqa: E402

EDITABLE = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]


def recompute_derived(r: pd.Series) -> pd.Series:
    """Keep engineered ratios consistent after the user edits raw inputs."""
    r = r.copy()
    div = lambda a, b: np.nan if (pd.isna(a) or pd.isna(b) or b == 0) else a / b
    r["CREDIT_INCOME_RATIO"] = div(r["AMT_CREDIT"], r["AMT_INCOME_TOTAL"])
    r["ANNUITY_INCOME_RATIO"] = div(r["AMT_ANNUITY"], r["AMT_INCOME_TOTAL"])
    r["ANNUITY_CREDIT_RATIO"] = div(r["AMT_ANNUITY"], r["AMT_CREDIT"])
    r["GOODS_CREDIT_RATIO"] = div(r["AMT_GOODS_PRICE"], r["AMT_CREDIT"])
    if "CNT_FAM_MEMBERS" in r:
        r["INCOME_PER_PERSON"] = div(r["AMT_INCOME_TOTAL"], r["CNT_FAM_MEMBERS"])
    ext = pd.to_numeric(pd.Series([r["EXT_SOURCE_1"], r["EXT_SOURCE_2"], r["EXT_SOURCE_3"]]), errors="coerce")
    r["EXT_SOURCE_MEAN"], r["EXT_SOURCE_STD"] = ext.mean(), ext.std()
    r["EXT_SOURCE_MIN"], r["EXT_SOURCE_MAX"] = ext.min(), ext.max()
    r["EXT_SOURCE_PROD"] = ext.prod() if ext.notna().all() else np.nan
    r["EXT_SOURCE_NAN_COUNT"] = int(ext.isna().sum())
    return r


class Scorer:
    def __init__(self, art: str):
        import lightgbm as lgb

        art = Path(art)
        self.features = json.load(open(art / "feature_list.json"))["features"]
        n_folds = json.load(open(art / "cv_results.json"))["n_folds"]
        self.boosters = [lgb.Booster(model_file=str(art / f"lgbm_tuned_fold{i}.txt")) for i in range(n_folds)]
        self.calibrator = pickle.load(open(art / "calibrator.pkl", "rb"))
        self.policy = json.load(open(art / "decision_policy.json"))
        self.applicants = pd.read_parquet(art / "demo_applicants.parquet")
        self.explainer_booster = self.boosters[0]

    def score(self, X: pd.DataFrame):
        raw = np.mean([b.predict(X[self.features]) for b in self.boosters], axis=0)
        return self.calibrator.transform(raw)

    def explain(self, X: pd.DataFrame):
        _, sv = shap_values(self.explainer_booster, X[self.features])
        return sv


def build_demo(art: str):
    import gradio as gr

    s = Scorer(art)
    ids = s.applicants["SK_ID_CURR"].astype(str).tolist()
    thr = s.policy["threshold"]

    def load(app_id):
        r = s.applicants[s.applicants.SK_ID_CURR.astype(str) == app_id].iloc[0]
        return [None if pd.isna(r[c]) else float(r[c]) for c in EDITABLE]

    def predict(app_id, *vals):
        base = s.applicants[s.applicants.SK_ID_CURR.astype(str) == app_id].iloc[0]
        r = base.copy()
        for c, v in zip(EDITABLE, vals):
            r[c] = np.nan if v is None else float(v)
        r = recompute_derived(r)
        X = pd.DataFrame([r[s.features]]).astype(s.applicants[s.features].dtypes.to_dict())
        pd_hat = float(s.score(X)[0])
        decision = "APPROVE" if pd_hat < thr else "REJECT"
        sv = s.explain(X)
        reasons = format_reasons(reason_codes(sv[0], X.iloc[0], k=5))
        md = (f"## {decision}\n**Probability of default:** {pd_hat:.1%}  (policy: approve if PD < {thr:.1%})\n\n"
              f"**Main factors increasing risk:**\n{reasons}")
        return md

    with gr.Blocks(title="Credit risk demo") as demo:
        gr.Markdown("# Credit default risk: calibrated, explainable scoring\n"
                    "LightGBM (5-fold ensemble) → isotonic/Platt calibration → profit-optimal threshold → SHAP reason codes.")
        with gr.Row():
            app_id = gr.Dropdown(ids, value=ids[0], label="Holdout applicant")
        inputs = [gr.Number(label=c) for c in EDITABLE]
        with gr.Row():
            btn = gr.Button("Score", variant="primary")
        out = gr.Markdown()
        app_id.change(load, app_id, inputs)
        btn.click(predict, [app_id, *inputs], out)
        demo.load(load, app_id, inputs)
    return demo


if __name__ == "__main__":
    build_demo(sys.argv[1] if len(sys.argv) > 1 else "artifacts").launch()
