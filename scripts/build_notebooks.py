"""Generates the Colab notebooks in ../notebooks from the cell definitions below.
Run:  python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# ---------- Setup: run this cell first ----------
import os, sys, subprocess, json, warnings
warnings.filterwarnings("ignore")
SMOKE = os.environ.get("CR_SMOKE") == "1"          # local smoke test on synthetic data (ignore on Colab)
if SMOKE:
    PROJECT_DIR = os.environ["CR_PROJECT_DIR"]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/masters-projects/credit-risk"   # <- change if you put it elsewhere
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{PROJECT_DIR}/requirements-colab.txt"], check=True)
sys.path.insert(0, f"{PROJECT_DIR}/src")
ART = f"{PROJECT_DIR}/artifacts"; FIG = f"{PROJECT_DIR}/reports/figures"
os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
SEED = 42

def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=160, bbox_inches="tight"); plt.show()

def save_json(obj, name):
    with open(f"{ART}/{name}.json", "w") as f: json.dump(obj, f, indent=2, default=float)
print("Project dir:", PROJECT_DIR)'''


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"provenance": [], "gpuType": "T4"}}
    for kind, src in cells:
        src = src.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return book


# =============================================================================================== 01
NB1 = [
("md", """# 01 · Data, EDA and feature engineering
**Project:** Credit default risk: a calibrated, explainable and fairness-audited ML system
**Data:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (307k loan applications, 7 relational tables)

This notebook downloads the data, analyses it, builds ~400 applicant-level features from the
relational tables, and creates the **train / holdout split**. The holdout is not touched again until notebook 03.

> Before running: (1) accept the competition rules on Kaggle (Rules tab -> "I Understand and Accept");
> (2) add your Kaggle API token as a **Colab secret** named `KAGGLE_API_TOKEN` (key icon in the left sidebar,
> toggle "Notebook access" on). Runtime: CPU is fine."""),
("code", SETUP),
("md", "## 1. Download and load the data"),
("code", '''from creditrisk import data as D
if SMOKE:
    from creditrisk.synthetic import make_synthetic_tables
    tables = make_synthetic_tables(n=4000)
else:
    print("Kaggle auth:", D.setup_kaggle_credentials(legacy_kaggle_json="/content/drive/MyDrive/masters-projects/kaggle.json"))
    DATA_DIR = D.download_home_credit("/content/data/home-credit")   # local disk is much faster than Drive
    tables = D.load_tables(DATA_DIR)
app = tables["application"]
print(app.shape)'''),
("md", "## 2. Target and class imbalance"),
("code", '''rate = app["TARGET"].mean()
print(f"Default rate: {rate:.2%}  ->  {1/rate:.1f} : 1 imbalance")
fig, ax = plt.subplots(figsize=(4, 3))
app["TARGET"].value_counts().sort_index().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["repaid (0)", "default (1)"], rotation=0); ax.set_title("Target distribution")
savefig("01_target")'''),
("md", "## 3. Data quality: missing values and anomalies"),
("code", '''miss = D.missing_report(app)
print(f"{(miss.missing_share > 0).sum()} / {app.shape[1]} columns have missing values; "
      f"{(miss.missing_share > 0.5).sum()} are >50% missing")
fig, ax = plt.subplots(figsize=(7, 6))
miss.head(30)["missing_share"][::-1].plot.barh(ax=ax, color="#8172B3"); ax.set_xlabel("share missing")
ax.set_title("Top-30 columns by missingness (application table)")
savefig("01_missingness")'''),
("code", '''anom = app["DAYS_EMPLOYED"] == 365243
print(f"DAYS_EMPLOYED == 365243 (sentinel ~1000 years): {anom.mean():.2%} of applicants")
print(app.groupby(anom)["TARGET"].mean().rename({False: "normal", True: "sentinel"}))
print("Income type of sentinel rows:\\n", app.loc[anom, "NAME_INCOME_TYPE"].value_counts().head(3))
# -> almost all pensioners/unemployed; we set it to NaN and keep a flag (DAYS_EMPLOYED_ANOM)'''),
("md", "## 4. What predicts default? Univariate views"),
("code", '''df = app.copy()
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes[0], ["EXT_SOURCE_2", "EXT_SOURCE_3"]):
    q = pd.qcut(df[col], 10, duplicates="drop")
    df.groupby(q, observed=True)["TARGET"].mean().plot(ax=ax, marker="o")
    ax.set_title(f"Default rate by {col} decile"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60)
df.groupby(pd.cut(df["AGE_YEARS"], [20, 25, 30, 35, 40, 45, 50, 55, 60, 70]), observed=True)["TARGET"].mean() \\
  .plot.bar(ax=axes[1, 0], color="#55A868"); axes[1, 0].set_title("Default rate by age")
df.groupby("NAME_EDUCATION_TYPE", observed=True)["TARGET"].mean().sort_values() \\
  .plot.barh(ax=axes[1, 1], color="#DD8452"); axes[1, 1].set_title("Default rate by education"); axes[1, 1].set_ylabel("")
savefig("01_univariate")'''),
("code", '''# Sensitive attributes we will audit later (gender is NOT used as a model feature)
print(app.groupby("CODE_GENDER", observed=True)["TARGET"].agg(["mean", "size"]))'''),
("code", '''# Strongest linear correlations with the target (numeric columns)
num = app.select_dtypes("number").drop(columns=["SK_ID_CURR"])
corr = num.corr()["TARGET"].drop("TARGET").sort_values()
pd.concat([corr.head(8), corr.tail(8)]).to_frame("pearson_r")'''),
("md", """## 5. Feature engineering
One row per applicant. From each auxiliary table we aggregate the applicant's *history*:
* **bureau / bureau_balance**: loans at other lenders, active debt, overdue amounts, monthly delinquency status
* **previous_application**: past Home Credit applications, refusal rate, requested vs. granted amounts
* **installments_payments**: days-past-due, under-payment, recent (12-month) behaviour
* **POS_CASH / credit_card_balance**: delinquency, utilisation, drawings
* **domain ratios**: credit/income, annuity/income, annuity/credit (≈ loan term), external-score combinations

All aggregations are per applicant and use only data from before the application, so the joins cannot leak the target."""),
("code", '''from creditrisk.features import build_features, feature_columns
feat = build_features(tables)
FEATURES = feature_columns(feat)                  # excludes ID, TARGET and CODE_GENDER
print(f"Feature matrix: {feat.shape},  model features: {len(FEATURES)}")
print(f"Memory: {feat.memory_usage(deep=True).sum()/1e6:.0f} MB")'''),
("md", "## 6. Train / holdout split (stratified 80/20; the holdout is used once, in notebook 03)"),
("code", '''from sklearn.model_selection import train_test_split
tr_ids, ho_ids = train_test_split(feat["SK_ID_CURR"], test_size=0.2, stratify=feat["TARGET"], random_state=SEED)
feat["SPLIT"] = np.where(feat["SK_ID_CURR"].isin(ho_ids), "holdout", "train")
print(feat.groupby("SPLIT")["TARGET"].agg(["size", "mean"]))
feat.to_parquet(f"{ART}/features.parquet", index=False)
save_json({"features": FEATURES}, "feature_list")
save_json({"n_rows": int(len(feat)), "n_features": len(FEATURES), "default_rate": float(feat.TARGET.mean()),
           "n_train": int((feat.SPLIT == "train").sum()), "n_holdout": int((feat.SPLIT == "holdout").sum()),
           "days_employed_sentinel_share": float(anom.mean()),
           "n_cols_over_50pct_missing": int((miss.missing_share > 0.5).sum())}, "eda_summary")
print("Saved features.parquet to", ART)'''),
("md", """## Observations (fill in after running, in your own words)
* Imbalance: ...
* Strongest signals: ...
* Data-quality issues and how they were handled: ..."""),
]

# =============================================================================================== 02
NB2 = [
("md", """# 02 · Modelling: baselines, gradient boosting, tuning, ablations
Every model is evaluated with **5-fold stratified CV on the training split** (AUC). The out-of-fold (OOF)
predictions are kept for calibration and threshold selection in notebook 03, so the holdout stays clean.

> Runtime: **GPU (T4)** recommended (XGBoost and CatBoost use it). Total time is about 60–90 min. Set `QUICK = True` for a ~15 min run."""),
("code", SETUP),
("code", '''QUICK = SMOKE or False          # True = 3 folds, fewer Optuna trials (for a fast first pass)
N_FOLDS = 3 if QUICK else 5
N_TRIALS = 4 if SMOKE else (10 if QUICK else 25)

from creditrisk import models as M
feat = pd.read_parquet(f"{ART}/features.parquet")
FEATURES = json.load(open(f"{ART}/feature_list.json"))["features"]
tr, ho = feat[feat.SPLIT == "train"].reset_index(drop=True), feat[feat.SPLIT == "holdout"].reset_index(drop=True)
X, y, X_ho = tr[FEATURES], tr["TARGET"].values, ho[FEATURES]
print(X.shape, X_ho.shape, "GPU:", M.has_gpu())
results = {}'''),
("md", "## 1. Baseline: regularised logistic regression (median-impute + scale + one-hot)"),
("code", '''results["logreg"] = M.cv_logistic(X, y, X_ho, n_folds=N_FOLDS, seed=SEED)
results["logreg"].summary(y)'''),
("md", "## 2. Gradient-boosted trees: LightGBM, XGBoost, CatBoost (default-ish params)"),
("code", '''results["lgbm"] = M.cv_lightgbm(X, y, X_ho, n_folds=N_FOLDS, seed=SEED)
results["lgbm"].summary(y)'''),
("code", '''results["xgb"] = M.cv_xgboost(X, y, X_ho, n_folds=N_FOLDS, seed=SEED)
results["xgb"].summary(y)'''),
("code", '''results["cat"] = M.cv_catboost(X, y, X_ho, n_folds=N_FOLDS, seed=SEED)
results["cat"].summary(y)'''),
("md", """## 3. Class imbalance: does re-weighting help?
A common approach is to up-weight the minority class (`scale_pos_weight`). We compare ranking quality (AUC)
**and** probability quality (Brier, ECE): re-weighting usually leaves AUC about the same but distorts the probabilities."""),
("code", '''from creditrisk.evaluate import classification_summary
spw = (y == 0).sum() / (y == 1).sum()
results["lgbm_weighted"] = M.cv_lightgbm(X, y, X_ho, {"scale_pos_weight": spw}, n_folds=N_FOLDS, seed=SEED,
                                         name="LightGBM (weighted)")
imb = pd.DataFrame({k: classification_summary(y, results[k].oof) for k in ["lgbm", "lgbm_weighted"]}).T
imb[["auc", "pr_auc", "brier", "ece"]]'''),
("md", "## 4. Hyper-parameter tuning (Optuna TPE, 3-fold CV on a 50% subsample, proxy LR=0.08)"),
("code", '''study = M.tune_lightgbm(X, y, n_trials=N_TRIALS, timeout_minutes=40, sample_frac=0.5 if not SMOKE else 1.0, seed=SEED)
best_params = study.best_params
print("best proxy AUC:", round(study.best_value, 4)); best_params'''),
("code", '''results["lgbm_tuned"] = M.cv_lightgbm(X, y, X_ho, {**best_params, "learning_rate": 0.02}, n_folds=N_FOLDS,
                                      seed=SEED, name="LightGBM (tuned)")
save_json(best_params, "best_params")
results["lgbm_tuned"].summary(y)'''),
("code", '''tr_df = study.trials_dataframe()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(tr_df["number"], tr_df["value"], "o", alpha=.5, label="trial"); ax.plot(tr_df["number"], tr_df["value"].cummax(), label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("CV AUC (proxy)"); ax.legend(); ax.set_title("Optuna search")
savefig("02_optuna")'''),
("md", """## 5. Ablation: how much does each data source add?
We add one relational table at a time to the application-only model (same params, 3-fold CV, LR=0.05)."""),
("code", '''prefix_groups = [("application only", None), ("+ bureau", "BURO_"), ("+ previous apps", "PREV_"),
                  ("+ POS cash", "POS_"), ("+ instalments", "INST_"), ("+ credit card", "CC_")]
derived = ["TOTAL_DEBT_INCOME_RATIO", "TOTAL_ANNUITY_INCOME_RATIO", "HAS_BUREAU", "HAS_PREV"]
aux_prefixes = [p for _, p in prefix_groups if p]
base_cols = [c for c in FEATURES if not any(c.startswith(p) for p in aux_prefixes) and c not in derived]
cols, rows = list(base_cols), []
for label, prefix in prefix_groups:
    if prefix:
        cols += [c for c in FEATURES if c.startswith(prefix)]
    r = M.cv_lightgbm(X[cols], y, X_ho[cols].iloc[:1], {**best_params, "learning_rate": 0.05}, n_folds=3, seed=SEED,
                      name=label, verbose=False)
    rows.append({"features": label, "n_features": len(cols), "cv_auc": np.mean(r.fold_auc)})
    print(rows[-1])
ablation = pd.DataFrame(rows); ablation["delta_auc"] = ablation["cv_auc"].diff()
ablation.to_csv(f"{ART}/ablation_data_sources.csv", index=False); ablation'''),
("code", '''fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ablation["features"], ablation["cv_auc"], marker="o"); ax.set_ylabel("CV AUC")
ax.set_title("Cumulative contribution of each data source"); ax.tick_params(axis="x", rotation=20)
savefig("02_ablation_sources")'''),
("md", "## 6. Blending (rank-average; weights chosen on OOF predictions only)"),
("code", '''blend_members = ["lgbm_tuned", "xgb", "cat"]
w, blend_auc = M.blend_weights({k: results[k].oof for k in blend_members}, y)
print("weights:", w, " OOF AUC:", round(blend_auc, 4))
blend_oof = M.apply_blend({k: results[k].oof for k in blend_members}, w)
blend_ho = M.apply_blend({k: results[k].holdout for k in blend_members}, w)'''),
("md", "## 7. Model comparison (cross-validated, training split only)"),
("code", '''from sklearn.metrics import roc_auc_score
summary = pd.DataFrame([results[k].summary(y) for k in results])
summary = pd.concat([summary, pd.DataFrame([{"model": "Blend (rank avg)", "oof_auc": roc_auc_score(y, blend_oof)}])])
summary = summary.sort_values("oof_auc", ascending=False).reset_index(drop=True)
summary.to_csv(f"{ART}/cv_model_comparison.csv", index=False)
summary.round(4)'''),
("code", '''from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(5.5, 5))
for k in ["logreg", "lgbm_tuned", "xgb", "cat"]:
    fpr, tpr, _ = roc_curve(y, results[k].oof); ax.plot(fpr, tpr, label=f"{results[k].name} ({roc_auc_score(y, results[k].oof):.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("OOF ROC curves"); ax.legend()
savefig("02_roc")'''),
("code", '''imp = pd.DataFrame({"feature": FEATURES,
                    "gain": np.mean([b.feature_importance("gain") for b in results["lgbm_tuned"].models], axis=0)})
imp = imp.sort_values("gain", ascending=False).head(25)
fig, ax = plt.subplots(figsize=(7, 7)); ax.barh(imp.feature[::-1], imp.gain[::-1], color="#4C72B0")
ax.set_title("LightGBM (tuned): top-25 features by gain"); savefig("02_importance_gain")'''),
("md", "## 8. Save predictions and models for notebook 03"),
("code", '''oof = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR, "TARGET": y, **{k: r.oof for k, r in results.items()}, "blend": blend_oof})
hop = pd.DataFrame({"SK_ID_CURR": ho.SK_ID_CURR, "TARGET": ho.TARGET.values, **{k: r.holdout for k, r in results.items()}, "blend": blend_ho})
oof.to_parquet(f"{ART}/oof_predictions.parquet", index=False); hop.to_parquet(f"{ART}/holdout_predictions.parquet", index=False)
for i, b in enumerate(results["lgbm_tuned"].models):
    b.save_model(f"{ART}/lgbm_tuned_fold{i}.txt")
save_json({"n_folds": N_FOLDS, "blend_weights": w, "final_model": "lgbm_tuned",
           "imbalance": imb[["auc", "brier", "ece"]].to_dict(orient="index"),
           "cv": summary.to_dict(orient="records")}, "cv_results")
print("saved.")'''),
("md", """## Observations (fill in after running)
* Baseline → boosted trees: ...
* Re-weighting: ...
* Which data sources matter most and why: ..."""),
]

# =============================================================================================== 03
NB3 = [
("md", """# 03 · Calibration, decisions, explainability and fairness
Turns the ranking model into a **decision system**: calibrated probabilities of default (PD), a
profit-maximising approval threshold, per-applicant explanations (reason codes) and a fairness audit with mitigation.
This is the **only** notebook that evaluates on the holdout.

> Runtime: CPU is fine (about 15–25 min)."""),
("code", SETUP),
("code", '''import lightgbm as lgb, pickle
from creditrisk.evaluate import (classification_summary, Calibrator, plot_reliability, profit_curve,
                                 best_threshold, breakeven_pd, expected_calibration_error)
from creditrisk import fairness as F, explain as E
feat = pd.read_parquet(f"{ART}/features.parquet")
FEATURES = json.load(open(f"{ART}/feature_list.json"))["features"]
oof, hop = pd.read_parquet(f"{ART}/oof_predictions.parquet"), pd.read_parquet(f"{ART}/holdout_predictions.parquet")
cvres = json.load(open(f"{ART}/cv_results.json"))
tr = feat[feat.SPLIT == "train"].set_index("SK_ID_CURR").loc[oof.SK_ID_CURR].reset_index()
ho = feat[feat.SPLIT == "holdout"].set_index("SK_ID_CURR").loc[hop.SK_ID_CURR].reset_index()
y, y_ho = oof.TARGET.values, hop.TARGET.values
MODEL = "lgbm_tuned"
boosters = [lgb.Booster(model_file=f"{ART}/lgbm_tuned_fold{i}.txt") for i in range(cvres["n_folds"])]
print(len(tr), len(ho))'''),
("md", """## 1. Calibration
Calibrators are fit on **OOF** predictions (training split) and evaluated on the holdout."""),
("code", '''cal = {"raw": None, "platt": Calibrator("platt").fit(oof[MODEL], y), "isotonic": Calibrator("isotonic").fit(oof[MODEL], y)}
pd_ho = {k: (hop[MODEL].values if c is None else c.transform(hop[MODEL])) for k, c in cal.items()}
# Also: the re-weighted model's raw probabilities, to show why weighting hurts calibration
pd_ho["weighted_raw"] = hop["lgbm_weighted"].values
calib_table = pd.DataFrame({k: classification_summary(y_ho, v) for k, v in pd_ho.items()}).T
calib_table[["auc", "brier", "log_loss", "ece"]].round(5)'''),
("code", '''fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_reliability(axes[0], y_ho, {k: pd_ho[k] for k in ["raw", "platt", "isotonic"]}); axes[0].set_title("Reliability (holdout)")
plot_reliability(axes[1], y_ho, {"weighted_raw": pd_ho["weighted_raw"], "isotonic": pd_ho["isotonic"]})
axes[1].set_title("Class-weighted model is badly mis-calibrated")
savefig("03_calibration")
best_cal = min(["platt", "isotonic"], key=lambda k: calib_table.loc[k, "brier"])
calibrator = cal[best_cal]; PD_OOF, PD_HO = calibrator.transform(oof[MODEL]), pd_ho[best_cal]
print("chosen calibrator:", best_cal)'''),
("md", "## 2. Final holdout performance (first and only use of the holdout)"),
("code", '''final_rows = {"Logistic regression": hop["logreg"], "LightGBM (default)": hop["lgbm"], "XGBoost": hop["xgb"],
              "CatBoost": hop["cat"], "LightGBM (tuned)": hop["lgbm_tuned"], "Blend": hop["blend"]}
final = pd.DataFrame({k: classification_summary(y_ho, v, is_probability=False) for k, v in final_rows.items()}).T
final.loc["LightGBM (tuned) + calibration", ["brier", "ece"]] = [calib_table.loc[best_cal, "brier"], calib_table.loc[best_cal, "ece"]]
final.to_csv(f"{ART}/holdout_metrics.csv"); final.round(4)'''),
("md", """## 3. From probabilities to decisions: a profit-maximising threshold
Simple P&L assumptions (stated, not learned): a repaid loan earns **10%** of the credit amount; a default loses
**50%** (loss given default). With calibrated PDs, theory says: approve iff PD < margin/(margin+LGD) = **0.167**.
We pick the threshold on the OOF predictions and report the holdout result."""),
("code", '''MARGIN, LGD = 0.10, 0.50
curve_tr = profit_curve(y, PD_OOF, tr.AMT_CREDIT, MARGIN, LGD)
bt = best_threshold(curve_tr); THR = bt["threshold"]
print(f"empirical best threshold (train OOF): {THR:.3f}   theoretical break-even PD: {breakeven_pd(MARGIN, LGD):.3f}")
curve_ho = profit_curve(y_ho, PD_HO, ho.AMT_CREDIT, MARGIN, LGD, thresholds=[THR, breakeven_pd(MARGIN, LGD), 1.01])
curve_ho["policy"] = ["optimised threshold", "theoretical break-even", "approve everyone"]
curve_ho'''),
("code", '''fig, ax1 = plt.subplots(figsize=(7, 4))
c = profit_curve(y_ho, PD_HO, ho.AMT_CREDIT, MARGIN, LGD)
ax1.plot(c.threshold, c.profit / 1e9, label="profit (bn)"); ax1.axvline(THR, ls="--", c="k", lw=1)
ax1.set_xlabel("PD threshold (approve if PD < t)"); ax1.set_ylabel("holdout profit (bn currency units)")
ax2 = ax1.twinx(); ax2.plot(c.threshold, c.approval_rate, c="#55A868", label="approval rate"); ax2.set_ylabel("approval rate")
ax1.set_xlim(0, 0.5); ax1.set_title("Profit vs. approval threshold (holdout)"); fig.legend(loc="lower right")
savefig("03_profit_curve")'''),
("code", '''# Sensitivity: how does the optimal threshold move with the business assumptions?
sens = []
for m in [0.05, 0.10, 0.15, 0.20]:
    for l in [0.3, 0.5, 0.7]:
        b = best_threshold(profit_curve(y, PD_OOF, tr.AMT_CREDIT, m, l))
        sens.append({"margin": m, "lgd": l, "best_threshold": b["threshold"], "breakeven_pd": breakeven_pd(m, l),
                     "approval_rate": b["approval_rate"]})
sens = pd.DataFrame(sens); sens.round(3)'''),
("md", "## 4. Explainability (SHAP)"),
("code", '''import shap
sample = ho.sample(min(3000, len(ho)), random_state=SEED)
explainer, sv = E.shap_values(boosters[0], sample[FEATURES])
gimp = E.global_importance(sv, FEATURES); gimp.head(15)'''),
("code", '''shap.plots.beeswarm(sv, max_display=20, show=False); plt.title("SHAP summary (holdout sample)"); savefig("03_shap_beeswarm")'''),
("code", '''top3 = gimp.feature.head(3).tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, f in zip(axes, top3):
    shap.plots.scatter(sv[:, f], ax=ax, show=False)
savefig("03_shap_dependence")'''),
("code", '''# Reason codes for a rejected applicant (adverse-action explanation) and an approved one
s_pd = calibrator.transform(np.mean([b.predict(sample[FEATURES]) for b in boosters], axis=0))
rej = int(np.argmax(s_pd)); acc = int(np.argmin(s_pd))
for label, i in [("REJECTED", rej), ("APPROVED", acc)]:
    print(f"=== {label} applicant {sample.SK_ID_CURR.iloc[i]}  PD={s_pd[i]:.3f} ===")
    print(E.format_reasons(E.reason_codes(sv[i], sample[FEATURES].iloc[i])), "\\n")
shap.plots.waterfall(sv[rej], max_display=12, show=False); savefig("03_shap_waterfall_rejected")'''),
("md", """## 5. Fairness audit
Groups: **gender** (not a model input) and **age band** (a model input, as in many real scorecards).
Decision: approve if PD < threshold from Section 3. Metrics are defined in `src/creditrisk/fairness.py`."""),
("code", '''approve_ho = PD_HO < THR
ho_gender = ho.CODE_GENDER.astype(str).values; ho_age = F.age_group(ho.AGE_YEARS).values
mask_g = np.isin(ho_gender, ["F", "M"])
rep_gender = F.group_report(y_ho[mask_g], PD_HO[mask_g], approve_ho[mask_g], ho_gender[mask_g], ho.AMT_CREDIT.values[mask_g])
rep_age = F.group_report(y_ho, PD_HO, approve_ho, ho_age, ho.AMT_CREDIT.values)
display(rep_gender.round(4)); display(rep_age.round(4))
disp = {"gender": F.disparity_summary(rep_gender), "age": F.disparity_summary(rep_age)}
pd.DataFrame(disp).round(4)'''),
("code", '''fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, col, t in zip(axes, ["default_rate", "approval_rate", "good_reject_rate"],
                      ["Observed default rate", "Approval rate", "Creditworthy applicants rejected"]):
    rep_age[col].plot.bar(ax=ax, color="#4C72B0"); ax.set_title(t + " by age"); ax.tick_params(axis="x", rotation=0)
savefig("03_fairness_age")'''),
("md", """### 5a. Does dropping gender make the model "blind" to it? (proxy test)
If the other features predict gender well, removing the column does not remove the information."""),
("code", '''mask_tr = tr.CODE_GENDER.astype(str).isin(["F", "M"]).values
n_proxy = min(60000, mask_tr.sum())
idx = np.random.RandomState(SEED).choice(np.where(mask_tr)[0], n_proxy, replace=False)
gender_proxy_auc = F.proxy_auc(tr[FEATURES].iloc[idx], tr.CODE_GENDER.astype(str).values[idx], seed=SEED)
print(f"Gender can be predicted from the model's features with AUC = {gender_proxy_auc:.3f}")'''),
("md", "### 5b. Ablation: training *with* gender as a feature"),
("code", '''from creditrisk import models as M
best_params = json.load(open(f"{ART}/best_params.json"))
F_WITH = FEATURES + ["CODE_GENDER"]
r_with = M.cv_lightgbm(tr[F_WITH], y, ho[F_WITH], {**best_params, "learning_rate": 0.05}, n_folds=3, seed=SEED, name="with gender")
r_wo = M.cv_lightgbm(tr[FEATURES], y, ho[FEATURES], {**best_params, "learning_rate": 0.05}, n_folds=3, seed=SEED, name="without gender")
rows = []
for name, r in [("without gender (ours)", r_wo), ("with gender", r_with)]:
    c = Calibrator(best_cal).fit(r.oof, y); p_ho = c.transform(r.holdout)
    t = best_threshold(profit_curve(y, c.transform(r.oof), tr.AMT_CREDIT, MARGIN, LGD))["threshold"]
    rep = F.group_report(y_ho[mask_g], p_ho[mask_g], (p_ho < t)[mask_g], ho_gender[mask_g])
    rows.append({"model": name, "holdout_auc": classification_summary(y_ho, p_ho)["auc"], **F.disparity_summary(rep)})
unaware = pd.DataFrame(rows).set_index("model"); unaware.round(4)'''),
("md", """### 5c. Mitigation: group-specific thresholds (post-processing)
* **Equal opportunity:** creditworthy applicants in every group are rejected at the same rate.
* **Demographic parity:** every group is approved at the same rate.

Thresholds are fit on the training OOF predictions and evaluated on the holdout.
*Caveat:* group-specific thresholds use the protected attribute at decision time. That is disparate treatment, which is illegal in many jurisdictions (e.g. under the US ECOA).
We report them to quantify the **cost of fairness**, not as a deployment recommendation."""),
("code", '''tr_gender = tr.CODE_GENDER.astype(str).values
m_tr = np.isin(tr_gender, ["F", "M"])
base_approve_tr = PD_OOF < THR
target_grr = (~base_approve_tr[m_tr & (y == 0)]).mean()   # overall good-reject rate
target_ar = base_approve_tr[m_tr].mean()                  # overall approval rate
policies = {
    "single threshold (baseline)": {g: THR for g in ["F", "M"]},
    "equal opportunity": F.group_thresholds(y[m_tr], PD_OOF[m_tr], tr_gender[m_tr], "good_reject_rate", target_grr),
    "demographic parity": F.group_thresholds(y[m_tr], PD_OOF[m_tr], tr_gender[m_tr], "approval_rate", target_ar),
}
rows = []
for name, th in policies.items():
    appr = F.apply_group_thresholds(PD_HO[mask_g], ho_gender[mask_g], th, THR)
    rep = F.group_report(y_ho[mask_g], PD_HO[mask_g], appr, ho_gender[mask_g], ho.AMT_CREDIT.values[mask_g], MARGIN, LGD)
    d = F.disparity_summary(rep)
    rows.append({"policy": name, "thr_F": th["F"], "thr_M": th["M"], "approval_rate": appr.mean(),
                 "profit": rep.profit.sum(), "approval_gap": d["approval_rate_gap"],
                 "good_reject_gap": d["good_reject_rate_gap"], "bad_approve_gap": d["bad_approve_rate_gap"]})
mitig = pd.DataFrame(rows).set_index("policy")
mitig["profit_change_%"] = 100 * (mitig.profit / mitig.profit.iloc[0] - 1)
mitig.to_csv(f"{ART}/fairness_mitigation.csv"); mitig.round(4)'''),
("md", "## 6. Save final artefacts and an auto-generated results summary"),
("code", '''pickle.dump(calibrator, open(f"{ART}/calibrator.pkl", "wb"))
save_json({"threshold": THR, "margin": MARGIN, "lgd": LGD, "calibrator": best_cal, "model": MODEL}, "decision_policy")
sample[["SK_ID_CURR"] + FEATURES].head(500).to_parquet(f"{ART}/demo_applicants.parquet", index=False)
gimp.to_csv(f"{ART}/shap_global_importance.csv", index=False)
eda = json.load(open(f"{ART}/eda_summary.json")); cvsum = pd.read_csv(f"{ART}/cv_model_comparison.csv")
abl = pd.read_csv(f"{ART}/ablation_data_sources.csv")
fin = final.loc["LightGBM (tuned)"]
lines = [
 "# Results summary (auto-generated by notebook 03)", "",
 f"* Data: {eda['n_rows']:,} applicants, {eda['n_features']} features, default rate {eda['default_rate']:.2%}; "
 f"train {eda['n_train']:,} / holdout {eda['n_holdout']:,}",
 f"* **Holdout (tuned LightGBM): AUC {fin.auc:.4f}, Gini {fin.gini:.4f}, KS {fin.ks:.4f}, PR-AUC {fin.pr_auc:.4f}**",
 f"* Logistic-regression baseline holdout AUC: {final.loc['Logistic regression','auc']:.4f}; blend: {final.loc['Blend','auc']:.4f}",
 f"* Calibration ({best_cal}): Brier {calib_table.loc[best_cal,'brier']:.5f}, ECE {calib_table.loc[best_cal,'ece']:.5f} "
 f"(raw ECE {calib_table.loc['raw','ece']:.5f}; class-weighted model ECE {calib_table.loc['weighted_raw','ece']:.5f})",
 f"* Decision threshold PD < {THR:.3f} (theory: {breakeven_pd(MARGIN, LGD):.3f}); holdout profit vs approve-all: "
 f"{curve_ho.loc[0,'profit_vs_approve_all']:,.0f} at approval rate {curve_ho.loc[0,'approval_rate']:.2%} "
 f"(bad rate among approved {curve_ho.loc[0,'bad_rate_approved']:.2%} vs {y_ho.mean():.2%} overall)",
 f"* Gender proxy AUC from features: {gender_proxy_auc:.3f}",
 "", "## CV comparison", cvsum.round(4).to_markdown(index=False),
 "", "## Holdout metrics", final.round(4).to_markdown(),
 "", "## Data-source ablation", abl.round(4).to_markdown(index=False),
 "", "## Calibration", calib_table.round(5).to_markdown(),
 "", "## Threshold sensitivity", sens.round(3).to_markdown(index=False),
 "", "## Fairness: gender (single threshold)", rep_gender.round(4).to_markdown(),
 "", "## Fairness: age", rep_age.round(4).to_markdown(),
 "", "## Fairness through unawareness", unaware.round(4).to_markdown(),
 "", "## Mitigation", mitig.round(4).to_markdown(),
 "", "## Top-15 SHAP features", gimp.head(15).round(5).to_markdown(index=False),
]
open(f"{PROJECT_DIR}/reports/results_summary.md", "w").write("\\n".join(lines))
print("\\n".join(lines[:9]))'''),
("md", "**Next:** download `reports/results_summary.md` and the `reports/figures/` folder and share them back to write the final report."),
]

# =============================================================================================== 04
NB4 = [
("md", """# 04 · Interactive demo (Gradio)
Loads the trained model, calibrator and decision policy and launches a web demo with a public link.
Pick a holdout applicant, edit key inputs, and see the PD, the decision and the reason codes."""),
("code", SETUP),
("code", '''sys.path.insert(0, f"{PROJECT_DIR}/app")
from app import build_demo
demo = build_demo(ART)
demo.launch(share=not SMOKE, debug=False, prevent_thread_lock=SMOKE)'''),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_data_eda_features", NB1), ("02_modeling", NB2), ("03_calibration_explainability_fairness", NB3),
                        ("04_demo_app", NB4)]:
        nbf.write(nb(cells), out / f"{name}.ipynb")
        print("wrote", out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
