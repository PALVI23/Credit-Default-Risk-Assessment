"""Feature engineering: one row per applicant (SK_ID_CURR).

Every auxiliary table is aggregated *per applicant* and describes history that existed before the
application, so there is no target leakage from joins. Aggregations are purely row-wise / per-group
(no statistics learned across applicants), so they can be computed once before cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANOMALY_DAYS = 365243  # Home Credit uses this as a "not applicable" sentinel in DAYS_* columns
SENSITIVE_COLUMNS = ["CODE_GENDER"]  # excluded from model features by default, kept for the audit
ID_COLUMNS = ["SK_ID_CURR", "TARGET"]


def _agg(df: pd.DataFrame, by: str, aggs: dict, prefix: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(aggs)
    g.columns = [f"{prefix}_{c}_{s}".upper() for c, s in g.columns]
    return g


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(a, dtype="float64") / np.asarray(b, dtype="float64")
    out[~np.isfinite(out)] = np.nan
    return out


# --------------------------------------------------------------------------------------------
def application_features(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == ANOMALY_DAYS).astype("int8")
    df.loc[df["DAYS_EMPLOYED"] == ANOMALY_DAYS, "DAYS_EMPLOYED"] = np.nan

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_CREDIT_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])  # ~ 1 / loan term
    df["GOODS_CREDIT_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["EMPLOYED_AGE_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype("float64")
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=3)
    df["EXT_SOURCE_NAN_COUNT"] = ext.isna().sum(axis=1).astype("int8")

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOC_COUNT"] = df[doc_cols].sum(axis=1).astype("int16")
    df["APP_NAN_COUNT"] = app.isna().sum(axis=1).astype("int16")
    return df


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb = bureau_balance.copy()
    status = bb["STATUS"].astype(str)
    bb["STATUS_NUM"] = pd.to_numeric(status, errors="coerce")  # '0'..'5' = DPD bucket; C/X -> NaN
    bb["LATE"] = bb["STATUS_NUM"].fillna(0).gt(0).astype("float32")
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_LATE_SHARE=("LATE", "mean"),
        BB_WORST_STATUS=("STATUS_NUM", "max"),
    )

    b = bureau.merge(bb_agg, how="left", left_on="SK_ID_BUREAU", right_index=True)
    active = b["CREDIT_ACTIVE"].astype(str)
    b["IS_ACTIVE"] = (active == "Active").astype("float32")
    b["IS_BAD_DEBT"] = (active == "Bad debt").astype("float32")
    b["DEBT_CREDIT_RATIO"] = _safe_div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    b["OVERDUE_DEBT_RATIO"] = _safe_div(b["AMT_CREDIT_SUM_OVERDUE"], b["AMT_CREDIT_SUM_DEBT"])

    aggs = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "OVERDUE_DEBT_RATIO": ["mean", "max"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "IS_ACTIVE": ["sum", "mean"],
        "IS_BAD_DEBT": ["sum"],
        "BB_MONTHS": ["mean", "sum"],
        "BB_LATE_SHARE": ["mean", "max"],
        "BB_WORST_STATUS": ["max"],
    }
    out = _agg(b, "SK_ID_CURR", aggs, "BURO")

    act = b[b["IS_ACTIVE"] == 1]
    act_agg = _agg(
        act,
        "SK_ID_CURR",
        {"AMT_CREDIT_SUM_DEBT": ["sum"], "AMT_CREDIT_SUM": ["sum"], "DAYS_CREDIT": ["mean"], "AMT_ANNUITY": ["sum"]},
        "BURO_ACTIVE",
    )
    return out.join(act_agg, how="left")


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    p = prev.copy()
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        if c in p:
            p.loc[p[c] == ANOMALY_DAYS, c] = np.nan
    status = p["NAME_CONTRACT_STATUS"].astype(str)
    p["APPROVED"] = (status == "Approved").astype("float32")
    p["REFUSED"] = (status == "Refused").astype("float32")
    p["APP_CREDIT_RATIO"] = _safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])

    aggs = {
        "SK_ID_PREV": ["count"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "APP_CREDIT_RATIO": ["mean", "min", "max"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "APPROVED": ["mean", "sum"],
        "REFUSED": ["mean", "sum"],
    }
    out = _agg(p, "SK_ID_CURR", aggs, "PREV")
    refused = _agg(p[p["REFUSED"] == 1], "SK_ID_CURR", {"DAYS_DECISION": ["max"], "AMT_APPLICATION": ["mean"]}, "PREV_REFUSED")
    return out.join(refused, how="left")


def pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy()
    p["COMPLETED"] = (p["NAME_CONTRACT_STATUS"].astype(str) == "Completed").astype("float32")
    aggs = {
        "MONTHS_BALANCE": ["size", "min"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "COMPLETED": ["mean"],
    }
    return _agg(p, "SK_ID_CURR", aggs, "POS")


def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    i = inst.copy()
    i["DPD"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["DBD"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    i["LATE"] = (i["DPD"] > 0).astype("float32")
    i["PAY_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]
    i["PAY_RATIO"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["UNDERPAID"] = (i["PAY_DIFF"] > 1).astype("float32")
    aggs = {
        "SK_ID_PREV": ["nunique"],
        "NUM_INSTALMENT_NUMBER": ["count"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean"],
        "LATE": ["mean", "sum"],
        "PAY_DIFF": ["max", "mean", "sum"],
        "PAY_RATIO": ["mean", "min"],
        "UNDERPAID": ["mean"],
        "AMT_PAYMENT": ["sum", "mean"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean"],
    }
    out = _agg(i, "SK_ID_CURR", aggs, "INST")
    # Recent behaviour (last 12 months) often matters more than lifetime averages
    recent = i[i["DAYS_INSTALMENT"] >= -365]
    rec = _agg(recent, "SK_ID_CURR", {"DPD": ["mean", "max"], "LATE": ["mean"], "PAY_RATIO": ["mean"]}, "INST_12M")
    return out.join(rec, how="left")


def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()
    c["UTIL"] = _safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    aggs = {
        "MONTHS_BALANCE": ["size"],
        "AMT_BALANCE": ["mean", "max"],
        "UTIL": ["mean", "max"],
        "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["mean"],
        "AMT_PAYMENT_CURRENT": ["mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max"],
    }
    return _agg(c, "SK_ID_CURR", aggs, "CC")


# --------------------------------------------------------------------------------------------
def build_features(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    df = application_features(tables["application"])
    blocks = {
        "bureau": lambda: bureau_features(tables["bureau"], tables["bureau_balance"]),
        "previous": lambda: previous_features(tables["previous"]),
        "pos": lambda: pos_features(tables["pos"]),
        "installments": lambda: installments_features(tables["installments"]),
        "credit_card": lambda: credit_card_features(tables["credit_card"]),
    }
    for name, fn in blocks.items():
        block = fn()
        df = df.merge(block, how="left", left_on="SK_ID_CURR", right_index=True)
        if verbose:
            print(f"+ {name:13s} {block.shape[1]:4d} features -> {df.shape[1]} columns")

    # Cross-table features
    df["TOTAL_DEBT_INCOME_RATIO"] = _safe_div(df["BURO_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"], df["AMT_INCOME_TOTAL"])
    df["TOTAL_ANNUITY_INCOME_RATIO"] = _safe_div(
        df["AMT_ANNUITY"].astype("float64") + df["BURO_ACTIVE_AMT_ANNUITY_SUM"].fillna(0), df["AMT_INCOME_TOTAL"]
    )
    df["HAS_BUREAU"] = df["BURO_SK_ID_BUREAU_COUNT"].notna().astype("int8")
    df["HAS_PREV"] = df["PREV_SK_ID_PREV_COUNT"].notna().astype("int8")

    num = df.select_dtypes(include=["float64", "float32"]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).astype("float32")
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category")
    return df


def feature_columns(df: pd.DataFrame, include_sensitive: bool = False) -> list[str]:
    drop = set(ID_COLUMNS) | (set() if include_sensitive else set(SENSITIVE_COLUMNS))
    return [c for c in df.columns if c not in drop]


# Human-readable names for the most important features (used for "reason codes")
FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_MEAN": "External credit score (average)",
    "EXT_SOURCE_1": "External credit score #1",
    "EXT_SOURCE_2": "External credit score #2",
    "EXT_SOURCE_3": "External credit score #3",
    "EXT_SOURCE_PROD": "External credit scores (product)",
    "EXT_SOURCE_MIN": "Lowest external credit score",
    "EXT_SOURCE_MAX": "Highest external credit score",
    "ANNUITY_CREDIT_RATIO": "Loan term (annuity / credit)",
    "CREDIT_INCOME_RATIO": "Loan amount relative to income",
    "ANNUITY_INCOME_RATIO": "Repayment burden (annuity / income)",
    "GOODS_CREDIT_RATIO": "Goods price relative to loan",
    "AGE_YEARS": "Applicant age",
    "DAYS_BIRTH": "Applicant age (days)",
    "DAYS_EMPLOYED": "Employment length",
    "EMPLOYED_AGE_RATIO": "Share of life employed",
    "DAYS_ID_PUBLISH": "Time since ID document was issued",
    "DAYS_LAST_PHONE_CHANGE": "Time since phone number change",
    "INST_DPD_MEAN": "Average days late on past instalments",
    "INST_DPD_MAX": "Worst days late on past instalments",
    "INST_12M_DPD_MEAN": "Average days late (last 12 months)",
    "INST_LATE_MEAN": "Share of past instalments paid late",
    "INST_PAY_RATIO_MEAN": "Share of instalment amounts actually paid",
    "INST_PAY_DIFF_SUM": "Total amount underpaid on instalments",
    "BURO_DAYS_CREDIT_MAX": "Most recent credit bureau loan",
    "BURO_DEBT_CREDIT_RATIO_MEAN": "Outstanding debt ratio (bureau)",
    "BURO_IS_ACTIVE_MEAN": "Share of bureau loans still active",
    "PREV_REFUSED_MEAN": "Share of previous applications refused",
    "PREV_APP_CREDIT_RATIO_MEAN": "Requested vs granted amount (previous loans)",
    "PREV_CNT_PAYMENT_MEAN": "Average term of previous loans",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "Remaining instalments on POS loans",
    "CC_UTIL_MEAN": "Credit card utilisation (average)",
    "CC_UTIL_MAX": "Credit card utilisation (max)",
    "AMT_ANNUITY": "Loan annuity",
    "AMT_CREDIT": "Loan amount",
    "AMT_GOODS_PRICE": "Goods price",
    "NAME_EDUCATION_TYPE": "Education level",
    "ORGANIZATION_TYPE": "Employer type",
    "OCCUPATION_TYPE": "Occupation",
    "TOTAL_DEBT_INCOME_RATIO": "Total active debt relative to income",
}


def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())
