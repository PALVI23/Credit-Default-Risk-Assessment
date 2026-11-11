"""Tiny synthetic stand-in for the Home Credit tables (same column names) - used only for
smoke tests / CI so the whole pipeline can run in seconds without the 2.6 GB Kaggle download."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_tables(n: int = 4000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100000, 100000 + n)
    ext = rng.beta(4, 3, size=(n, 3))
    ext[rng.random((n, 3)) < np.array([0.55, 0.01, 0.2])] = np.nan
    age_days = -rng.integers(21 * 365, 69 * 365, n)
    income = rng.lognormal(11.8, 0.5, n)
    credit = income * rng.uniform(0.5, 6, n)
    annuity = credit / rng.integers(6, 60, n)
    employed = np.where(rng.random(n) < 0.18, 365243, -rng.integers(30, 12000, n))
    gender = rng.choice(["F", "M", "XNA"], n, p=[0.65, 0.3499, 0.0001])
    late_tendency = rng.beta(1, 8, n)
    z = (-2.6 - 3.0 * (np.nan_to_num(ext, nan=0.5).mean(1) - 0.55) * 3 + 4 * late_tendency
         + 0.3 * (credit / income > 4) + 0.15 * (gender == "M") + 0.01 * (age_days / 365 + 40))
    target = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": gender, "FLAG_OWN_CAR": rng.choice(["Y", "N"], n), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "CNT_CHILDREN": rng.poisson(0.4, n), "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Commercial associate", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], n),
        "OCCUPATION_TYPE": np.where(rng.random(n) < 0.3, None, rng.choice(["Laborers", "Sales staff", "Core staff", "Drivers"], n)),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "Self-employed", "XNA", "Other"], n),
        "DAYS_BIRTH": age_days, "DAYS_EMPLOYED": employed, "DAYS_REGISTRATION": -rng.integers(0, 15000, n).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n), "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float), "REGION_RATING_CLIENT": rng.integers(1, 4, n),
        "EXT_SOURCE_1": ext[:, 0], "EXT_SOURCE_2": ext[:, 1], "EXT_SOURCE_3": ext[:, 2],
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n), "FLAG_DOCUMENT_6": rng.integers(0, 2, n),
    })
    # bureau + bureau_balance
    nb = rng.poisson(4, n)
    b_curr = np.repeat(ids, nb); m = len(b_curr)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_curr, "SK_ID_BUREAU": np.arange(m) + 5_000_000,
        "CREDIT_ACTIVE": rng.choice(["Closed", "Active", "Sold", "Bad debt"], m, p=[0.6, 0.38, 0.019, 0.001]),
        "DAYS_CREDIT": -rng.integers(0, 2900, m), "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 30], m),
        "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 3000, m).astype(float), "AMT_CREDIT_MAX_OVERDUE": rng.exponential(500, m),
        "CNT_CREDIT_PROLONG": rng.poisson(0.01, m), "AMT_CREDIT_SUM": rng.lognormal(12, 1, m),
        "AMT_CREDIT_SUM_DEBT": rng.lognormal(10, 2, m), "AMT_CREDIT_SUM_OVERDUE": rng.exponential(10, m),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card"], m), "DAYS_CREDIT_UPDATE": -rng.integers(0, 2000, m),
        "AMT_ANNUITY": rng.exponential(10000, m),
    })
    nbb = rng.integers(1, 12, m)
    bb = pd.DataFrame({"SK_ID_BUREAU": np.repeat(bureau.SK_ID_BUREAU.values, nbb),
                       "MONTHS_BALANCE": -rng.integers(0, 90, nbb.sum()),
                       "STATUS": rng.choice(["C", "X", "0", "1", "2", "5"], nbb.sum(), p=[0.4, 0.2, 0.35, 0.03, 0.01, 0.01])})
    # previous applications
    npv = rng.poisson(3, n)
    p_curr = np.repeat(ids, npv); k = len(p_curr)
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(k) + 2_000_000, "SK_ID_CURR": p_curr,
        "NAME_CONTRACT_TYPE": rng.choice(["Consumer loans", "Cash loans"], k),
        "AMT_ANNUITY": rng.exponential(15000, k), "AMT_APPLICATION": rng.lognormal(11, 1, k),
        "AMT_CREDIT": rng.lognormal(11, 1, k), "AMT_DOWN_PAYMENT": rng.exponential(3000, k),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], k, p=[0.6, 0.2, 0.2]),
        "DAYS_DECISION": -rng.integers(1, 2900, k), "CNT_PAYMENT": rng.integers(6, 48, k).astype(float),
        "DAYS_FIRST_DRAWING": np.where(rng.random(k) < 0.7, 365243.0, -rng.integers(0, 2900, k)),
        "DAYS_TERMINATION": np.where(rng.random(k) < 0.3, 365243.0, -rng.integers(0, 2900, k)),
    })
    pid = prev.SK_ID_PREV.values; pc = prev.SK_ID_CURR.values
    nrep = rng.integers(1, 10, k)
    late_by_curr = dict(zip(ids, late_tendency))
    ic = np.repeat(pc, nrep)
    lt = np.array([late_by_curr[c] for c in ic])
    days_inst = -rng.integers(1, 2900, len(ic)).astype(float)
    inst = pd.DataFrame({
        "SK_ID_PREV": np.repeat(pid, nrep), "SK_ID_CURR": ic, "NUM_INSTALMENT_VERSION": rng.integers(0, 3, len(ic)).astype(float),
        "NUM_INSTALMENT_NUMBER": rng.integers(1, 40, len(ic)), "DAYS_INSTALMENT": days_inst,
        "DAYS_ENTRY_PAYMENT": days_inst + np.where(rng.random(len(ic)) < lt, rng.integers(1, 60, len(ic)), -rng.integers(0, 10, len(ic))),
        "AMT_INSTALMENT": rng.exponential(10000, len(ic)),
    })
    inst["AMT_PAYMENT"] = inst.AMT_INSTALMENT * np.where(rng.random(len(ic)) < lt, rng.uniform(0.3, 1, len(ic)), 1.0)
    pos = pd.DataFrame({"SK_ID_PREV": np.repeat(pid, nrep), "SK_ID_CURR": ic, "MONTHS_BALANCE": -rng.integers(1, 96, len(ic)),
                        "CNT_INSTALMENT": rng.integers(6, 48, len(ic)).astype(float),
                        "CNT_INSTALMENT_FUTURE": rng.integers(0, 30, len(ic)).astype(float),
                        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], len(ic), p=[0.9, 0.1]),
                        "SK_DPD": (rng.random(len(ic)) < lt * 0.3) * rng.integers(1, 40, len(ic)), "SK_DPD_DEF": 0})
    cc_curr = rng.choice(ids, n // 3, replace=False)
    ncc = rng.integers(3, 24, len(cc_curr)); cid = np.repeat(cc_curr, ncc)
    limit = rng.choice([0, 45000, 90000, 135000], len(cid)).astype(float)
    cc = pd.DataFrame({"SK_ID_PREV": np.repeat(np.arange(len(cc_curr)) + 3_000_000, ncc), "SK_ID_CURR": cid,
                       "MONTHS_BALANCE": -rng.integers(1, 96, len(cid)), "AMT_BALANCE": limit * rng.uniform(0, 1.1, len(cid)),
                       "AMT_CREDIT_LIMIT_ACTUAL": limit, "AMT_DRAWINGS_CURRENT": rng.exponential(5000, len(cid)),
                       "CNT_DRAWINGS_CURRENT": rng.poisson(1, len(cid)).astype(float), "AMT_PAYMENT_CURRENT": rng.exponential(5000, len(cid)),
                       "SK_DPD": rng.choice([0, 0, 0, 0, 5], len(cid)), "SK_DPD_DEF": 0})
    from .data import downcast
    return {name: downcast(df) for name, df in
            {"application": app, "bureau": bureau, "bureau_balance": bb, "previous": prev, "pos": pos,
             "installments": inst, "credit_card": cc}.items()}
