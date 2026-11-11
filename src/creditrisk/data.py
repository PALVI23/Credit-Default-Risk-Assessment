"""Loading the Home Credit Default Risk tables (Kaggle competition `home-credit-default-risk`)."""
from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous": "previous_application.csv",
    "pos": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

KAGGLE_COMPETITION = "home-credit-default-risk"


def setup_kaggle_credentials(legacy_kaggle_json: str | Path | None = None) -> str:
    """Configure Kaggle API access. Checked in order:
    1. Colab secret `KAGGLE_API_TOKEN` (recommended: key icon in Colab's left sidebar, enable notebook access)
    2. environment variable KAGGLE_API_TOKEN
    3. legacy kaggle.json at `legacy_kaggle_json`
    The token is never printed."""
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    token = None
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("KAGGLE_API_TOKEN")
    except Exception:  # noqa: BLE001 - not on Colab, or secret missing / access not granted
        pass
    token = token or os.environ.get("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token.strip()
        f = kdir / "access_token"
        f.write_text(token.strip())
        os.chmod(f, 0o600)
        return "access token (KAGGLE_API_TOKEN)"
    if legacy_kaggle_json and Path(legacy_kaggle_json).exists():
        dst = kdir / "kaggle.json"
        dst.write_bytes(Path(legacy_kaggle_json).read_bytes())
        os.chmod(dst, 0o600)
        return "legacy kaggle.json"
    raise RuntimeError(
        "No Kaggle credentials found. In Colab: click the key icon (Secrets) in the left sidebar -> "
        "'Add new secret' -> name KAGGLE_API_TOKEN, value = your KGAT_... token -> enable 'Notebook access'.")


def download_home_credit(data_dir: str | Path) -> Path:
    """Download + unzip the competition data. You must accept the competition rules on Kaggle first."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in TABLES.values()):
        print("Data already present:", data_dir)
        return data_dir
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(data_dir)],
        check=True,
    )
    for z in data_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_dir)
        z.unlink()
    return data_dir


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory: float64->float32, int64->smallest int, object->category."""
    for col in df.columns:
        s = df[col]
        if s.dtype == "float64":
            df[col] = s.astype("float32")
        elif s.dtype == "int64":
            df[col] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == "object":
            df[col] = s.astype("category")
    return df


def load_tables(data_dir: str | Path, nrows: int | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    out = {}
    for name, fname in TABLES.items():
        df = pd.read_csv(data_dir / fname, nrows=nrows)
        out[name] = downcast(df)
        if verbose:
            mb = out[name].memory_usage(deep=True).sum() / 1e6
            print(f"{name:15s} {out[name].shape!s:>20s}  {mb:8.1f} MB")
    return out


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return pd.DataFrame({"missing_share": miss, "dtype": df.dtypes[miss.index].astype(str)})


def stratified_holdout(df: pd.DataFrame, target: str = "TARGET", test_size: float = 0.2, seed: int = 42):
    """Single stratified split; the holdout is touched only once, at the very end."""
    from sklearn.model_selection import train_test_split

    tr, ho = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


__all__ = [
    "TABLES",
    "setup_kaggle_credentials",
    "download_home_credit",
    "load_tables",
    "downcast",
    "missing_report",
    "stratified_holdout",
]
