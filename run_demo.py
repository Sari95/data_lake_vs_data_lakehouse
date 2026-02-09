import os
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Optional: DuckDB (für Mini-Lakehouse)
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_LAKE_PATH = os.path.join(PROJECT_ROOT, "data_lake", "churn_data.csv")
LAKEHOUSE_DIR = os.path.join(PROJECT_ROOT, "lakehouse")
LAKEHOUSE_V1 = os.path.join(LAKEHOUSE_DIR, "churn_v1.csv")
LAKEHOUSE_V2 = os.path.join(LAKEHOUSE_DIR, "churn_v2.csv")
LAKEHOUSE_DB = os.path.join(LAKEHOUSE_DIR, "lakehouse.duckdb")

META_DIR = os.path.join(PROJECT_ROOT, "scripts", "runs_meta")
os.makedirs(META_DIR, exist_ok=True)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dataset_v1(n=500, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, size=n)
    charges = np.round(rng.normal(65, 20, size=n).clip(10, 150), 2)
    tickets = rng.poisson(1.2, size=n)

    # "True" relationship: more tickets + higher charges + low tenure => higher churn
    logit = -2.0 + 0.06 * tickets + 0.015 * (charges - 65) - 0.02 * (tenure - 30)
    p = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, p)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets": tickets,
        "churn": churn
    })
    return df


def make_dataset_v2_from_v1(df_v1: pd.DataFrame, seed=11) -> pd.DataFrame:
    """
    Simuliert realistische Änderungen:
    - neue Kundenzeilen
    - unterschiedliche Missing-Value-Kodierung (Tickets -> NaN statt 0 für einige)
    - leicht veränderte Feature-Berechnung (charges gerundet/angepasst für eine Teilmenge)
    """
    rng = np.random.default_rng(seed)
    df = df_v1.copy()

    # 1) neue Kunden hinzufügen
    n_new = 60
    new_ids = np.arange(df["customer_id"].max() + 1, df["customer_id"].max() + 1 + n_new)
    tenure_new = rng.integers(1, 24, size=n_new)
    charges_new = np.round(rng.normal(75, 22, size=n_new).clip(10, 160), 2)
    tickets_new = rng.poisson(1.6, size=n_new)

    logit = -1.8 + 0.07 * tickets_new + 0.018 * (charges_new - 65) - 0.02 * (tenure_new - 30)
    p = 1 / (1 + np.exp(-logit))
    churn_new = rng.binomial(1, p)

    df_new = pd.DataFrame({
        "customer_id": new_ids,
        "tenure_months": tenure_new,
        "monthly_charges": charges_new,
        "support_tickets": tickets_new,
        "churn": churn_new
    })

    df = pd.concat([df, df_new], ignore_index=True)

    # 2) Missing-Value-Kodierung: bei ~8% tickets -> NaN (statt "0" oder sauber imputiert)
    idx = rng.choice(df.index, size=int(0.08 * len(df)), replace=False)
    df.loc[idx, "support_tickets"] = np.nan

    # 3) "Feature changed": charges bei ~15% anders gerundet/angepasst (z. B. neues Billing rounding)
    idx2 = rng.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[idx2, "monthly_charges"] = np.round(df.loc[idx2, "monthly_charges"] * 1.02, 2)

    return df


def basic_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "missing_support_tickets": int(df["support_tickets"].isna().sum()),
        "charges_mean": float(df["monthly_charges"].mean()),
        "tickets_mean": float(pd.to_numeric(df["support_tickets"], errors="coerce").mean()),
        "churn_rate": float(df["churn"].mean())
    }


def train_and_eval(df: pd.DataFrame, run_name: str, data_ref: str, data_hash: str) -> dict:
    # Minimal preprocessing (absichtlich): NaNs werden zu 0 imputiert -> realistisch, aber gefährlich
    X = df[["tenure_months", "monthly_charges", "support_tickets"]].copy()
    X["support_tickets"] = X["support_tickets"].fillna(0)

    y = df["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    meta = {
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_ref": data_ref,
        "data_hash": data_hash,
        "accuracy": float(acc),
        "stats": basic_stats(df),
        "coef": {
            "tenure_months": float(model.coef_[0][0]),
            "monthly_charges": float(model.coef_[0][1]),
            "support_tickets": float(model.coef_[0][2]),
            "intercept": float(model.intercept_[0]),
        }
    }
    return meta


def save_meta(meta: dict):
    out = os.path.join(META_DIR, f"{meta['run_name']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] Saved run metadata -> {out}")


def write_lake_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Wrote -> {path}")


def lakehouse_load_csv_version(version: str) -> pd.DataFrame:
    if version == "v1":
        path = LAKEHOUSE_V1
    elif version == "v2":
        path = LAKEHOUSE_V2
    else:
        raise ValueError("version must be v1 or v2")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Create datasets first.")
    return pd.read_csv(path)


def lakehouse_duckdb_register(version: str):
    if not HAS_DUCKDB:
        print("[SKIP] DuckDB not installed.")
        return

    con = duckdb.connect(LAKEHOUSE_DB)
    con.execute("CREATE TABLE IF NOT EXISTS churn_v1 AS SELECT * FROM read_csv_auto(?)", [LAKEHOUSE_V1])
    # For v2 we replace table to keep example simple (but in a real lakehouse we'd keep versions separately or time travel)
    con.execute("CREATE OR REPLACE TABLE churn_v2 AS SELECT * FROM read_csv_auto(?)", [LAKEHOUSE_V2])

    rows_v1 = con.execute("SELECT COUNT(*) FROM churn_v1").fetchone()[0]
    rows_v2 = con.execute("SELECT COUNT(*) FROM churn_v2").fetchone()[0]
    con.close()
    print(f"[OK] DuckDB tables ready (v1 rows={rows_v1}, v2 rows={rows_v2}) -> {LAKEHOUSE_DB}")


def diff_versions(df1: pd.DataFrame, df2: pd.DataFrame):
    # Quick diffs useful for the article
    new_ids = set(df2["customer_id"]) - set(df1["customer_id"])
    print("\n=== DIFF (v1 -> v2) ===")
    print(f"New customers in v2: {len(new_ids)}")
    print(f"Missing support_tickets v1: {df1['support_tickets'].isna().sum()} | v2: {df2['support_tickets'].isna().sum()}")
    print(f"Mean charges v1: {df1['monthly_charges'].mean():.2f} | v2: {df2['monthly_charges'].mean():.2f}")
    print("=======================")


def main():
    os.makedirs(LAKEHOUSE_DIR, exist_ok=True)

    # 1) Create v1 and write it to both lake and lakehouse/v1
    df_v1 = make_dataset_v1()
    write_lake_csv(LAKEHOUSE_V1, df_v1)
    write_lake_csv(DATA_LAKE_PATH, df_v1)  # "Data lake": single mutable file

    # 2) Train on "lake" (v1)
    hash_v1_lake = sha256_of_file(DATA_LAKE_PATH)
    meta1 = train_and_eval(pd.read_csv(DATA_LAKE_PATH), "lake_run_1", "data_lake/churn_data.csv (v1)", hash_v1_lake)
    print("\n[LAKE RUN 1]", meta1["accuracy"], meta1["stats"])
    save_meta(meta1)

    # 3) Create v2 and OVERWRITE lake file (this simulates reality)
    df_v2 = make_dataset_v2_from_v1(df_v1)
    write_lake_csv(LAKEHOUSE_V2, df_v2)
    write_lake_csv(DATA_LAKE_PATH, df_v2)  # overwrite!

    # 4) Train on "lake" again (now it's v2 but path is identical)
    hash_v2_lake = sha256_of_file(DATA_LAKE_PATH)
    meta2 = train_and_eval(pd.read_csv(DATA_LAKE_PATH), "lake_run_2", "data_lake/churn_data.csv (overwritten to v2)", hash_v2_lake)
    print("\n[LAKE RUN 2]", meta2["accuracy"], meta2["stats"])
    save_meta(meta2)

    # 5) Lakehouse training explicitly by version
    df_lh_v1 = lakehouse_load_csv_version("v1")
    df_lh_v2 = lakehouse_load_csv_version("v2")
    diff_versions(df_lh_v1, df_lh_v2)

    meta3 = train_and_eval(df_lh_v1, "lakehouse_run_v1", "lakehouse/churn_v1.csv", sha256_of_file(LAKEHOUSE_V1))
    print("\n[LAKEHOUSE v1]", meta3["accuracy"], meta3["stats"])
    save_meta(meta3)

    meta4 = train_and_eval(df_lh_v2, "lakehouse_run_v2", "lakehouse/churn_v2.csv", sha256_of_file(LAKEHOUSE_V2))
    print("\n[LAKEHOUSE v2]", meta4["accuracy"], meta4["stats"])
    save_meta(meta4)

    lakehouse_duckdb_register("v2")

    print("\nDone. Check scripts/runs_meta/*.json for reproducibility evidence (data hash + stats).")


if __name__ == "__main__":
    main()
