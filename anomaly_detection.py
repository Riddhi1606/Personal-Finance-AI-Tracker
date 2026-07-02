"""
anomaly_detection.py
---------------------
Flags unusual transactions (e.g. an unexpectedly large one-off spend,
or a spend far outside a category's normal range) using Isolation
Forest, an unsupervised model well-suited to catching rare/irregular
points without needing labelled fraud data.

Two complementary signals are used:
1. Global anomaly score - is this transaction unusual across ALL spending?
2. Category-relative anomaly - is this unusual FOR ITS CATEGORY?
   (e.g. Rs.2000 is normal for Shopping but very unusual for Bills)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    Expects a dataframe with 'abs_amount', 'day_of_month', and
    'category' (predicted or true) columns. Adds 'is_anomaly' (bool)
    and 'anomaly_score' (float, lower = more anomalous) columns.
    """
    df = df.copy()
    spend_df = df[~df["is_credit"]].copy()

    if len(spend_df) < 10:
        df["is_anomaly"] = False
        df["anomaly_score"] = 0.0
        return df

    # --- Global model ---
    features = spend_df[["abs_amount", "day_of_month"]].fillna(0)
    global_model = IsolationForest(
        contamination=contamination, random_state=42, n_estimators=200
    )
    global_model.fit(features)
    spend_df["global_anomaly"] = global_model.predict(features) == -1
    spend_df["global_score"] = global_model.decision_function(features)

    # --- Category-relative model (z-score based, robust to small groups) ---
    cat_col = "category" if "category" in spend_df.columns else "true_category"
    stats = spend_df.groupby(cat_col)["abs_amount"].agg(["mean", "std"]).fillna(0)
    zero_std_mask = stats["std"] == 0
    stats.loc[zero_std_mask, "std"] = stats.loc[zero_std_mask, "mean"] * 0.3 + 1  # avoid div/0

    def cat_z(row):
        m, s = stats.loc[row[cat_col], ["mean", "std"]]
        return (row["abs_amount"] - m) / s

    spend_df["category_zscore"] = spend_df.apply(cat_z, axis=1)
    spend_df["category_anomaly"] = spend_df["category_zscore"].abs() > 2.5

    spend_df["is_anomaly"] = spend_df["global_anomaly"] | spend_df["category_anomaly"]
    spend_df["anomaly_score"] = spend_df["global_score"]

    df = df.merge(
        spend_df[["transaction_id", "is_anomaly", "anomaly_score", "category_zscore"]],
        on="transaction_id", how="left",
    )
    df["is_anomaly"] = df["is_anomaly"].fillna(False)
    df["anomaly_score"] = df["anomaly_score"].fillna(0.0)
    return df


def top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ["date", "narration", "clean_merchant", "category",
                         "true_category", "abs_amount", "anomaly_score"]
            if c in df.columns]
    return (
        df[df["is_anomaly"]]
        .sort_values("anomaly_score")
        .head(n)[cols]
    )


if __name__ == "__main__":
    from preprocessing import preprocess_transactions

    raw = pd.read_csv("data/transactions.csv")
    df = preprocess_transactions(raw)
    df["category"] = df["true_category"]
    result = detect_anomalies(df)
    print(f"Flagged {result['is_anomaly'].sum()} anomalies out of {len(result)} transactions")
    print(top_anomalies(result))
