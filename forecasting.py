"""
forecasting.py
---------------
Forecasts next month's total spend and per-category spend using a
simple, explainable approach: weighted moving average + linear trend
over monthly aggregates. This is intentionally not a heavy time-series
model (ARIMA/Prophet) because most student/individual users only have
a few months of transaction history, which isn't enough data to fit
something more complex reliably.
"""

import numpy as np
import pandas as pd


def monthly_spend(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    spend = df[~df["is_credit"]].copy()
    grouped = (
        spend.groupby(["month", category_col])["abs_amount"]
        .sum()
        .reset_index()
        .rename(columns={"abs_amount": "total_spend"})
    )
    return grouped


def forecast_next_month_total(df: pd.DataFrame) -> dict:
    """
    Weighted moving average (recent months weighted higher) + linear
    trend adjustment. Returns a dict with the forecast and a naive
    confidence range.
    """
    spend = df[~df["is_credit"]]
    monthly_totals = spend.groupby("month")["abs_amount"].sum().sort_index()

    if len(monthly_totals) < 2:
        last = monthly_totals.iloc[-1] if len(monthly_totals) else 0.0
        return {"forecast": round(last, 2), "low": round(last * 0.85, 2),
                "high": round(last * 1.15, 2), "trend": "insufficient_data"}

    values = monthly_totals.values
    n = len(values)
    weights = np.arange(1, n + 1)  # later months weighted more
    weighted_avg = np.average(values, weights=weights)

    # Linear trend via simple least squares on month index
    x = np.arange(n)
    slope, intercept = np.polyfit(x, values, 1)
    trend_forecast = slope * n + intercept

    forecast = 0.6 * weighted_avg + 0.4 * trend_forecast
    forecast = max(forecast, 0)

    std = np.std(values)
    trend_label = "increasing" if slope > 50 else "decreasing" if slope < -50 else "stable"

    return {
        "forecast": round(float(forecast), 2),
        "low": round(float(max(forecast - std, 0)), 2),
        "high": round(float(forecast + std), 2),
        "trend": trend_label,
        "monthly_history": monthly_totals.round(2).to_dict(),
    }


def forecast_by_category(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    grouped = monthly_spend(df, category_col)
    results = []
    for cat, sub in grouped.groupby(category_col):
        sub = sub.sort_values("month")
        values = sub["total_spend"].values
        if len(values) == 1:
            forecast = values[0]
        else:
            weights = np.arange(1, len(values) + 1)
            forecast = float(np.average(values, weights=weights))
        results.append({"category": cat, "forecast_next_month": round(forecast, 2),
                         "avg_past_months": round(float(np.mean(values)), 2)})
    return pd.DataFrame(results).sort_values("forecast_next_month", ascending=False)


if __name__ == "__main__":
    from preprocessing import preprocess_transactions

    raw = pd.read_csv("data/transactions.csv")
    df = preprocess_transactions(raw)
    df["category"] = df["true_category"]

    print(forecast_next_month_total(df))
    print()
    print(forecast_by_category(df))
