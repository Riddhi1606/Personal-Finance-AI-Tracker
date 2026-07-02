"""
generate_sample_data.py
------------------------
Generates a realistic synthetic dataset of Indian UPI/bank transactions
for the Personal Finance AI Tracker project.

Why synthetic data with UPI-style noise:
Real Indian bank/UPI statements have messy narration strings like
'UPI-SWIGGY-swiggy@ybl-4441-XXXXX@YESB-123456789012-Payment'
This generator mimics that so the preprocessing module has something
real to normalize.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ---- Merchant / category mapping (used to generate + later to evaluate) ----
MERCHANTS = {
    "Food & Dining": [
        ("SWIGGY", "swiggy@ybl"), ("ZOMATO", "zomato@paytm"),
        ("DOMINOS", "dominos@okhdfcbank"), ("CHAIPOINT", "chaipoint@ybl"),
        ("MCDONALDS", "mcd@okicici"),
    ],
    "Groceries": [
        ("BIGBASKET", "bigbasket@ybl"), ("BLINKIT", "blinkit@okhdfcbank"),
        ("DMART", "dmart@paytm"), ("ZEPTO", "zepto@ybl"),
        ("LOCAL KIRANA STORE", "kirana123@ybl"),
    ],
    "Transport": [
        ("UBER", "uber@paytm"), ("OLA", "ola@okicici"),
        ("RAPIDO", "rapido@ybl"), ("IRCTC", "irctc@sbi"),
        ("INDIAN OIL PETROL PUMP", "iocl@ybl"),
    ],
    "Shopping": [
        ("AMAZON", "amazon@apl"), ("FLIPKART", "flipkart@icici"),
        ("MYNTRA", "myntra@ybl"), ("AJIO", "ajio@paytm"),
        ("RELIANCE DIGITAL", "reliancedigital@ybl"),
    ],
    "Bills & Utilities": [
        ("JAIPUR VIDYUT VITRAN NIGAM", "jvvnl@sbi"), ("AIRTEL POSTPAID", "airtel@paytm"),
        ("JIO RECHARGE", "jio@icici"), ("TATA SKY", "tatasky@ybl"),
        ("MUNICIPAL WATER BOARD", "waterboard@ybl"),
    ],
    "Entertainment": [
        ("NETFLIX", "netflix@hdfcbank"), ("BOOKMYSHOW", "bms@ybl"),
        ("SPOTIFY", "spotify@okaxis"), ("PVR CINEMAS", "pvr@paytm"),
        ("HOTSTAR", "hotstar@icici"),
    ],
    "Education": [
        ("ARYA COLLEGE FEE PORTAL", "arya.edu@sbi"), ("UDEMY", "udemy@okhdfcbank"),
        ("COURSERA", "coursera@ybl"), ("BYJUS", "byjus@paytm"),
        ("BOOK STORE", "bookstore@ybl"),
    ],
    "Healthcare": [
        ("APOLLO PHARMACY", "apollo@ybl"), ("PRACTO", "practo@icici"),
        ("MEDPLUS", "medplus@paytm"), ("CITY HOSPITAL", "cityhospital@sbi"),
    ],
    "Rent & Housing": [
        ("LANDLORD RAJESH KUMAR", "rajeshk@ybl"), ("PG ACCOMMODATION", "pgowner@paytm"),
    ],
    "Income": [
        ("SALARY CREDIT EVOASTRA VENTURE", "evoastra@hdfcbank"),
        ("FREELANCE PAYMENT", "client@ybl"), ("SCHOLARSHIP CREDIT", "college@sbi"),
        ("FAMILY TRANSFER", "father@ybl"),
    ],
    "Investment": [
        ("ZERODHA", "zerodha@icici"), ("GROWW MUTUAL FUND", "groww@ybl"),
        ("PPF DEPOSIT", "ppf@sbi"),
    ],
}

BANKS = ["okhdfcbank", "ybl", "okicici", "okaxis", "paytm", "sbi"]


def random_ref():
    return "".join(random.choices("0123456789", k=12))


def make_narration(merchant, handle):
    """Builds a messy real-world-looking UPI narration string."""
    ref = random_ref()
    templates = [
        f"UPI-{merchant}-{handle}-{ref}-Payment from Phone",
        f"UPI/{merchant}/{ref}/{handle}",
        f"{merchant}-{handle}-UPI-{ref}",
        f"IMPS-{merchant}-{ref}",
        f"NEFT-{merchant.replace(' ', '')}-{ref}",
    ]
    return random.choice(templates)


def generate(n_days=270, start_date="2025-10-01", out_path="data/transactions.csv"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for day_offset in range(n_days):
        date = start + timedelta(days=day_offset)

        # Monthly salary/income around the 1st-3rd of each month
        if date.day in (1, 2, 3) and random.random() < 0.5:
            cat = "Income"
            merchant, handle = random.choice(MERCHANTS[cat])
            amt = round(random.uniform(15000, 25000), 2)
            rows.append([date, make_narration(merchant, handle), amt, "Credit", cat])

        # Rent around the 5th
        if date.day == 5 and random.random() < 0.3:
            cat = "Rent & Housing"
            merchant, handle = random.choice(MERCHANTS[cat])
            amt = -round(random.uniform(6000, 9000), 2)
            rows.append([date, make_narration(merchant, handle), amt, "Debit", cat])

        # Daily random spends (0-4 transactions/day)
        n_txn = np.random.poisson(1.6)
        for _ in range(n_txn):
            cat = random.choices(
                population=[c for c in MERCHANTS if c not in ("Income", "Rent & Housing")],
                weights=[18, 15, 14, 12, 10, 8, 5, 6, 4],
                k=1,
            )[0]
            merchant, handle = random.choice(MERCHANTS[cat])
            amt = -round(np.random.gamma(shape=2.0, scale=180) + 20, 2)
            rows.append([date, make_narration(merchant, handle), amt, "Debit", cat])

        # Occasional anomaly: unusually large one-off spend (~2% of days)
        if random.random() < 0.02:
            cat = random.choice(["Shopping", "Healthcare", "Entertainment"])
            merchant, handle = random.choice(MERCHANTS[cat])
            amt = -round(random.uniform(8000, 20000), 2)
            rows.append([date, make_narration(merchant, handle), amt, "Debit", cat])

    df = pd.DataFrame(rows, columns=["date", "narration", "amount", "type", "true_category"])
    df = df.sort_values("date").reset_index(drop=True)
    df["transaction_id"] = ["TXN" + str(i).zfill(6) for i in range(len(df))]
    df = df[["transaction_id", "date", "narration", "amount", "type", "true_category"]]

    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions -> {out_path}")
    return df


if __name__ == "__main__":
    generate()
