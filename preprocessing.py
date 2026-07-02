"""
preprocessing.py
-----------------
Cleans raw Indian bank/UPI narration strings into a normalized merchant
name that the categorization model can learn from.

Real UPI narrations look like:
    'UPI-SWIGGY-swiggy@ybl-4441-223344556677-Payment from Phone'
This module strips reference numbers, VPA handles, bank codes, and
transaction-type prefixes to recover a clean signal: 'SWIGGY'.
"""

import re
import pandas as pd

# Common prefixes/suffixes seen in Indian statements
NOISE_PATTERNS = [
    r"^UPI[-/]",
    r"^IMPS[-/]",
    r"^NEFT[-/]",
    r"^RTGS[-/]",
    r"-Payment from Phone$",
    r"-Payment$",
]

# Recognized UPI handle suffixes (bank/PSP identifiers) to strip
UPI_HANDLES = [
    "ybl", "okhdfcbank", "okicici", "okaxis", "paytm", "sbi",
    "apl", "icici", "hdfcbank", "axis",
]


def strip_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def remove_reference_numbers(text: str) -> str:
    # Strip long numeric reference/UTR numbers (8+ digits)
    text = re.sub(r"\b\d{8,}\b", "", text)
    # Strip shorter standalone numeric tokens too (e.g. card-ending digits
    # like '4441'), but only once they're isolated by whitespace - i.e.
    # after normalize_whitespace has already split on hyphens/slashes.
    text = re.sub(r"(?<!\S)\d{3,}(?!\S)", "", text)
    return text


def remove_vpa_handles(text: str) -> str:
    # Remove tokens like 'SWIGGY@YBL' entirely (VPA handle). Operates on
    # whitespace-separated tokens only, so it must run AFTER hyphens/
    # slashes have been converted to spaces - otherwise '\S+@\S+' greedily
    # swallows the whole hyphen-joined string.
    text = re.sub(r"\S*@\S+", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_narration(raw: str) -> str:
    """Full pipeline: raw UPI string -> clean merchant token."""
    if not isinstance(raw, str):
        return ""
    text = raw.upper()
    text = strip_noise(text)
    text = normalize_whitespace(text)   # hyphens/slashes -> spaces FIRST
    text = remove_vpa_handles(text)     # now safe to strip 'X@Y' tokens
    text = remove_reference_numbers(text)
    text = normalize_whitespace(text)   # tidy up leftover double spaces
    return text.strip()


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw transactions dataframe (with a 'narration' column) and
    returns it enriched with cleaned features used downstream by both
    the categorization model and the anomaly detector.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["clean_merchant"] = df["narration"].apply(clean_narration)
    df["amount"] = df["amount"].astype(float)
    df["abs_amount"] = df["amount"].abs()
    df["day_of_week"] = df["date"].dt.day_name()
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["is_credit"] = df["amount"] > 0
    return df


if __name__ == "__main__":
    samples = [
        "UPI-SWIGGY-swiggy@ybl-4441-223344556677-Payment from Phone",
        "UPI/BIGBASKET/998877665544/bigbasket@ybl",
        "IMPS-LANDLORD RAJESH KUMAR-112233445566",
    ]
    for s in samples:
        print(f"{s!r} -> {clean_narration(s)!r}")
