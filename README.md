# Personal Finance AI Tracker

An ML-powered personal finance dashboard built for Indian users, with
UPI-aware transaction cleaning, automatic spend categorization, anomaly
detection, and next-month spend forecasting.

🔗 **Live demo:** https://personal-finance-ai-tracker-nfftsityfwfjmsvuxicfgv.streamlit.app/

🔗 **Repo:** https://github.com/riddhi1606/personal-finance-ai-tracker

**Live focus:** most personal finance apps are built for US/EU bank
statement formats. This project is designed around the messy,
hyphen-and-VPA-handle-heavy narration strings that Indian UPI apps
(GPay, PhonePe, Paytm) actually produce, e.g.:

```
UPI-SWIGGY-swiggy@ybl-4441-223344556677-Payment from Phone
```

## Features

- **UPI-aware preprocessing** — strips VPA handles (`merchant@bank`),
  UTR/reference numbers, and transaction-type prefixes (UPI/IMPS/NEFT)
  to recover a clean merchant signal.
- **ML transaction categorization** — TF-IDF (character n-grams) +
  Multinomial Naive Bayes classifies each transaction into one of 11
  categories (Food & Dining, Groceries, Transport, Bills, etc.).
  Character n-grams were chosen over word-level features because
  merchant names in bank statements are often truncated or misspelled.
- **Anomaly detection** — Isolation Forest combined with
  category-relative z-scores flags transactions that are unusual
  either globally or *for their specific category* (₹3,000 is normal
  for Shopping but very unusual for Bills & Utilities).
- **Spend forecasting** — weighted moving average blended with a
  linear trend forecasts next month's total and per-category spend.
  Deliberately lightweight (not ARIMA/Prophet) since individual users
  typically only have a few months of transaction history.
- **Interactive dashboard** — Streamlit app with KPIs, charts, an
  anomaly explorer, and a live "try it yourself" categorizer.

## Project structure

```
finance-ai-tracker/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt
├── data/
│   ├── generate_sample_data.py # synthetic UPI transaction generator
│   └── transactions.csv        # generated sample dataset
├── models/
│   └── categorizer.joblib      # trained model (generated on first run)
├── src/
│   ├── preprocessing.py        # UPI narration cleaning
│   ├── categorization.py       # ML categorization model
│   ├── anomaly_detection.py    # Isolation Forest + z-score anomaly detection
│   └── forecasting.py          # spend forecasting
└── .streamlit/
    └── config.toml             # theme config
```

## Running locally

```bash
git clone https://github.com/riddhi1606/personal-finance-ai-tracker.git
cd finance-ai-tracker
pip install -r requirements.txt

# generate sample data (skip if you have your own CSV)
python data/generate_sample_data.py

streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Using your own data

Upload a CSV via the sidebar with these columns:

| column      | description                          |
|-------------|---------------------------------------|
| `date`      | transaction date                      |
| `narration` | raw bank/UPI description              |
| `amount`    | signed amount (negative = debit)      |
| `type`      | `Credit` or `Debit`                   |

The app will clean, categorize, and analyze it automatically. Note:
without labelled categories, the app uses the model trained on the
bundled sample dataset — for best results on your own spending
patterns, you can retrain (`python src/categorization.py`) after
adding a `true_category` column to a sample of your own transactions.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Select this repo, branch `main`, main file path `app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## Tech stack

Python, pandas, scikit-learn (TF-IDF, Multinomial Naive Bayes, Isolation
Forest), Streamlit, Plotly.

## Author

Riddhi Sharma — B.Tech AI & Data Science, Arya College of Engineering & IT, Jaipur
