# Data folder

Place `cleaned_churn_data.csv` here before running `train_model.py`.

Source dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The raw Kaggle CSV is cleaned in `churn_analysis.ipynb` (handles missing
`TotalCharges` values, drops duplicates, fixes dtypes) and exported here as
`cleaned_churn_data.csv`. This file is git-ignored — generate it locally by
running the notebook end to end.
