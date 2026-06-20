# 📊 Customer Churn Analysis — End-to-End Data Analytics Project

> Analyzed telecom customer data to identify churn patterns and predict churn risk in real time using Python, SQL, Power BI, and Flask.

---
## Live Demo: https://customerchurnanalysis-fyxgpqrtrqfzqew4jhq3wq.streamlit.app/

## Highlights

- **Analyzed a telecom dataset of 7,032 records** using Python and MySQL — computed a churn rate of **26.58%**, executed 5 business queries, and uncovered key drivers including contract type and payment method.
- **Built an interactive Power BI dashboard** with 4 KPI cards (Churned: 1,869 | Avg. Monthly Charges: ₹64.80), 3 analysis charts, and a dynamic slicer for contract-type filtering.
- **Deployed a Flask web application** for real-time churn prediction, integrating the trained ML model (Random Forest, 78.32% accuracy) with a user-facing interface for instant churn probability scores.

**Tools Used:** Python · Pandas · Scikit-learn · MySQL · Flask · Power BI · Matplotlib · Seaborn

📄 **Full project report:** [`docs/Customer_Churn_Analysis_Report.pdf`](docs/Customer_Churn_Analysis_Report.pdf)

---

## 🔍 Project Overview

Telecommunication companies lose revenue when customers leave. This project analyzes **7,032 telecom customer records** to uncover why customers churn and deploys an ML model as a live web application for instant churn predictions.

---

## 📁 Project Structure

```
churn_flask_app/
│
├── app.py                  # Flask web application (prediction API)
├── streamlit_app.py        # Streamlit Cloud entry point (self-training, no Flask)
├── train_model.py          # Train & save the ML model (run this first)
├── churn_analysis.ipynb    # Full EDA + model training notebook
├── Churn_Analysis_sql.sql  # SQL business queries (MySQL)
├── requirements.txt        # Python dependencies
│
├── model/                  # Auto-generated after running train_model.py
│   ├── churn_model.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
│
├── templates/
│   └── index.html          # Frontend UI
│
├── data/
│   └── cleaned_churn_data.csv   # Place your dataset here
│
├── powerbi/
│   └── Churn_Analysis_Dashboard.pbix   # Power BI dashboard (slicer fix applied)
│
└── docs/
    └── Customer_Churn_Analysis_Report.pdf   # Full project report
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download from [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the raw CSV at:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```
Then run `churn_analysis.ipynb` top to bottom — it cleans the data, runs the EDA, trains the model, and exports `data/cleaned_churn_data.csv`.

### 4. Train the model (if you skipped the notebook)
```bash
python train_model.py
```

### 5. Run the Flask app
```bash
python app.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## ☁️ Deploying

This repo has **two** front ends to the same model, for two different hosting situations:

| File | Run with | Where it works |
|---|---|---|
| `app.py` | `python app.py` | Any Flask-friendly host (Render, Railway, Fly.io, PythonAnywhere) |
| `streamlit_app.py` | `streamlit run streamlit_app.py` | **Streamlit Community Cloud** |

Streamlit Cloud only runs apps built on the `streamlit` package — it does
**not** run Flask apps, regardless of what's in `requirements.txt`. Point
Streamlit Cloud's "Main file path" at `streamlit_app.py`, not `app.py`.

`streamlit_app.py` is self-contained: if no trained model exists yet, it
trains one on cold start from `data/cleaned_churn_data.csv` (which, unlike
the raw Kaggle CSV, **is** committed to this repo for exactly this reason)
and caches it for the life of the server process. No manual training step
needed before deploying.

---

## 📈 Key Findings

| Factor | Insight |
|--------|---------|
| Contract Type | Month-to-month customers churn most |
| Payment Method | Electronic check users show highest churn |
| Monthly Charges | Higher charges = higher churn probability |
| Tenure | New customers (< 12 months) are highest risk |
| Long-term Contracts | Two-year contracts retain customers best |

---

## 📊 KPI Metrics

| KPI | Value |
|-----|-------|
| Total Customers | 7,032 |
| Churned Customers | 1,869 |
| Churn Rate | 26.58% |
| Avg Monthly Charges | ₹64.80 |
| Model Accuracy | 78.32% |

---

## 🧠 ML Model Details

- **Algorithm:** Random Forest Classifier
- **Features:** 19 customer attributes
- **Train/Test Split:** 80/20
- **Class balancing:** `class_weight='balanced'` (handles churn imbalance)
- **Accuracy:** 78.32%

---

## 🗃️ SQL Analysis

5 business queries executed in MySQL Workbench (`Churn_Analysis_sql.sql`):
- Total & churned customer count
- Churn rate by contract type
- Churn by payment method
- Average monthly charges by churn status
- High-risk customer segments

---

## 📊 Power BI Dashboard

An interactive dashboard with 4 KPI cards, 3 analysis charts, and a contract-type slicer. The file is at `powerbi/Churn_Analysis_Dashboard.pbix`.

> The dashboard's slicer used to carry a hardcoded "Month-to-month only"
> filter, which understated every KPI card. That filter has been removed
> directly from the saved report state, so this file should now open
> against the full, unfiltered dataset (7,032 total / 1,869 churned /
> 26.58%). One thing still worth doing by hand: the Churn Rate (%) card
> displays a raw decimal instead of a percentage — right-click the field →
> Format → Percentage, 2 decimals — since that format string lives in the
> compressed data model and isn't something that can be patched from
> outside Power BI Desktop.
>
> After exporting a fresh screenshot, save it as `docs/dashboard.png` and
> embed it here with `![Dashboard](docs/dashboard.png)`.

---

## 💡 Business Recommendations

1. Promote long-term contracts with discount incentives
2. Improve experience for electronic check payment users
3. Proactively monitor customers with high monthly charges
4. Implement loyalty programs for customers in their first 12 months
5. Add tech support and online security bundles to reduce churn

---

## 📌 Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

*Prepared by Alekhya Viswanadhapalli*
