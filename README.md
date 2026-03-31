# Credit Risk Scoring

An end-to-end machine learning pipeline that predicts the probability of loan default, converts it into a credit score, and assigns a lending decision.

Built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset — 307,511 real loan applications.

## Live Demo

[Open the app](https://credit-risk-wissem.streamlit.app)

---

## Results

| Metric | Score |
|---|---|
| AUC | 0.775 |
| KS Statistic | 0.416 |
| Gini Coefficient | 0.551 |
| Brier Score | 0.067 |
| Calibration Error | 0.054 |

Champion model: **XGBoost** — selected by composite score across all metrics.

---

## What This Project Does

1. Loads and validates 8 raw CSV files from Home Credit
2. Engineers domain features — credit ratios, age, employment, external scores
3. Aggregates bureau and previous application data per applicant
4. Trains and compares 3 models — Logistic Regression, Random Forest, XGBoost
5. Selects the champion model and saves it to a timestamped registry
6. Converts predicted default probability into a credit score (300–850)
7. Assigns a lending decision — Approve, Manual Review, or Reject
8. Serves predictions via a REST API and an interactive Streamlit app
9. Monitors feature drift between training and production data

---

## Project Structure
```
CREDIT_RISK_SCORING/
├── configs/
│   ├── model.yaml                   # model hyperparameters
│   └── training.yaml                # training configuration
├── data/
│   ├── raw/                         # Kaggle CSVs — gitignored
│   └── processed/                   # generated features — gitignored
├── models/
│   └── feature_pipeline.joblib      # fitted preprocessing pipeline
├── models_artifacts/
│   └── xgboost_champion_<timestamp>/
│       ├── model.pkl
│       ├── metrics.json
│       ├── eval_predictions.csv
│       ├── predictions.csv
│       └── scorecard.csv
├── monitoring_reports/
│   └── data_drift_report.csv
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_experiments.ipynb
│   └── 06_inference_experiments.ipynb
├── src/
│   ├── api/main.py                  # FastAPI REST API
│   ├── data/
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── features/
│   │   ├── build.py
│   │   └── transformers.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── registry.py
│   │   ├── scorecard.py
│   │   └── train.py
│   ├── monitoring/
│   │   ├── data_drift.py
│   │   └── performance.py
│   └── pipelines/
│       ├── training_pipeline.py
│       ├── evaluation_pipeline.py
│       └── inference_pipeline.py
├── app.py                           # Streamlit demo app
├── requirements.txt
└── README.md
```

---

## ML Pipeline
```
Raw Data (8 CSV files from Home Credit)
        ↓
Data Ingestion & Validation
        ↓
Feature Engineering
  - Domain features: PAYMENT_RATE, ANNUITY_INCOME_PERC,
    CREDIT_INCOME_RATIO, AGE_YEARS, EMPLOYED_TO_AGE_RATIO
  - External scores: EXT_SOURCE_1/2/3 mean, std, min
  - Bureau aggregations: loan count, active credits, overdue count
  - Previous application aggregations: approved/refused counts
        ↓
Train / Eval Split — 80/20 stratified
        ↓
Model Training — Logistic Regression, Random Forest, XGBoost
        ↓
Champion Selection — composite score (KS + Gini + AUC - Calibration Error)
        ↓
Model Registry — timestamped artifact folder
        ↓
Scorecard — PD → Credit Score (300–850) → Decision
```

---

## Dataset

| File | Description | Rows |
|---|---|---|
| application_train.csv | Main loan applications with target | 307,511 |
| application_test.csv | Test applications | 48,744 |
| bureau.csv | Previous credits from other institutions | 1,716,428 |
| bureau_balance.csv | Monthly bureau credit balances | 27,299,925 |
| previous_application.csv | Previous Home Credit applications | 1,670,214 |
| installments_payments.csv | Payment history | 13,605,401 |
| POS_CASH_balance.csv | POS and cash loan balances | 10,001,358 |
| credit_card_balance.csv | Credit card balances | 3,840,312 |

Target: `1` = defaulted, `0` = repaid. Default rate ~8%.

---

## Installation
```bash
git clone https://github.com/wissemhammami/credit-risk-scoring.git
cd credit-risk-scoring
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Download the data

Download all CSV files from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place them in `data/raw/`.

### 2. Run the full training pipeline
```bash
python -m src.pipelines.training_pipeline
```

### 3. Run evaluation
```bash
python -m src.pipelines.evaluation_pipeline
```

### 4. Run inference
```bash
python -m src.pipelines.inference_pipeline
```

### 5. Run drift monitoring
```bash
python -m src.monitoring.data_drift
```

### 6. Launch Streamlit app
```bash
python -m streamlit run app.py
```

### 7. Launch REST API
```bash
python -m uvicorn src.api.main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

---

## API Example

**POST** `/predict`
```json
{
  "SK_ID_CURR": 100001,
  "AMT_INCOME_TOTAL": 150000,
  "AMT_CREDIT": 500000,
  "AMT_ANNUITY": 25000,
  "AMT_GOODS_PRICE": 450000,
  "DAYS_BIRTH": -12775,
  "DAYS_EMPLOYED": -1825,
  "DAYS_REGISTRATION": -1000,
  "DAYS_ID_PUBLISH": -1000,
  "CODE_GENDER": "M",
  "NAME_CONTRACT_TYPE": "Cash loans",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_HOUSING_TYPE": "House / apartment",
  "NAME_INCOME_TYPE": "Working",
  "EXT_SOURCE_1": 0.5,
  "EXT_SOURCE_2": 0.6,
  "EXT_SOURCE_3": 0.5
}
```

**Response**
```json
{
  "SK_ID_CURR": 100001,
  "PD": 0.043,
  "Score": 721,
  "Decision": "Approve"
}
```

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Feature Engineering | scikit-learn Pipeline, ColumnTransformer |
| API | FastAPI |
| Demo App | Streamlit |
| Model Registry | Custom timestamped registry |
| Drift Monitoring | KS Test, PSI |
| Config Management | YAML |

---

## Author

**Wissem Hammami**
- GitHub: [wissemhammami](https://github.com/wissemhammami)
