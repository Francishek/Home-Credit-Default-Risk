# DS.v2.5.3.4.1

# Risk Evaluation as a Service – POC

This project is a **proof-of-concept (POC)** for a startup idea: providing **risk evaluation as a service** for retail banks.
We use the [Home Credit dataset](https://www.kaggle.com/competitions/home-credit-default-risk/) to explore, model, and deploy machine learning solutions that predict loan default risk.

---

## Project Goals

* Build a **robust modeling pipeline** for credit risk evaluation.
* Test different **ML algorithms** (logistic regression, LightGBM, CatBoost, XGBoost, ensembles).
* Deliver a **deployable API** for real-time risk scoring.
* Provide **interpretability** through feature importance and SHAP analysis.

---

## Repository Structure

```
project-root/
│
├── README.md           # Project overview (what, why, how, results)
├── LICENSE             # License file (MIT)
├── requirements        # All dependencies
│
├── deploy/              # Deployment files (Flask app, Docker, configs)
│   ├── app.py
│   ├── requirements.txt
|   ├── Dockerfile
│   └── other files 
│
├── data/                # Data utils + results
│   ├── utils_modeling.py    
│   ├── utils_EDA.py 
│   ├── Optuna_lgbm_xgb_goggle_colab.ipynb
│   └── results            
│
├── eda/                 # Automated reports (Sweetviz, YData)
│   ├── EDA_for_additional_datasets/ 
│   ├── sweetviz_reports/
│   └── ydata_reports/
│
└── notebooks/           # Jupyter notebooks for main workflow
    ├── 01_EDA_train_test.ipynb
    ├── 02_Modeling.ipynb
    ├── 03_Investigation_plan.ipynb
    └── 04_POC.ipynb

```

## Setup

1. Clone the repo:

   Link [GitHub](https://github.com/Francishek/Home-Credit-Default-Risk) or
   ```bash
   git clone https://https://github.com/Francishek/Home-Credit-Default-Risk
   cd project-root
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Workflow

1. **EDA** (`01_EDA_train_test.ipynb`)

   * Explore dataset structure, distributions, missing values, correlations.
   * Automated reports generated in `eda/`.

2. **Modeling** (`02_Modeling.ipynb`)

   * Feature engineering, anomaly handling, aggregation.
   * Train/test split, multiple models, Optuna tuning.
   * Feature importance analysis.
  
3. **Investigation_plan** (`03_Investigation_plan.ipyn`)

   * Step-by-step plan, assumptions, findings.
   * Results from best models, key features, business insights.b

4. **POC Summary** (`04_POC.ipynb`)

   * Step-by-step plan, objectives,assumptions.

5. **Deployment** (`deploy/app.py`)

   * Flask API for risk scoring.
   * Supports JSON input of customer data.
   * Returns probability of default + prediction.

---

## Key Results

* Top-performing models: **LightGBM, XGBoost**
* Evaluation metric: **ROC-AUC** (\~0.76 baseline → improved to \~0.79 (test set) with feature engineering, Optuna hyperparameter tuning & ensembles).
* Most important features include:

  * `EXT_SOURCE_1/2/3`
  * `YEARS_BIRTH`
  * `YEARS_EMPLOYED`
  * `CODE_GENDER`
  * `AMT_ANNUITY`
  * `NAME_FAMILY_STATUS`
  * `INSTAL_LATE_FLAG`
---

## Run deployed model

```powershell
$url = "https://loan-prediction-api-153692760554.us-central1.run.app/predict"
$body = @'
{
    "application": [
        {
            "SK_ID_CURR": 100001,
            "DAYS_BIRTH": -12000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 450000,
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "REGION_POPULATION_RELATIVE": 0.01,
            "DAYS_EMPLOYED": -2000,
            "DAYS_REGISTRATION": -3000,
            "DAYS_ID_PUBLISH": -1000,
            "OWN_CAR_AGE": 5,
            "CNT_FAM_MEMBERS": 3,
            "CNT_CHILDREN": 1
        }
    ],
    "bureau": [],
    "bureau_balance": [],
    "prev_app": [],
    "installments": [],
    "credit_card_balance": [],
    "pos_cash": []
}
'@

$response = Invoke-WebRequest -Uri $url -Method Post -Body $body -ContentType "application/json"
$response.Content
```

```powershell
# Minimal required data
$response = Invoke-WebRequest -Uri "https://loan-prediction-api-153692760554.us-central1.run.app/predict" -Method Post -Body '{"application": [{"SK_ID_CURR": 100001, "DAYS_BIRTH": -12000}]}' -ContentType "application/json"
$response.Content
```

The model also can used by RESTED in Mozilla Firefox (how to use explained in POC notebook)

---

## Next Steps

* Extend feature engineering with domain knowledge.
* Add calibration and threshold optimization for production.
* Explore explainability dashboards (e.g., SHAP in Streamlit).

## Notebooks Structure:

### EDA Notebooks (additional datasets):

   1. Introduction
   
   2. Exploratory Data Analysis (EDA)

   3. Summary

### 01_EDA_train_test /  Home Credit Default Risk - EDA TRAIN / TEST

   1. Introduction

   2. Exploratory Data Analysis (EDA)

      A. Data loading & Initial overview

      B. Checking target variable (TARGET)

      C. Train vs Test datasets comparison

      D. Data Types

      E. Missing values

      F. Univariate and Bivariate analysis

      G. Statistical inference

      H. Detect Anomalies

      I. Feature Engineering

   3. Summary Train & Test Datasets

### 02_Modeling.ipynb / Home Credit Default Risk - Modeling Part

   1. Introduction to Modeling Part

   2. Data loading

   3. Best model selection

   4. Merging external data

   5. LightGBM and XGBoost training on merged dataset (284 features)

   6. LightGBM and XGBoost models performance on top 170 selected features and best params

   7. Ensembling

   8. Deploying the model

   9. Summary

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.

## Usage Policy

- **Educational Use**: Free to use for educational and research purposes
- **Commercial Use**: Contact author for commercial licensing
- **Attribution**: Please credit the author when using this API
- **Rate Limiting**: Please be respectful of server resources
- **No Guarantee**: Provided "as-is" without warranties

