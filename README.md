# AI/ML Practical Application 2 – Used Car Price Modeling

This repository contains the code and analysis for **AI/ML Practical Application 2**, applying the **CRISP-DM framework** to build a regression model that predicts **used car prices** and extracts insights for a hypothetical used-car dealership.

The project uses a Kaggle-style `vehicles.csv` dataset and focuses on:
- Understanding **which factors drive price**.
- Building and evaluating **linear family models**, including polynomial regression.
- Demonstrating **feature engineering, PCA, and clustering (DBSCAN)**.

---

## 1. Business Problem

A used car dealership must decide how to **price incoming inventory** and which vehicles to **prioritize purchasing**. They want to know:

- Which characteristics (age, mileage, manufacturer, condition, fuel type, etc.) have the **strongest impact** on price?
- How can they **systematically estimate a fair price** for a given car?
- Can they identify **unusual or suspicious listings** that may need manual review?

**Goal:** Build a regression model that predicts **price** given vehicle attributes and use it to derive **actionable pricing and inventory recommendations**.

---

## 2. Data Description

- **File:** `vehicles.csv`
- **Rows:** Each row represents a single used vehicle listing.
- **Key columns:**
  - **Target**
    - `price` – asking price (USD)
  - **Core numeric**
    - `year` – model year  
    - `odometer` – mileage  
  - **Engineered features**
    - `car_age` – derived as `reference_year - year`  
    - `price_per_mile` – `price / odometer` (where odometer > 0)  
    - `log_price` – log-transformed price (for analysis)
  - **Categorical** (this analysis was limited to 
    - `manufacturer`, `model`, `condition`, `fuel`, `title_status`, `transmission`,
      `drive`, `size`, `type`, `paint_color`, `region`, `state`, etc.
    - This anaylysis was limited to `manufacturer`, `condition`, `size`

Basic cleaning steps:
- Remove rows with **missing or non-positive price**.
- Remove price outliers (below 1st and above 99th percentile).
- Remove rows with **missing odomoter reading**.
- Remove 'odometer' outliers (below 1st and above 99th percentile).
- Handle missing values in numeric and categorical features via **SimpleImputer**.

---

## 3. Methodology (CRISP-DM Summary)

### 3.1 Business Understanding

- Frame the problem as a **supervised regression** task.
- Connect model outputs to decisions

### 3.2 Data Understanding

- Inspect data types, missing values, and distributions.
- Explore relationships:
  - `car_age` vs price  
  - `odometer` vs price  
  - Price by condition, manufacturer, type, and region.

### 3.3 Data Preparation

- Target cleaning:
  - Drop invalid / missing prices; trim extreme outliers.
- Feature engineering:
  - `car_age`, `price_per_mile`, `log_price`.
- Feature selection:
  - Use `car_age` instead of raw `year` to avoid redundancy.
- Preprocessing pipelines:
  - **Numeric:** median imputation + StandardScaler  
  - **Categorical:** most frequent imputation + OneHotEncoder

### 3.4 Modeling

Models built and compared:

1. **Baseline Multiple Linear Regression**
2. **Ridge Regression**
3. **Polynomial Regression (degree 2)**:
   - Polynomial expansion of numeric features (`odometer`, `car_age`, `price_per_mile`) to degree 2.
   - One-hot encoded categorical features.
5. **PCA pipelines**
   - PCA on scaled numeric features combined with one-hot categoricals.


Evaluation metric:
- **Root Mean Squared Error (RMSE)** via cross-validation and a held-out test set.
- Also report **MAE** and **R²** for interpretability.

### 3.5 Evaluation (High-Level)

- Baseline Linear Regression & Ridge:
  - CV RMSE ≈ **10,958**
- Polynomial Regression (degree 2):
  - CV RMSE ≈ **9,375**
  - ~15% reduction in RMSE vs baseline linear model.
- PCA-based models:
  - **No meaningful improvement** compared to non-PCA models.
  - PCA + polynomial model performed worse, suggesting over-compression.

**Chosen final model:** **Degree-2 Polynomial Regression** (numeric polynomial + categorical one-hot).

---

## 4. Key Findings

- **Age and mileage** are the **dominant drivers** of price:
  - Older, high-mileage vehicles are significantly discounted.
- **Non-linear effects** matter:
  - The polynomial model captures curvature in the age and mileage relationships, explaining why it outperforms simple linear regression.
- **Manufacturer, type, and condition** have strong additional effects:
  - Some brands and vehicle types retain value better.
  - Better stated condition raises price substantially.
- **Region/state** effects capture local pricing differences.
- **PCA** did not improve performance for this problem, and in some settings hurt performance.

- We tested the following models:

Model	Cross-Val RMSE
Baseline Linear Regression	~9884
Ridge Regression	~9885
Polynomial Regression (degree 2)	~9363 (best pre-PCA)
PCA + Linear Regression	~9884
PCA + Poly Regression	~12056
PCA + Ridge Regression	~10958
Best Model: Polynomial Regression (Degree 2)

Test RMSE: ~9267
Test R²: 0.514
Test MAE: ~6613

Polynomial regression (degree-2) captures non-linear interactions between age and mileage that linear models cannot.


---

## 5. Recommendations

- Use the **degree-2 polynomial regression model** as the primary pricing aid.
- Prioritize purchasing:
  - Newer, low-mileage vehicles in high-value manufacturer/body-type segments.
- Incorporate model outputs in:
  - Acquisition decisions (what to buy, at what cost).
  - Pricing decisions (setting initial ask and discount bands).
- Use anomaly flags (DBSCAN “noise” + large model residuals) for:
  - Data quality checks.
  - Manual review of unusual listings.

(See notebook section **“Recommendations”** for a more detailed, business-oriented narrative.)

---

## 6. Repository Structure

```text
.
├── AI ML App 2.ipynb                # Main Jupyter notebook (analysis & modeling)
├── README.md                        # This file
└── data
    └──vehicles.csv                 # Source dataset
├── images                        
    └── crisp.png                   # Car Salesman Man
    └── kurt.png                    # CRISP_DM Model
