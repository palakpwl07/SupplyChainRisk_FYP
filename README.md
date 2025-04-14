# 🌍 Supply Chain Risk Assessment using Ensemble Machine Learning

**Author:** Palak Porwal  
**Institution:** Nanyang Technological University (NTU)  
**Thesis Title:** AI in Risk Management: Using AI for Proactive Risk Mitigation  
**Specialization:** Smart Manufacturing and Digital Factory  
**Date:** April 2025

---

## 📘 Project Overview

This project implements a **stacked ensemble learning model** to proactively assess supply chain risk — with a focus on predicting climate-related delivery delays using a data-driven approach. The pipeline integrates three base regression models: **Random Forest**, **Gradient Boosting**, and **XGBoost**, which are fused into a meta-model using **Ridge Regression** for superior generalization.

The goal is to provide **robust, interpretable, and business-usable insights** into risk likelihood, enabling early interventions and better resilience planning.

---

## 📂 Folder Structure

---

## 🧠 Key Concepts

- **Ensemble Modeling**: Stacked generalization using multiple ML regressors and a meta-learner
- **Business Feature Engineering**: Includes derived metrics such as inventory-to-lead-time ratio, resilience index, and volatility-risk interactions
- **Robust Evaluation**: Cross-validation with RMSE, MAE, and R² scores reported
- **Dimensionality Reduction**: PCA applied post-polynomial expansion for efficiency
- **Pipeline Architecture**: Fully modular `Pipeline` and `ColumnTransformer` structures for scalability

---

## 📊 Datasets Used

The following datasets were compiled, cleaned, and merged to form the `final_cleaned_dataset.csv`:

| Dataset Name | Description |
|--------------|-------------|
| `emdat_disaster_impact.csv` | Contains historical records of disaster occurrences and their socio-economic impacts (Source: EM-DAT) |
| `inform_climate_risk_2020_2025.csv` | INFORM Risk Index with sub-national scores related to climate vulnerability and exposure 

---
## 🔎 Live Dashboard

The risk assessment model is deployed in a **user-friendly Streamlit interface** that allows business stakeholders to:

- Interactively adjust feature values (e.g., inventory days, supplier reliability)
- View predicted risk scores in real-time
- Access simplified decision support visuals
- Use dropdown filters and sliders without needing to understand model internals

📍 **Try it now** → [https://scdashboard.streamlit.app/](https://scdashboard.streamlit.app/)

This is ideal for:
- Procurement teams
- Logistics managers
- Risk analysts and resilience strategists


## ⚙️ How to Run

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
