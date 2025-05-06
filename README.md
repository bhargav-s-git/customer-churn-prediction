# Customer Churn Prediction

# 📉 Customer Churn Prediction

Predicting whether a customer will churn using machine learning, with a focus on business impact and model explainability.

---

## 🚀 Project Overview

Customer churn has a significant impact on revenue. This project aims to develop a predictive model to identify high-risk customers before they leave, enabling proactive retention strategies.

---

## 📊 Problem Statement

**Goal**: Classify whether a customer will churn based on behavioral and demographic attributes.

**Business Value**: Improve customer retention through targeted interventions, reducing churn-related losses.

---

## 🛠️ Tech Stack

- **Python**, **pandas**, **scikit-learn**, **matplotlib**, **seaborn**
- Jupyter notebooks for development and storytelling
- Feature engineering, model selection, evaluation metrics
- Placeholder for deployment: `Flask` or `FastAPI` (to be added)

---

## 📁 Project Structure

customer-churn-prediction/
├── data/ ← Raw dataset (not included in repo)
├── notebooks/ ← Jupyter notebooks for EDA and modeling
├── src/ ← Modular Python scripts (preprocessing, training)
├── outputs/ ← Figures, metrics, reports
├── requirements.txt ← Python dependencies
└── README.md ← You're here

---

## 📈 Features Used

- Customer demographics (gender, senior citizen, tenure)
- Service usage patterns (internet, phone, contract type)
- Billing and payment details (monthly charges, payment method)

---

## 🧠 Model Pipeline (Coming Soon)

- Data Preprocessing
- Feature Engineering
- Model Training: Logistic Regression, Random Forest, XGBoost
- Evaluation: ROC-AUC, Precision, Recall
- SHAP/Feature Importance (Explainability)

---

## 🔍 Model Explainability (SHAP)

To make model predictions interpretable, we used **SHAP (SHapley Additive Explanations)**:

- **Top Churn Drivers** (from SHAP summary plot):
  - Contract type (month-to-month)
  - Tenure (shorter customers more likely to churn)
  - High monthly charges

<p align="center">
  <img src="outputs/figures/shap_summary_plot.png" width="600" alt="SHAP Summary">
</p>

We also used SHAP **force plots** to explain individual predictions:  
🔗 [View Sample Force Plot (HTML)](outputs/figures/shap_force_plot.html)

These insights help translate model predictions into actionable business strategy.

---

## 📌 Key Insights (To Be Added)

- High churn among customers with month-to-month contracts
- Electronic check payment method associated with higher churn
- Long-tenured customers less likely to churn

---

## 🚧 TODOs

- [ ] Add preprocessing script in `src/`
- [ ] Implement ML pipeline and evaluation
- [ ] Include SHAP-based model explainability
- [ ] Deploy via API or Streamlit (optional)

---

## 📎 Dataset

- [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 👤 Author

Bhargav Somepalli  
[GitHub](https://github.com/bhargav-s-git) | [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN/)  
*Open to collaborations and freelance work in data science & analytics.*

---

## 📜 License

This project is licensed under the MIT License.
