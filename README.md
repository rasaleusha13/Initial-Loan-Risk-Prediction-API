# 🏦 Loan Risk Prediction API

A production-style Machine Learning API built using **Python**, **FastAPI**, and **Scikit-learn** to predict the risk level of loan applicants based on financial and credit-related information.

---

## 📌 Project Overview

Financial institutions evaluate loan applications using multiple customer attributes such as income, credit score, debt ratio, employment history, and loan purpose.

This project demonstrates an end-to-end Machine Learning workflow by:

- Training a Loan Risk Prediction model
- Saving the trained model using Joblib
- Exposing predictions through a FastAPI REST API
- Returning loan risk as Low, Medium, or High

---

## 🚀 Tech Stack

- Python
- FastAPI
- Pandas
- Scikit-learn
- Joblib
- Git

---

## 📂 Project Structure

```
loan-risk-prediction-api/

│
├── data/
│   └── loan_data.csv
│
├── models/
│   ├── loan_risk_model.pkl
│   ├── purpose_encoder.pkl
│   └── risk_encoder.pkl
│
├── services/
│   ├── training_service.py
│   └── prediction_service.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Features

- Loan Risk Prediction
- Machine Learning Model Training
- Random Forest Classifier
- Model Serialization using Joblib
- REST API using FastAPI
- Interactive Swagger Documentation
- Production-style Project Structure

---

## 🧠 Machine Learning Workflow

```
Loan Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Random Forest Model
      │
      ▼
Model Serialization
      │
      ▼
FastAPI REST API
      │
      ▼
Risk Prediction
```

---

## 📊 Input Features

| Feature | Description |
|----------|-------------|
| Income | Applicant Annual Income |
| Credit Score | Customer Credit Score |
| Loan Amount | Requested Loan Amount |
| Employment Years | Years of Employment |
| Debt Ratio | Debt-to-Income Ratio |
| Loan Purpose | Purpose of Loan |

---

## 📡 API Endpoint

### POST /predict

### Sample Request

```json
{
  "income": 90000,
  "credit_score": 780,
  "loan_amount": 180000,
  "employment_years": 8,
  "debt_ratio": 0.22,
  "loan_purpose": "Home"
}
```

### Sample Response

```json
{
  "predicted_risk": "Low"
}
```

---

## ▶️ Run the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Start the API

```bash
python -m uvicorn main:app --reload
```

Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 📷 API Documentation

FastAPI automatically generates interactive Swagger documentation for testing all API endpoints.

---

## 🎯 Future Improvements

- Model Performance Metrics
- Feature Importance Visualization
- Input Validation
- Logging
- Unit Testing
- Docker Support

---

## 👨‍💻 Author

**Usha Rasale**

Senior Generative AI / Machine Learning Engineer

GitHub: https://github.com/rasaleusha13
