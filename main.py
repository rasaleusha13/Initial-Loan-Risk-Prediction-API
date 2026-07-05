from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Loan Risk Prediction API",
    version="1.0.0"
)

# Load model and encoders
model = joblib.load("models/loan_risk_model.pkl")
purpose_encoder = joblib.load("models/purpose_encoder.pkl")
risk_encoder = joblib.load("models/risk_encoder.pkl")


class LoanApplication(BaseModel):
    income: float
    credit_score: int
    loan_amount: float
    employment_years: int
    debt_ratio: float
    loan_purpose: str


@app.get("/")
def home():
    return {
        "message": "Loan Risk Prediction API is Running"
    }


@app.post("/predict")
def predict(application: LoanApplication):

    purpose = purpose_encoder.transform(
        [application.loan_purpose]
    )[0]

    input_df = pd.DataFrame([{
        "income": application.income,
        "credit_score": application.credit_score,
        "loan_amount": application.loan_amount,
        "employment_years": application.employment_years,
        "debt_ratio": application.debt_ratio,
        "loan_purpose": purpose
    }])

    prediction = model.predict(input_df)[0]

    risk = risk_encoder.inverse_transform([prediction])[0]

    return {
        "predicted_risk": risk
    }
    