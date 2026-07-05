import joblib
import pandas as pd


# Load saved model and encoders
model = joblib.load("models/loan_risk_model.pkl")
purpose_encoder = joblib.load("models/purpose_encoder.pkl")
risk_encoder = joblib.load("models/risk_encoder.pkl")


def predict_loan_risk(
    income,
    credit_score,
    loan_amount,
    employment_years,
    debt_ratio,
    loan_purpose
):

    # Encode loan purpose
    purpose = purpose_encoder.transform([loan_purpose])[0]

    # Create DataFrame
    input_data = pd.DataFrame(
        [[
            income,
            credit_score,
            loan_amount,
            employment_years,
            debt_ratio,
            purpose
        ]],
        columns=[
            "income",
            "credit_score",
            "loan_amount",
            "employment_years",
            "debt_ratio",
            "loan_purpose"
        ]
    )

    # Predict
    prediction = model.predict(input_data)[0]

    risk = risk_encoder.inverse_transform([prediction])[0]

    return {
        "predicted_risk": risk
    }
    