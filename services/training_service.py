import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load Dataset
df = pd.read_csv("data/loan_data.csv")


# Encode Loan Purpose
purpose_encoder = LabelEncoder()
df["loan_purpose"] = purpose_encoder.fit_transform(df["loan_purpose"])


# Encode Risk
risk_encoder = LabelEncoder()
df["risk"] = risk_encoder.fit_transform(df["risk"])


# Features
X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "employment_years",
        "debt_ratio",
        "loan_purpose"
    ]
]


# Target
y = df["risk"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy : {accuracy:.2f}")


# Save Model
joblib.dump(model, "models/loan_risk_model.pkl")

joblib.dump(purpose_encoder, "models/purpose_encoder.pkl")

joblib.dump(risk_encoder, "models/risk_encoder.pkl")


print("Model Saved Successfully")


