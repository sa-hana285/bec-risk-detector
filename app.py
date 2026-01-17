from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware


# import preprocessing logic
from preprocess import (
    compute_urgency_score,
    fin_intent,
    role_action_mismatch
)

app = FastAPI(title="BEC Risk Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (safe for demo)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS
    allow_headers=["*"],
)
# -------------------------
# Request schema
# -------------------------
class EmailRequest(BaseModel):
    email_body: str
    sender_role: str

# -------------------------
# Load trained model bundle
# -------------------------
model_bundle = joblib.load("bec_model.pkl")

model = model_bundle["model"]
fin_stems = model_bundle["fin_stems"]
feature_cols = model_bundle["feature_cols"]

# -------------------------
# Health check
# -------------------------
@app.get("/")
def home():
    return {
        "message": "BEC Detection API is running",
        "model_loaded": True
    }

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict_email_risk(request: EmailRequest):

    # ---- Feature extraction (SAME AS TRAINING) ----
    urgency_score = compute_urgency_score(request.email_body)
    financial_intent = fin_intent(request.email_body)
    role_mismatch = role_action_mismatch(
        request.email_body,
        request.sender_role
    )

    X = [[urgency_score, financial_intent, role_mismatch]]

    # ---- Isolation Forest prediction ----
    anomaly_pred = model.predict(X)[0]          # -1 or 1
    anomaly_flag = 1 if anomaly_pred == -1 else 0
    anomaly_score = float(model.decision_function(X)[0])

    # ---- Risk logic (API version of assign_risk_level) ----
    reasons = []

    if role_mismatch == 1 and financial_intent > 0:
        risk_level = "HIGH"
        reasons.append(
            "Financial request from a role that normally does not handle payments"
        )

    else:
        if anomaly_flag == 1:
            reasons.append(
                "Email behavior deviates from normal internal patterns"
            )

        if urgency_score > 0.1:
            reasons.append(
                "Unusual urgency detected in the email"
            )

        if reasons:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            reasons.append(
                "No suspicious behavioral patterns detected"
            )

    # ---- Response ----
    return {
        "risk_level": risk_level,
        "anomaly_flag": anomaly_flag,
        "anomaly_score": round(anomaly_score, 4),
        "features": {
            "urgency_score": round(urgency_score, 4),
            "financial_intent": round(financial_intent, 4),
            "role_mismatch": role_mismatch
        },
        "reasons": reasons
    }
