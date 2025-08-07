from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


def load_model():
    try:
        model = joblib.load("D:\YAZILIMLARIM\PythonProjelerim\Churn\Model\churn_model.pkl")
        print("Model Başarıyla Yüklendi.")
        return model
    except Exception as e:
        print("model yüklenirken hata alındı!!",e)
        return None

app = FastAPI(title="Customer Prediction API",version="1.0.0")
model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend adresini izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Message":"Churn prediction API is running",
            "Model_loaded":model is not None}

@app.get("/model-info")
def model_info():
    if model is None:
        return {"error" : "Mddel Error!!"}
    return {"model type": str(type(model)),
            "Features Count":model.n_features_in_,
            "Status":"Model Hazır"}

class CustomerInput(BaseModel):
    gender: str  # "Male" veya "Female"
    SeniorCitizen: int  # 0 veya 1
    Partner: str  # "Yes" veya "No"
    Dependents: str  # "Yes" veya "No"
    tenure: int  # ay sayısı
    PhoneService: str  # "Yes" veya "No"
    MultipleLines: str  # "Yes", "No", "No phone service"
    InternetService: str  # "DSL", "Fiber optic", "No"
    OnlineSecurity: str  # "Yes", "No", "No internet service"
    OnlineBackup: str  # "Yes", "No", "No internet service"
    DeviceProtection: str  # "Yes", "No", "No internet service"
    TechSupport: str  # "Yes", "No", "No internet service"
    StreamingTV: str  # "Yes", "No", "No internet service"
    StreamingMovies: str  # "Yes", "No", "No internet service"
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" veya "No"
    MonthlyCharges: float
    TotalCharges: float
    PaymentMethod: str


def preprocess_customer_data(customer: CustomerInput):

    data = customer.dict()
    df = pd.DataFrame([data])

    categorical1 = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col1 in categorical1:
        df[col1] = df[col1].map({"Yes": 1, "No": 0})

    categorical2 = ["StreamingMovies", "StreamingTV", "TechSupport",
                    "DeviceProtection", "OnlineBackup", "OnlineSecurity"]
    for col2 in categorical2:
        df[col2] = df[col2].map({"Yes": 1, "No internet service": 0, "No": 0})

    df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1, "No phone service": 0, "No": 0})
    df["InternetService"] = df["InternetService"].map({"DSL": 1, "Fiber optic": 1, "No": 0})
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
    df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

    df = pd.get_dummies(df, columns=['PaymentMethod'], prefix='PaymentMethod')

    required_payment_columns = [
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    for col in required_payment_columns:
        if col not in df.columns:
            df[col] = 0

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    expected_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
                        'TotalCharges', 'PaymentMethod_Bank transfer (automatic)',
                        'PaymentMethod_Credit card (automatic)',
                        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    df = df[expected_columns]

    return df


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model Yüklenemedi!!")

    input_data = preprocess_customer_data(customer)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    return {
        "Churn_Prediction": int(prediction),
        "Churn_Probability": float(probability[1]),  # ✅ probability[1] = churn olasılığı
        "No_Churn_Probability": float(probability[0]),  # Bonus: churn olmama olasılığı
        "risk_level": "HIGH" if probability[1] > 0.6 else "MEDIUM" if probability[1] > 0.4 else "LOW"
    }


