from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np


def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "Model", "churn_model.pkl")
        model = joblib.load("Model/churn_model.pkl")

        print("Model Başarıyla Yüklendi.")
        return model
    except Exception as e:
        print("model yüklenirken hata alındı!!",e)
        return None

app = FastAPI(title="Customer Prediction API",version="1.0.0")
model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
    gender: str  
    SeniorCitizen: int  
    Partner: str  
    Dependents: str  
    tenure: int 
    PhoneService: str  
    MultipleLines: str  
    InternetService: str  
    OnlineSecurity: str 
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingTV: str  
    StreamingMovies: str 
    Contract: str  
    PaperlessBilling: str 
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
        "Churn_Probability": float(probability[1]),  
        "No_Churn_Probability": float(probability[0]), 
        "risk_level": "HIGH" if probability[1] > 0.6 else "MEDIUM" if probability[1] > 0.4 else "LOW"
    }


