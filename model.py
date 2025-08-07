import pandas as pd
import numpy as np

df = pd.read_csv("D:\YAZILIMLARIM\PythonProjelerim\Churn\Veri\Customer-Churn.csv")


categorical1 = ["Partner","Dependents","PhoneService","PaperlessBilling","Churn"]
categorical2 = ["StreamingMovies","StreamingTV","TechSupport","DeviceProtection","OnlineBackup","OnlineSecurity"]

for col1 in categorical1:
    df[col1] = df[col1].map({"Yes": 1,"No": 0})

for col2 in categorical2:
    df[col2] = df[col2].map({"Yes": 1,"No internet service": 0,"No":0})

df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1,"No phone service": 0,"No":0})
df["InternetService"] = df["InternetService"].map({"DSL": 1,"Fiber optic": 1,"No":0})
df["gender"] = df["gender"].map({"Female": 1,"Male": 0})
df["Contract"] = df["Contract"].map({"Month-to-month": 0,"One year": 1,"Two year":2})
df = pd.get_dummies(df, columns=['PaymentMethod'], prefix='PaymentMethod')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df = df.dropna(subset="TotalCharges")
df = df.drop(columns = "customerID")

bol1 = ["PaymentMethod_Bank transfer (automatic)","PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check"]
for b in bol1:
    df[b] = df[b].astype(int)


from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,precision_recall_curve

x = df.drop(columns = "Churn")
y = df["Churn"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from collections import Counter
print(Counter(y))

scale_weight = (y == 0).sum() / (y == 1).sum()


param_grid = {
    'n_estimators': [100],             # Ağaç sayısı
    'max_depth': [3, 5],                    # Ağaçların maksimum derinliği
    'learning_rate': [0.01, 0.1],           # Öğrenme hızı
    'subsample': [0.8, 1],                  # Her ağaç için kullanılacak veri oranı
    'colsample_bytree': [0.8, 1],           # Her ağaçta kullanılacak özellik oranı
    'gamma': [0, 1],                        # Minimum kayıp azaltma (regularizasyon)
    'reg_alpha': [0, 0.1],                  # L1 regularizasyon
    'reg_lambda': [1, 1.5],                 # L2 regularizasyon
}

model_xgb1 = xgb.XGBClassifier(eval_metric="logloss",scale_pos_weight= scale_weight,random_state=0)

grid_search  =GridSearchCV(
    estimator=model_xgb1,
    param_grid = param_grid,
    scoring="f1",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(x_train,y_train)
best_model1 = grid_search.best_estimator_

y_pred_proba = best_model1.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold}")

y_pred_custom = (y_pred_proba > optimal_threshold).astype(int)

AS = accuracy_score(y_pred_custom,y_test)
PS = precision_score(y_pred_custom,y_test)
f1 = f1_score(y_pred_custom,y_test)
RC = recall_score(y_pred_custom,y_test)

print(f"Scale Weight:\nF1 SCORE : {f1}\nAccuracy Score : {AS}\nPrecision Score : {PS}\nRecall Score : {RC}")
print("*"*100,"\n")
print(x.columns)

import joblib

joblib.dump(best_model1, "Model/churn_model.pkl")