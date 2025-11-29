import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
from preprocess import preprocess_data

df = pd.read_csv("data/nsl_kdd.csv")

X, y, scaler = preprocess_data(df)

model = RandomForestClassifier(n_estimators=150)
model.fit(X, y)

pred = model.predict(X)
print(classification_report(y, pred))

dump(model, "models/random_forest.pkl")
dump(scaler, "models/scaler.pkl")

print("Model training complete.")
