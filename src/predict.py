import numpy as np
from joblib import load

model = load("models/random_forest.pkl")
scaler = load("models/scaler.pkl")

sample = np.array([[1, 0, 3, 100, 200, 0.1, 0.01, 0.0]])  # Example traffic

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("Malicious" if prediction == 1 else "Normal")
