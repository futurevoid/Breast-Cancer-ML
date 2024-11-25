from tensorflow.keras.models import load_model  # type: ignore 
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved model
model = load_model('bestfit_model.keras')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Patient input
age = float(input("Enter patient's age: "))
year = float(input("Enter patient's year of operation: "))
nodes = float(input("Enter patient's number of positive axillary nodes: "))

# Example single input data
single_input = {'age': age, 'year': year, 'nodes': nodes}

# Convert the single input into a NumPy array
single_input_array = np.array([list(single_input.values())], dtype=np.float32)

# Apply scaling using the loaded scaler
single_input_scaled = scaler.transform(single_input_array)

# Debugging: Output scaled input
print(f"Scaled Input: {single_input_scaled}")

# Make the prediction
prediction_prob = model.predict(single_input_scaled)[0, 0]  # Probability output
prediction = (prediction_prob > 0.5).astype("int32")  # Threshold

# Debugging: Output prediction probability
print(f"Prediction Probability: {prediction_prob}")

# Interpret the prediction result
survival_status = "Survived 5 years or more" if prediction == 1 else "the patient died within 5 year"
print(f"Predicted Survival Status: {survival_status}")
