import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Example input
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

# Convert to array
input_data = np.array([[study_hours, attendance]])

# Prediction
prediction = model.predict(input_data)

print("Predicted Final Score:", prediction[0])