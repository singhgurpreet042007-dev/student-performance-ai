import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[[
    "study_hours",
    "attendance",
    "internal_marks",
    "preparation_level",
    "days_before_exam"
]]

# Target
y = data["final_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Advanced Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model R2 Score:", r2_score(y_test, y_pred))

# Save
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")