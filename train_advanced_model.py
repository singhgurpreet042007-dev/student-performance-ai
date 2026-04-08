
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("advanced_student_data.csv")

# Features and target
X = df[[
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_score",
    "social_media_usage",
    "stress_level",
    "preparation_level",
    "extra_activities"
]]
y = df["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model trained successfully!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save model + features together
with open("best_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("best_model.pkl saved successfully!")