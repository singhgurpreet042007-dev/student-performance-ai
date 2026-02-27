import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("advanced_student_data.csv")

# EXACT features jo CSV me hain
X = df[[
    'study_hours',
    'attendance',
    'sleep_hours',
    'previous_score',
    'social_media_usage',
    'stress_level',
    'preparation_level',
    'extra_activities'
]]

y = df['final_score']

model = RandomForestRegressor()
model.fit(X, y)

# Save model + correct feature order
pickle.dump((model, X.columns.tolist()), open("best_model.pkl", "wb"))

print("✅ Advanced Model trained successfully!")