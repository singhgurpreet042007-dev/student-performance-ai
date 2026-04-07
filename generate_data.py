import pandas as pd
import numpy as np

np.random.seed(42)

num_students = 1000

study_hours = np.random.uniform(0.5, 10, num_students)
attendance = np.random.uniform(40, 100, num_students)
sleep_hours = np.random.uniform(4, 9, num_students)
previous_score = np.random.uniform(30, 95, num_students)
social_media_usage = np.random.uniform(0, 6, num_students)
stress_level = np.random.uniform(1, 10, num_students)
preparation_level = np.random.randint(1, 6, num_students)
extra_activities = np.random.randint(0, 5, num_students)

# Realistic scoring formula
final_score = (
    study_hours * 5 +
    attendance * 0.3 +
    sleep_hours * 2 +
    previous_score * 0.4 -
    social_media_usage * 2 -
    stress_level * 1.5 +
    preparation_level * 3 -
    extra_activities * 1.2
)

# Add noise
final_score = final_score + np.random.normal(0, 5, num_students)

# Limit score between 0-100
final_score = np.clip(final_score, 0, 100)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "previous_score": previous_score,
    "social_media_usage": social_media_usage,
    "stress_level": stress_level,
    "preparation_level": preparation_level,
    "extra_activities": extra_activities,
    "final_score": final_score
})

data.to_csv("advanced_student_data.csv", index=False)

print("Advanced dataset generated successfully!")