import streamlit as st
import pandas as pd
import pickle
from auth import init_db, create_user, verify_user, save_prediction, get_user_history

# -------------------------
# 🔐 Login System
# -------------------------
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(login_user, login_pass):
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account"):
            ok, msg = create_user(new_user, new_pass)
            if ok:
                st.success(msg + " Now login.")
            else:
                st.error(msg)

    st.stop()
# -------------------------
# Load Model
# -------------------------
model, features = pickle.load(open("best_model.pkl", "rb"))

st.set_page_config(page_title="AI Student Performance System", layout="centered")

st.title("🎓 AI-Powered Student Performance Prediction System")

# -------------------------
# 🔓 Logout Button
# -------------------------
st.sidebar.success(f"Logged in as: {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# -------------------------
# Student Input Section
# -------------------------
st.header("📥 Student Input")

study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 65.0)
social_media_usage = st.slider("Social Media Usage (hrs/day)", 0.0, 10.0, 2.0)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
preparation_level = st.slider("Preparation Level (1-5)", 1, 5, 3)
extra_activities = st.slider("Extra Activities (hrs/week)", 0, 20, 5)

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict Performance"):

    input_data = pd.DataFrame([[study_hours,
                                attendance,
                                sleep_hours,
                                previous_score,
                                social_media_usage,
                                stress_level,
                                preparation_level,
                                extra_activities]],
                              columns=features)

    prediction = model.predict(input_data)
    predicted_score = prediction[0]

    st.session_state.predicted_score = predicted_score

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Final Score: {predicted_score:.2f}")

    # -------------------------
    # Risk Level
    # -------------------------
    if predicted_score >= 75:
        risk = "Low Risk ✅"
    elif predicted_score >= 50:
        risk = "Medium Risk ⚠️"
    else:
        risk = "High Risk ❌"

    st.subheader("🚨 Risk Analysis")
    st.info(f"Risk Level: {risk}")
    # ✅ Save prediction to DB (History)
    save_prediction(
    st.session_state.username,
    study_hours,
    attendance,
    predicted_score,
    risk
)

    # -------------------------
    # Recommendations
    # -------------------------
    st.subheader("💡 Personalized Recommendations")

    if study_hours < 4:
        st.write("✔ Increase study hours by 1–2 hrs daily.")
    if attendance < 70:
        st.write("✔ Improve class attendance.")
    if sleep_hours < 6:
        st.write("✔ Improve sleep schedule (7–8 hrs recommended).")
    if social_media_usage > 4:
        st.write("✔ Reduce social media usage.")
    if stress_level > 7:
        st.write("✔ Practice stress management (meditation/exercise).")

         # -------------------------
    # 🔍 Feature Importance
    # -------------------------
    st.subheader("🔍 Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))   

        # -------------------------
    # 📈 What-If Analysis (Study Hours vs Score)
    # -------------------------
    st.subheader("📈 What-If Analysis: Study Hours Impact")

    if "study_hours" in features:
        hours_range = list(range(0, 13))
        scores = []

        temp_df = input_data.copy()

        for h in hours_range:
            temp_df["study_hours"] = h
            predicted = model.predict(temp_df)[0]
            scores.append(predicted)

        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(hours_range, scores)
        plt.xlabel("Study Hours")
        plt.ylabel("Predicted Score")
        plt.title("Impact of Study Hours on Performance")

        st.pyplot(fig)

# -------------------------
# AI Academic Assistant
# -------------------------
st.markdown("---")
st.header("🤖 AI Academic Assistant")

if "predicted_score" in st.session_state:

    user_question = st.text_input("Ask something about improving performance:")

    if user_question:

        score = st.session_state.predicted_score   # ✅ IMPORTANT LINE
        question = user_question.lower()

        if "improve" in question:
            response = "Increase study hours, improve attendance, and reduce distractions."

        elif "sleep" in question:
            response = "Sleep 7-8 hours daily. Proper sleep improves memory retention."

        elif "stress" in question:
            response = "Practice meditation and proper time management."

        elif "attendance" in question:
            response = "Maintain attendance above 75% for better results."

        elif "social" in question:
            response = "Reduce social media usage to improve focus."

        elif score < 50:
            response = "Your overall performance is low. Major improvement is needed."

        elif score < 75:
            response = "You are average. Improve consistency and revision."

        else:
            response = "Great performance! Maintain your routine."

        st.write("🤖:", response)

else:
    st.warning("Please predict performance first.")

  # -------------------------
# 📜 Prediction History
# -------------------------
st.subheader("📜 My Past Predictions")

history = get_user_history(st.session_state.username)

if history:
    history_df = pd.DataFrame(
        history,
        columns=["Study Hours", "Attendance", "Predicted Score", "Risk"]
    )
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No history yet. Make a prediction first.")