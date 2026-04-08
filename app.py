import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from auth import init_db, create_user, verify_user, save_prediction, get_user_history

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Study Planner Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# DATABASE INIT
# -------------------------
init_db()

# -------------------------
# SESSION STATE
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "predicted_score" not in st.session_state:
    st.session_state.predicted_score = None

# -------------------------
# LOAD / TRAIN MODEL
# -------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("advanced_student_data.csv")

    features = [
        "study_hours",
        "attendance",
        "sleep_hours",
        "previous_score",
        "social_media_usage",
        "stress_level",
        "preparation_level",
        "extra_activities"
    ]
    target = "final_score"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model, features

model, features = load_model()

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
/* Global */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 25%),
        radial-gradient(circle at bottom right, rgba(168,85,247,0.18), transparent 25%),
        linear-gradient(135deg, #020617 0%, #081120 45%, #0f172a 100%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081120, #0b1220, #111827);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Card styles */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

.glass-card-strong {
    background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(124,58,237,0.14));
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(18px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.38);
}

/* Auth shell */
.auth-shell {
    max-width: 1100px;
    margin: 0 auto 1.2rem auto;
    padding: 24px;
    border-radius: 28px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.45);
}

.auth-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    word-break: break-word;
    overflow-wrap: anywhere;
}

.auth-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
    line-height: 1.7;
}

.feature-chip {
    display: inline-block;
    margin: 6px 8px 0 0;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e2e8f0;
    font-size: 0.92rem;
}

/* Hero */
.hero-wrap {
    width: 100%;
    padding: 12px 8px 4px 8px;
}

.hero-title {
    font-size: clamp(1.9rem, 3.6vw, 3.8rem);
    font-weight: 900;
    line-height: 1.2;
    text-align: center;
    margin: 0 auto 0.6rem auto;
    width: 100%;
    max-width: 1050px;
    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.08rem;
    margin: 0 auto 1rem auto;
    max-width: 950px;
    line-height: 1.7;
    padding: 0 12px;
}

.hero-badge-wrap {
    text-align: center;
    margin-bottom: 2rem;
}

.hero-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    color: #dbeafe;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 0 30px rgba(96,165,250,0.18);
    font-weight: 700;
}

/* Labels & inputs */
label, .stMarkdown, p, div {
    color: inherit;
}

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.85rem 1.1rem;
    font-size: 1rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    box-shadow: 0 10px 25px rgba(37,99,235,0.35);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(124,58,237,0.42);
}

/* Section heading */
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 1.4rem 0 1rem 0;
    color: #f8fafc;
}

/* Result */
.result-score {
    font-size: clamp(2.1rem, 4vw, 3.2rem);
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.3rem 0;
    word-break: break-word;
}

.risk-low {
    color: #22c55e;
    font-size: 1.15rem;
    font-weight: 800;
}

.risk-medium {
    color: #facc15;
    font-size: 1.15rem;
    font-weight: 800;
}

.risk-high {
    color: #ef4444;
    font-size: 1.15rem;
    font-weight: 800;
}

/* Footer */
.footer {
    margin-top: 2rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.95rem;
    padding: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN / SIGNUP PAGE
# -------------------------
if not st.session_state.logged_in:
    st.markdown("""
        <div class="auth-shell">
            <div class="auth-title">AI Study Planner Pro</div>
            <div class="auth-subtitle">
                Premium AI-powered student performance prediction dashboard with risk analysis,
                smart recommendations, prediction history, and academic insights.
            </div>
            <span class="feature-chip">🚀 Predict Performance</span>
            <span class="feature-chip">⚠️ Risk Detection</span>
            <span class="feature-chip">📈 Analytics</span>
            <span class="feature-chip">🧠 Smart Suggestions</span>
            <span class="feature-chip">🔐 Secure Login</span>
        </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("""
        <div class="glass-card-strong">
            <h3 style="margin-top:0;">Why this dashboard feels premium</h3>
            <p style="color:#cbd5e1; line-height:1.7;">
                This is not a basic college form. It is a futuristic student analytics dashboard
                designed to look like a startup-level AI product.
            </p>
            <ul style="color:#cbd5e1; line-height:1.9;">
                <li>Smart score prediction using machine learning</li>
                <li>Low / Medium / High risk analysis</li>
                <li>Prediction history for each user</li>
                <li>Personalized improvement recommendations</li>
                <li>What-if study hours analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Login", "✨ Signup"])

        with tab1:
            st.markdown("### Welcome Back")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login to Dashboard"):
                if verify_user(login_user, login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = login_user.strip()
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            st.markdown("### Create Your Account")
            new_user = st.text_input("Create Username", key="signup_user")
            new_pass = st.text_input("Create Password", type="password", key="signup_pass")

            if st.button("Create Premium Account"):
                ok, msg = create_user(new_user, new_pass)
                if ok:
                    st.success(msg + " Now login.")
                else:
                    st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("## ⚡ Control Panel")
    st.success(f"Logged in as: {st.session_state.username}")
    st.markdown("---")
    st.markdown("### System Highlights")
    st.write("• Performance Prediction")
    st.write("• Risk Analysis")
    st.write("• Recommendations")
    st.write("• Feature Importance")
    st.write("• Prediction History")
    st.markdown("---")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.predicted_score = None
        st.rerun()

# -------------------------
# HERO
# -------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">AI-Powered Student Performance Prediction System</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-subtitle">
    Predict. Analyze. Improve. A futuristic academic intelligence dashboard for smarter student decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-badge-wrap">
    <span class="hero-badge">🚀 Next-Gen Academic Intelligence</span>
</div>
""", unsafe_allow_html=True)

# -------------------------
# FEATURE CARDS
# -------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0;">📊 Performance Prediction</h4>
        <p style="color:#cbd5e1;">Estimate final score using academic and behavioral inputs.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0;">🚨 Risk Detection</h4>
        <p style="color:#cbd5e1;">Understand whether the student falls into low, medium, or high risk.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0;">💡 Smart Recommendations</h4>
        <p style="color:#cbd5e1;">Get personalized suggestions for better academic performance.</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# INPUT SECTION
# -------------------------
st.markdown('<div class="section-title">📥 Student Input</div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 5.0)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
        sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
        previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 65.0)

    with b:
        social_media_usage = st.slider("Social Media Usage (hrs/day)", 0.0, 10.0, 2.0)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        preparation_level = st.slider("Preparation Level (1-5)", 1, 5, 3)
        extra_activities = st.slider("Extra Activities (hrs/week)", 0, 20, 5)

    predict_btn = st.button("🔍 Predict Performance")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if predict_btn:
        input_data = pd.DataFrame(
            [[
                study_hours,
                attendance,
                sleep_hours,
                previous_score,
                social_media_usage,
                stress_level,
                preparation_level,
                extra_activities
            ]],
            columns=features
        )

        prediction = model.predict(input_data)
        predicted_score = float(prediction[0])
        st.session_state.predicted_score = predicted_score

        if predicted_score >= 75:
            risk = "Low Risk ✅"
            risk_class = "risk-low"
        elif predicted_score >= 50:
            risk = "Medium Risk ⚠️"
            risk_class = "risk-medium"
        else:
            risk = "High Risk ❌"
            risk_class = "risk-high"

        save_prediction(
            st.session_state.username,
            study_hours,
            attendance,
            predicted_score,
            risk
        )

        st.markdown(f"""
        <div class="glass-card-strong">
            <div style="color:#cbd5e1; font-size:1.05rem;">Predicted Final Score</div>
            <div class="result-score">{predicted_score:.2f}</div>
            <div class="{risk_class}">{risk}</div>
            <p style="color:#cbd5e1; margin-top:10px; line-height:1.7;">
                AI-generated score based on academic and behavioral attributes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        score_bar = int(max(0, min(100, predicted_score)))
        st.progress(score_bar)

    else:
        st.markdown("""
        <div class="glass-card-strong">
            <h3 style="margin-top:0;">Prediction Panel</h3>
            <p style="color:#cbd5e1; line-height:1.7;">
                Fill the student details and click <b>Predict Performance</b> to view score,
                risk level, analytics, and recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# RECOMMENDATIONS
# -------------------------
if st.session_state.predicted_score is not None:
    st.markdown('<div class="section-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)

    recs = []

    if study_hours < 4:
        recs.append("Increase study hours by 1–2 hours daily.")
    if attendance < 70:
        recs.append("Improve class attendance for better subject understanding.")
    if sleep_hours < 6:
        recs.append("Maintain 7–8 hours of sleep for better focus and memory.")
    if social_media_usage > 4:
        recs.append("Reduce social media usage to improve concentration.")
    if stress_level > 7:
        recs.append("Practice stress management through planning, exercise, or meditation.")
    if preparation_level < 3:
        recs.append("Start earlier preparation and maintain regular revision.")
    if not recs:
        recs.append("Excellent balance. Maintain the same routine and consistency.")

    cols = st.columns(min(3, len(recs)))
    for i, rec in enumerate(recs):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">✅ Action Tip</h4>
                <p style="color:#cbd5e1; line-height:1.7;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.markdown('<div class="section-title">🔍 Feature Importance</div>', unsafe_allow_html=True)

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))
else:
    st.info("Feature importance is not available for this model.")

# -------------------------
# WHAT-IF ANALYSIS
# -------------------------
st.markdown('<div class="section-title">📈 What-If Analysis: Study Hours Impact</div>', unsafe_allow_html=True)

if st.session_state.predicted_score is not None and "study_hours" in features:
    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            sleep_hours,
            previous_score,
            social_media_usage,
            stress_level,
            preparation_level,
            extra_activities
        ]],
        columns=features
    )

    hours_range = list(range(0, 13))
    scores = []
    temp_df = input_data.copy()

    for h in hours_range:
        temp_df["study_hours"] = h
        predicted = model.predict(temp_df)[0]
        scores.append(predicted)

    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(hours_range, scores, linewidth=3)
    plt.xlabel("Study Hours")
    plt.ylabel("Predicted Score")
    plt.title("Impact of Study Hours on Predicted Performance")
    plt.grid(alpha=0.25)
    st.pyplot(fig)
else:
    st.info("Make a prediction first to unlock what-if analysis.")

# -------------------------
# AI ACADEMIC ASSISTANT
# -------------------------
st.markdown("---")
st.markdown('<div class="section-title">🤖 AI Academic Assistant</div>', unsafe_allow_html=True)

if st.session_state.predicted_score is not None:
    user_question = st.text_input("Ask something about improving performance:")

    if user_question:
        score = st.session_state.predicted_score
        question = user_question.lower()

        if "improve" in question:
            response = "Increase study hours, improve attendance, revise regularly, and reduce distractions."
        elif "sleep" in question:
            response = "Sleep 7–8 hours daily. Good sleep improves concentration and memory retention."
        elif "stress" in question:
            response = "Manage stress with breaks, exercise, meditation, and better planning."
        elif "attendance" in question:
            response = "Try to maintain attendance above 75% for better classroom continuity."
        elif "social" in question:
            response = "Reduce social media time and use focused study sessions."
        elif score < 50:
            response = "Your overall performance is low right now. Major improvement is needed in consistency."
        elif score < 75:
            response = "You are in the average range. Better routine and revision can raise your score."
        else:
            response = "Great performance. Maintain your routine and keep tracking your progress."

        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top:0;">🤖 Assistant Response</h4>
            <p style="color:#cbd5e1; line-height:1.7;">{response}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Please predict performance first.")

# -------------------------
# HISTORY
# -------------------------
st.markdown('<div class="section-title">📜 My Past Predictions</div>', unsafe_allow_html=True)

history = get_user_history(st.session_state.username)

if history:
    history_df = pd.DataFrame(
        history,
        columns=["Study Hours", "Attendance", "Predicted Score", "Risk"]
    )
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No history yet. Make a prediction first.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, Python, SQLite, and Machine Learning<br>
    Premium Academic Dashboard UI
</div>
""", unsafe_allow_html=True)