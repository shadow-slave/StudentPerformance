import streamlit as st
import pandas as pd
import sqlite3
import joblib
import shap
import google.generativeai as genai
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Univ. AI Portal", layout="wide", page_icon="🎓")
genai.configure(api_key="Swantham Key ideda bunde") 

# Load assets
try:
    model = joblib.load("student_grade_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# --- CUSTOM CSS (MODERN UI) ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* Metrics */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4F8BF9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; }
    
    /* Result Boxes */
    .pred-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 15px;
    }
    .pred-good { background: linear-gradient(135deg, #28a745, #20c997); }
    .pred-bad { background: linear-gradient(135deg, #dc3545, #ff6b6b); }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

def get_all_students():
    conn = sqlite3.connect('college_data.db')
    df = pd.read_sql_query("SELECT s.*, p.* FROM students s JOIN proctorial p ON s.usn = p.usn", conn)
    conn.close()
    
    # --- FIX: Drop duplicate columns (removes the second 'usn') ---
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def add_new_student(data):
    conn = sqlite3.connect('college_data.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                  (data['usn'], data['name'], data['dob'], data['sem'], data['g1'], data['g2'], data['absences'], data['failures']))
        c.execute("INSERT INTO proctorial VALUES (?, ?, ?, ?, ?, ?)", 
                  (data['usn'], data['study_time'], data['health'], data['famrel'], data['goout'], data['freetime']))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def verify_student(usn, dob):
    conn = sqlite3.connect('college_data.db')
    df = pd.read_sql_query("SELECT s.*, p.* FROM students s JOIN proctorial p ON s.usn = p.usn WHERE s.usn = ? AND s.dob = ?", conn, params=(usn, dob))
    conn.close()
    return df.iloc[0] if not df.empty else None

def run_prediction(student_row):
    # Mapping Logic (Handles both Series and Dict)
    input_data = {
        'G1': student_row['internal1'], 'G2': student_row['internal2'],
        'failures': student_row['failures'], 'absences': student_row['absences'],
        'studytime': student_row['study_time'], 'health': student_row['health'],
        'famrel': student_row['famrel'], 'goout': student_row['goout'],
        'freetime': student_row['freetime'],
        # Defaults
        'age': 21, 'Medu': 3, 'Fedu': 3, 'traveltime': 1, 'romantic': 0, 'internet': 1,
        'schoolsup': 0, 'famsup': 1, 'paid': 0, 'activities': 1, 'nursery': 1,
        'higher': 1, 'famsize': 0, 'Pstatus': 1, 'sex': 1, 'school': 0, 'address': 1,
        'reason': 1, 'guardian': 1, 'Mjob': 2, 'Fjob': 2, 'walc': 1, 'dalc': 1
    }
    input_df = pd.DataFrame([input_data])
    for c in set(feature_names) - set(input_df.columns): input_df[c] = 0
    input_df = input_df[feature_names]
    
    pred = model.predict(input_df)[0]
    
    # SHAP only needed if we want factors
    shap_values = shap.TreeExplainer(model).shap_values(input_df)
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': shap_values[0]}).sort_values(by='importance', key=abs, ascending=False).head(3)
    factors = [f"{row['feature']} ({'Positive' if row['importance']>0 else 'Negative'})" for _, row in feat_imp.iterrows()]
    
    return pred, ", ".join(factors)

def generate_report(name, score, factors):
    prompt = f"Student: {name}. Predicted Grade: {score:.2f}/20. Key Factors: {factors}. Write a strict 3-bullet academic summary."
    return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text

def generate_timetable(student_data):
    prompt = f"""
    Create a personalized 3-day study plan (Markdown Table) for:
    - Current Performance: Internal 1 ({student_data['internal1']}), Internal 2 ({student_data['internal2']}).
    - Habits: Study Level {student_data['study_time']}/4, Free Time {student_data['freetime']}/5.
    - Health Status: {student_data['health']}/5.
    Include specific breaks if health is low. Focus on weak subjects.
    """
    return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text

# --- SESSION STATE ---
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None
    st.session_state['user_data'] = None

# ==========================================
# 1. LOGIN SCREEN
# ==========================================
if st.session_state['user_role'] is None:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 50px;'><h1>🎓 University AI Portal</h1></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Secure Access")
            usn = st.text_input("Identity ID (USN)", placeholder="e.g., 1RV23MCA003")
            dob = st.text_input("Password (DOB)", type="password", placeholder="YYYY-MM-DD")
            
            if st.button("Log In", type="primary"):
                if usn == "ADMIN" and dob == "admin123":
                    st.session_state['user_role'] = "ADMIN"
                    st.rerun()
                else:
                    user = verify_student(usn, dob)
                    if user is not None:
                        st.session_state['user_role'] = "STUDENT"
                        st.session_state['user_data'] = user
                        st.rerun()
                    else:
                        st.error("Invalid Credentials")

# ==========================================
# 2. ADMIN DASHBOARD
# ==========================================
# ==========================================
# 2. ADMIN DASHBOARD (UPDATED)
# ==========================================
elif st.session_state['user_role'] == "ADMIN":
    with st.sidebar:
        st.title("Admin Console")
        if st.button("Logout"):
            st.session_state['user_role'] = None
            st.rerun()
    
    st.title("🛠️ Administration")
    tab1, tab2 = st.tabs(["➕ Registration", "📂 Master Database"])
    
    # --- TAB 1: COMPLETE REGISTRATION FORM ---
    with tab1:
        with st.container(border=True):
            st.subheader("Enroll New Student")
            
            # Using a form ensures all data is submitted at once
            with st.form("reg_form"):
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("### 📚 Academic Profile")
                    new_usn = st.text_input("USN ID (Unique)", placeholder="1RV23MCA00X")
                    new_name = st.text_input("Full Name")
                    new_dob = st.text_input("DOB (YYYY-MM-DD)", placeholder="2002-05-20")
                    new_sem = st.number_input("Semester", 1, 8, 4)
                    
                    st.markdown("---")
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        new_g1 = st.number_input("Internal 1", 0, 20)
                        new_abs = st.number_input("Absences", 0, 100)
                    with col_g2:
                        new_g2 = st.number_input("Internal 2", 0, 20)
                        new_fail = st.number_input("Failures", 0, 10)

                with c2:
                    st.markdown("### 🧠 Proctorial / Lifestyle")
                    st.info("Scale: 1 (Low) to 5 (High)")
                    
                    p_study = st.slider("Study Time (1-4)", 1, 4, 2, help="1: <2hrs, 2: 2-5hrs, 3: 5-10hrs, 4: >10hrs")
                    p_health = st.slider("Health Status", 1, 5, 5)
                    p_fam = st.slider("Family Relationship Quality", 1, 5, 4)
                    p_goout = st.slider("Social Life Frequency", 1, 5, 3)
                    p_free = st.slider("Free Time Availability", 1, 5, 3)
                
                st.markdown("---")
                submit_btn = st.form_submit_button("💾 Register Student to Database", type="primary")
                
                if submit_btn:
                    # Construct the full data dictionary
                    data = {
                        'usn': new_usn, 
                        'name': new_name, 
                        'dob': new_dob, 
                        'sem': new_sem, 
                        'g1': new_g1, 
                        'g2': new_g2, 
                        'absences': new_abs,   # Now linked to input
                        'failures': new_fail,  # Now linked to input
                        'study_time': p_study, 
                        'health': p_health, 
                        'famrel': p_fam,       # Now linked to input
                        'goout': p_goout,      # Now linked to input
                        'freetime': p_free     # Now linked to input
                    }
                    
                    if new_usn and new_name and new_dob:
                        if add_new_student(data): 
                            st.success(f"✅ Student {new_name} ({new_usn}) successfully enrolled!")
                        else: 
                            st.error("❌ Error: USN already exists in the database.")
                    else:
                        st.warning("⚠️ Please fill in all required fields (USN, Name, DOB).")

    # --- TAB 2: DATABASE VIEW (NO CHANGES NEEDED) ---
    with tab2:
        st.dataframe(get_all_students(), use_container_width=True)

# ==========================================
# 3. STUDENT DASHBOARD
# ==========================================
elif st.session_state['user_role'] == "STUDENT":
    s = st.session_state['user_data']
    
    with st.sidebar:
        st.title(s['name'])
        st.caption(f"USN: {s['usn']}")
        st.divider()
        if st.button("Secure Logout"):
            st.session_state['user_role'] = None
            st.rerun()

    # --- METRICS ROW ---
    st.subheader("📊 Academic Overview")
    m1, m2, m3, m4 = st.columns(4)
    def metric_html(label, value):
        return f"""<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>"""
    
    m1.markdown(metric_html("Internal 1", s['internal1']), unsafe_allow_html=True)
    m2.markdown(metric_html("Internal 2", s['internal2']), unsafe_allow_html=True)
    m3.markdown(metric_html("Absences", s['absences']), unsafe_allow_html=True)
    m4.markdown(metric_html("Failures", s['failures']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN FEATURES TABS ---
    tab_pred, tab_sim, tab_plan = st.tabs(["🤖 AI Predictor", "🧪 What-If Simulator", "📅 Study Planner"])
    
    # --- TAB 1: STANDARD PREDICTION ---
    with tab_pred:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info("Analyzes official academic records and proctorial data.")
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Processing..."):
                    score, factors = run_prediction(s)
                    advice = generate_report(s['name'], score, factors)
                
                pct = (score/20)*100
                cls = "pred-good" if pct > 70 else "pred-bad"
                status = "ON TRACK" if pct > 70 else "RISK"
                
                st.markdown(f"""
                <div class="pred-box {cls}">
                    <h2>{score:.2f} / 20</h2>
                    <p>{status} ({pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📌 AI Strategic Advice", expanded=True):
                    st.write(advice)
        with c2:
            st.image("https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-8023.jpg", width=400)

    # --- TAB 2: WHAT-IF SIMULATOR (FEATURE 2) ---
    with tab_sim:
        st.markdown("### 🧪 Counterfactual Analysis")
        st.caption("Adjust the sliders to see how lifestyle changes would impact your final grade.")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            # We use keys to prevent state reload issues
            sim_study = st.slider("Study Time Level", 1, 4, int(s['study_time']), key="s_st")
            sim_abs = st.slider("projected Absences", 0, 50, int(s['absences']), key="s_ab")
            sim_goout = st.slider("Social Life", 1, 5, int(s['goout']), key="s_go")
            sim_health = st.slider("Health Status", 1, 5, int(s['health']), key="s_he")
            
        with col_s2:
            # Create Modified Profile
            sim_profile = s.copy()
            sim_profile['study_time'] = sim_study
            sim_profile['absences'] = sim_abs
            sim_profile['goout'] = sim_goout
            sim_profile['health'] = sim_health
            
            # Run Live Prediction
            base_score, _ = run_prediction(s)
            new_score, _ = run_prediction(sim_profile)
            delta = new_score - base_score
            
            st.metric("Simulated Grade", value=f"{new_score:.2f} / 20", delta=f"{delta:.2f}")
            
            if delta > 1:
                st.success("Great! This change positively impacts your grade.")
            elif delta < -1:
                st.error("Warning! This change will lower your grade.")
            else:
                st.info("Minimal impact observed.")

    # --- TAB 3: AI STUDY PLANNER (FEATURE 3) ---
    with tab_plan:
        st.markdown("### 📅 Personalized Timetable Generator")
        st.write("Generate a markdown schedule based on your weak subjects and health status.")
        
        if st.button("Generate My Plan"):
            with st.spinner("AI is crafting your schedule..."):
                plan = generate_timetable(s)
                st.markdown(plan)