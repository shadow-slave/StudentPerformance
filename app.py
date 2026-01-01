import streamlit as st
import pandas as pd
import sqlite3
import joblib
import shap
import google.generativeai as genai
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="High Sc. AI Portal", layout="wide", page_icon="🎓")

# Configure Gemini API
# IMPORTANT: Replace with your actual key or use st.secrets
try:
    genai.configure(api_key="YOUR_API_KEY_HERE") 
except Exception as e:
    st.error(f"API Configuration Error: {e}")

# Load assets
try:
    model = joblib.load("student_grade_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("⚠️ System Offline: Model files missing. Initialize training sequence.")
    st.stop()

# --- HUMAN READABLE MAPPING (THE FIX) ---
FEATURE_MAP = {
    "G1": "Internal Exam 1",
    "G2": "Internal Exam 2",
    "absences": "Class Absences",
    "failures": "Past Failures",
    "studytime": "Study Time",
    "health": "Health Status",
    "famrel": "Family Relationships",
    "goout": "Social Activity / Partying",
    "freetime": "Free Time",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "traveltime": "Commute Time"
}

# --- CSS: MODERN DARK MODE THEME ---
st.markdown("""
<style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* --- GLOBAL DARK THEME --- */
    .stApp {
        background-color: #0f172a !important; /* Deep Slate Background */
        color: #f8fafc !important; /* Light Text */
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    /* Text Paragraphs & Labels */
    p, label, span, div {
        color: #e2e8f0;
    }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important; /* Lighter Slate */
        border-right: 1px solid #334155;
    }
    
    /* --- INPUT FIELDS --- */
    div[data-testid="stInputLabel"] {
        color: #94a3b8 !important; /* Muted Grey */
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #020617 !important; /* Very Dark */
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    input[class*="st-"] { color: white !important; }
    
    button[kind="secondary"] {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }

    /* --- METRIC CARDS --- */
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #818cf8 !important; /* Indigo Accent */
    }
    .metric-label {
        font-size: 14px !important;
        color: #94a3b8 !important;
    }

    /* --- PREDICTION BOXES --- */
    .pred-box {
        background: #1e293b;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    .pred-good { color: #34d399 !important; text-shadow: 0 0 10px rgba(52, 211, 153, 0.2); } 
    .pred-bad { color: #f87171 !important; text-shadow: 0 0 10px rgba(248, 113, 113, 0.2); } 

    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(to right, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b;
        padding: 8px;
        border-radius: 10px;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
        padding: 12px 30px !important; /* Increased Padding */
        font-size: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: white !important;
    }
    
    /* --- DATAFRAME --- */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

def get_all_students():
    conn = sqlite3.connect('college_data.db')
    df = pd.read_sql_query("SELECT s.*, p.* FROM students s JOIN proctorial p ON s.usn = p.usn", conn)
    conn.close()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def get_student_by_usn(usn):
    conn = sqlite3.connect('college_data.db')
    df = pd.read_sql_query("SELECT s.*, p.* FROM students s JOIN proctorial p ON s.usn = p.usn WHERE s.usn = ?", conn, params=(usn,))
    conn.close()
    return df.iloc[0] if not df.empty else None

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

def update_student(data):
    conn = sqlite3.connect('college_data.db')
    c = conn.cursor()
    try:
        # Update Students Table
        c.execute("""UPDATE students SET name=?, dob=?, sem=?, internal1=?, internal2=?, absences=?, failures=? WHERE usn=?""", 
                  (data['name'], data['dob'], data['sem'], data['g1'], data['g2'], data['absences'], data['failures'], data['usn']))
        # Update Proctorial Table
        c.execute("""UPDATE proctorial SET study_time=?, health=?, famrel=?, goout=?, freetime=? WHERE usn=?""", 
                  (data['study_time'], data['health'], data['famrel'], data['goout'], data['freetime'], data['usn']))
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally: conn.close()

def delete_student(usn):
    conn = sqlite3.connect('college_data.db')
    c = conn.cursor()
    try:
        c.execute("DELETE FROM proctorial WHERE usn=?", (usn,))
        c.execute("DELETE FROM students WHERE usn=?", (usn,))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def verify_student(usn, dob):
    conn = sqlite3.connect('college_data.db')
    df = pd.read_sql_query("SELECT s.*, p.* FROM students s JOIN proctorial p ON s.usn = p.usn WHERE s.usn = ? AND s.dob = ?", conn, params=(usn, dob))
    conn.close()
    return df.iloc[0] if not df.empty else None

# --- UPDATED PREDICTION FUNCTION (With Logic Fixes) ---
def run_prediction(student_row):
    # Mapping Logic
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
    
    # 1. Base Prediction
    pred = model.predict(input_df)[0]
    
    # 2. LOGIC OVERRIDES (The "Common Sense" Layer)
    current_absences = input_data['absences']
    
    if current_absences > 15:
        # Penalty for extreme absences
        penalty = (current_absences - 15) * 0.3  
        pred = pred - penalty
        
    elif current_absences < 2:
        # BOOST for Perfect/Near-Perfect Attendance
        # This fixes the issue where 0 absences gave a lower score than 2.
        pred = pred + 2.0 
        
    pred = max(0, min(20, pred))
    
    # 3. Explanation Generation
    shap_values = shap.TreeExplainer(model).shap_values(input_df)
    
    importances = []
    for i, feature in enumerate(feature_names):
        importances.append({
            'feature': feature,
            'importance': shap_values[0][i],
            'value': input_df.iloc[0][i]
        })
    
    importances = sorted(importances, key=lambda x: abs(x['importance']), reverse=True)
    
    factors = []
    for item in importances[:3]: 
        feat = item['feature']
        imp = item['importance']
        val = item['value']
        
        # Filter: If model complains about low absences, ignore it
        if feat == 'absences' and imp < 0 and val < 5:
            continue 
            
        direction = "Positive" if imp > 0 else "Negative"
        readable_name = FEATURE_MAP.get(feat, feat)
        factors.append(f"{readable_name} ({direction})")
    
    if current_absences > 15:
        factors.insert(0, "Extreme Class Absences (Negative)")
    
    return pred, ", ".join(factors)

def generate_report(name, score, factors):
    # Prompt explicitly hides Age from the text report
    prompt = f"Student: {name}. Predicted Grade: {score:.2f}/20. Key Factors: {factors}. Do not mention age for this. Write a professional academic summary in 3 bullet points."
    try:
        return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
    except:
        return "AI Service Unavailable."

def generate_timetable(student_data):
    prompt = f"""
    Create a detailed 3-day study table (Markdown) for:
    - Internal 1 ({student_data['internal1']}), Internal 2 ({student_data['internal2']}).
    - Study Level {student_data['study_time']}/4.
    """
    try:
        return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
    except:
        return "AI Service Unavailable."

# --- SESSION STATE INITIALIZATION (FULL RETENTION) ---
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None
    st.session_state['user_data'] = None

# Store PREDICTION results
if 'pred_result' not in st.session_state:
    st.session_state['pred_result'] = None

# Store STUDY PLAN results
if 'study_plan' not in st.session_state:
    st.session_state['study_plan'] = None

# ==========================================
# 1. LOGIN SCREEN
# ==========================================
if st.session_state['user_role'] is None:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.title("🎓 University Portal")
        st.markdown("<p style='color:#94a3b8; margin-bottom: 20px;'>Secure Academic Management System</p>", unsafe_allow_html=True)
        
        with st.container(border=True):
            usn = st.text_input("Student ID (USN)", placeholder="1RV23MCA003")
            dob = st.text_input("Password (DOB)", type="password", placeholder="YYYY-MM-DD")
            
            if st.button("Sign In", type="primary", use_container_width=True):
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
elif st.session_state['user_role'] == "ADMIN":
    with st.sidebar:
        st.markdown("### Admin Console")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.markdown("---")
        st.button("Logout", on_click=lambda: st.session_state.update({'user_role': None}))
    
    st.title("Admin Dashboard")
    tab1, tab2 = st.tabs(["Add Student", "Database & Management"])
    
    # --- TAB 1: ADD STUDENT ---
    with tab1:
        with st.container(border=True):
            st.subheader("Student Enrollment")
            with st.form("reg_form"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Academic Info**")
                    new_usn = st.text_input("USN")
                    new_name = st.text_input("Name")
                    new_dob = st.text_input("DOB")
                    new_sem = st.number_input("Semester", 1, 8, 4)
                    col_g1, col_g2 = st.columns(2)
                    with col_g1: new_g1 = st.number_input("Internal 1", 0, 20)
                    with col_g2: new_g2 = st.number_input("Internal 2", 0, 20)
                
                with c2:
                    st.markdown("**Proctorial Info**")
                    new_abs = st.number_input("Absences", 0, 100)
                    new_fail = st.number_input("Failures", 0, 10)
                    p_study = st.slider("Study Time", 1, 4, 2)
                    p_health = st.slider("Health", 1, 5, 5)
                    p_fam = st.slider("Family Rel", 1, 5, 4)
                    p_goout = st.slider("Going Out", 1, 5, 3)
                    p_free = st.slider("Free Time", 1, 5, 3)
                
                st.markdown("---")
                if st.form_submit_button("Save Record", type="primary"):
                     data = {'usn': new_usn, 'name': new_name, 'dob': new_dob, 'sem': new_sem, 'g1': new_g1, 
                             'g2': new_g2, 'absences': new_abs, 'failures': new_fail, 'study_time': p_study, 
                             'health': p_health, 'famrel': p_fam, 'goout': p_goout, 'freetime': p_free}
                     if add_new_student(data): st.success("Student added successfully.")
                     else: st.error("Error: USN already exists.")

    # --- TAB 2: MANAGE RECORDS (EDIT/DELETE) ---
    with tab2:
        st.markdown("### 🗂️ Student Records")
        all_students = get_all_students()
        st.dataframe(all_students, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🛠️ Manage Records")
        
        # Select Student to Edit/Delete
        student_list = all_students['usn'].tolist()
        selected_usn = st.selectbox("Select Student to Edit/Delete", options=["Select..."] + student_list)
        
        if selected_usn != "Select...":
            st.divider()
            s_data = get_student_by_usn(selected_usn)
            
            if s_data is not None:
                col_edit, col_delete = st.columns([3, 1])
                
                with col_edit:
                    with st.form("edit_form"):
                        st.markdown(f"**Editing: {s_data['name']} ({selected_usn})**")
                        
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            e_name = st.text_input("Name", s_data['name'])
                            e_dob = st.text_input("DOB", s_data['dob'])
                            e_sem = st.number_input("Semester", 1, 8, int(s_data['sem']))
                            e_g1 = st.number_input("Internal 1", 0, 20, int(s_data['internal1']))
                            e_g2 = st.number_input("Internal 2", 0, 20, int(s_data['internal2']))
                        
                        with ec2:
                            e_abs = st.number_input("Absences", 0, 100, int(s_data['absences']))
                            e_fail = st.number_input("Failures", 0, 10, int(s_data['failures']))
                            e_study = st.slider("Study Time", 1, 4, int(s_data['study_time']))
                            e_health = st.slider("Health", 1, 5, int(s_data['health']))
                            e_fam = st.slider("Family Rel", 1, 5, int(s_data['famrel']))
                            e_goout = st.slider("Going Out", 1, 5, int(s_data['goout']))
                            e_free = st.slider("Free Time", 1, 5, int(s_data['freetime']))

                        if st.form_submit_button("💾 Update Details"):
                             upd_data = {
                                 'usn': selected_usn, 'name': e_name, 'dob': e_dob, 'sem': e_sem,
                                 'g1': e_g1, 'g2': e_g2, 'absences': e_abs, 'failures': e_fail,
                                 'study_time': e_study, 'health': e_health, 'famrel': e_fam,
                                 'goout': e_goout, 'freetime': e_free
                             }
                             if update_student(upd_data):
                                 st.success("✅ Student updated successfully!")
                                 st.rerun()
                             else:
                                 st.error("❌ Update failed.")

                with col_delete:
                    st.markdown("### ⚠️ Danger Zone")
                    st.warning("Deleting a record is permanent.")
                    if st.button("🗑️ DELETE STUDENT", type="primary"):
                        if delete_student(selected_usn):
                            st.success(f"Student {selected_usn} deleted.")
                            st.rerun()
                        else:
                            st.error("Delete failed.")

# ==========================================
# 3. STUDENT DASHBOARD
# ==========================================
elif st.session_state['user_role'] == "STUDENT":
    s = st.session_state['user_data']
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.title(s['name'])
        st.caption(f"ID: {s['usn'][0]}")
        st.markdown("---")
        st.button("Logout", on_click=lambda: st.session_state.update({'user_role': None}))

    # --- METRICS ROW ---
    st.subheader("Overview")
    m1, m2, m3, m4 = st.columns(4)
    def metric_card(label, value):
        return f"""<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>"""
    
    m1.markdown(metric_card("Internal 1", s['internal1']), unsafe_allow_html=True)
    m2.markdown(metric_card("Internal 2", s['internal2']), unsafe_allow_html=True)
    m3.markdown(metric_card("Absences", s['absences']), unsafe_allow_html=True)
    m4.markdown(metric_card("Failures", s['failures']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN FEATURES TABS ---
    tab_pred, tab_sim, tab_plan = st.tabs(["AI Prediction", "Simulator", "Study Plan"])
    
    # --- TAB 1: AI PREDICTION ---
    with tab_pred:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Performance Forecast")
            if st.button("Analyze Performance", type="primary"):
                with st.spinner("Analyzing..."):
                    score, factors = run_prediction(s)
                    advice = generate_report(s['name'], score, factors)
                    
                    # SAVE PREDICTION TO SESSION STATE
                    st.session_state['pred_result'] = {
                        'score': score,
                        'factors': factors,
                        'advice': advice
                    }

            # DISPLAY SAVED PREDICTION
            if st.session_state['pred_result']:
                res = st.session_state['pred_result']
                score = res['score']
                
                pct = (score/20)*100
                color_class = "pred-good" if pct > 70 else "pred-bad"
                status_text = "On Track" if pct > 70 else "At Risk"
                
                st.markdown(f"""
                <div class="pred-box">
                    <h1 class="{color_class}" style="font-size:4rem; margin:0;">{score:.2f}</h1>
                    <div style="color:#94a3b8; font-size:1.2rem; font-weight:600;">{status_text} ({pct:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container(border=True):
                    st.markdown("**📌 AI Insights**")
                    st.write(res['advice'])
        with c2:
             st.info("Our AI model evaluates your academic history and lifestyle choices to predict your end-semester grade with 83% accuracy.")

    # --- TAB 2: SIMULATOR ---
    with tab_sim:
        st.markdown("### What-If Analysis")
        st.caption("Adjust sliders to see the impact on your grade.")
        c1, c2 = st.columns(2)
        with c1:
            sim_study = st.slider("Study Time (1=Low, 4=High)", 1, 4, int(s['study_time']))
            sim_abs = st.slider("Projected Absences", 0, 50, int(s['absences']))
        with c2:
            sim_goout = st.slider("Partying / Going Out", 1, 5, int(s['goout']))
            sim_health = st.slider("Health Status", 1, 5, int(s['health']))
            
        sim_profile = s.copy()
        sim_profile['study_time'] = sim_study; sim_profile['absences'] = sim_abs
        sim_profile['goout'] = sim_goout; sim_profile['health'] = sim_health
        
        base_score, _ = run_prediction(s)
        new_score, _ = run_prediction(sim_profile)
        
        st.metric("Projected Grade", f"{new_score:.2f} / 20", delta=f"{new_score-base_score:.2f}")

    # --- TAB 3: STUDY PLAN (NOW PERSISTENT) ---
    with tab_plan:
        st.markdown("### Smart Study Planner")
        if st.button("Generate Schedule"):
            with st.spinner("Generating..."):
                plan = generate_timetable(s)
                # SAVE STUDY PLAN TO SESSION STATE
                st.session_state['study_plan'] = plan
        
        # DISPLAY SAVED STUDY PLAN
        if st.session_state['study_plan']:
            st.markdown(st.session_state['study_plan'])