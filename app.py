import streamlit as st
import pandas as pd
import sqlite3
import joblib
import shap
import google.generativeai as genai
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
from datetime import datetime, date

import base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CONFIGURATION ---
st.set_page_config(page_title="Uni. AI Portal", layout="wide", page_icon="🎓")

# Configure Gemini API
# IMPORTANT: Replace with your actual key or use st.secrets
try:
    genai.configure(api_key="AIzaSyDfxPEiQZy_1Inqg5Bh_KNPdiLNXU4Ve7Y") 
except Exception as e:
    st.error(f"API Configuration Error: {e}")

# Load assets
try:
    model = joblib.load("student_grade_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("⚠️ System Offline: Model files missing. Initialize training sequence.")
    st.stop()

# --- HELPER: LOAD LOTTIE ANIMATION ---
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# --- HUMAN READABLE MAPPING ---
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #0f172a !important; color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; font-weight: 700; }
    p, label, span, div { color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
    div[data-testid="stInputLabel"] { color: #94a3b8 !important; font-size: 13px; text-transform: uppercase; font-weight: 600; }
    div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] { background-color: #020617 !important; border: 1px solid #334155 !important; border-radius: 8px !important; color: white !important; }
    input[class*="st-"] { color: white !important; }
    button[kind="secondary"] { background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
    .metric-value { font-size: 28px !important; font-weight: 700 !important; color: #818cf8 !important; }
    .metric-label { font-size: 14px !important; color: #94a3b8 !important; }
    .pred-box { background: #1e293b; padding: 25px; border-radius: 16px; text-align: center; border: 1px solid #334155; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
    .pred-good { color: #34d399 !important; text-shadow: 0 0 10px rgba(52, 211, 153, 0.2); } 
    .pred-bad { color: #f87171 !important; text-shadow: 0 0 10px rgba(248, 113, 113, 0.2); } 
    .stButton > button { background: linear-gradient(to right, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 8px; padding: 10px 24px; font-weight: 600; transition: transform 0.2s; }
    .stButton > button:hover { transform: scale(1.02); box-shadow: 0 0 15px rgba(99, 102, 241, 0.4); }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---
@st.cache_data
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

# --- PREDICTION LOGIC ---
def run_prediction(student_row):
    input_data = {
        'G1': student_row['internal1'], 'G2': student_row['internal2'],
        'failures': student_row['failures'], 'absences': student_row['absences'],
        'studytime': student_row['study_time'], 'health': student_row['health'],
        'famrel': student_row['famrel'], 'goout': student_row['goout'],
        'freetime': student_row['freetime'],
        'age': 21, 'Medu': 3, 'Fedu': 3, 'traveltime': 1, 'romantic': 0, 'internet': 1,
        'schoolsup': 0, 'famsup': 1, 'paid': 0, 'activities': 1, 'nursery': 1,
        'higher': 1, 'famsize': 0, 'Pstatus': 1, 'sex': 1, 'school': 0, 'address': 1,
        'reason': 1, 'guardian': 1, 'Mjob': 2, 'Fjob': 2, 'walc': 1, 'dalc': 1
    }
    input_df = pd.DataFrame([input_data])
    for c in set(feature_names) - set(input_df.columns): input_df[c] = 0
    input_df = input_df[feature_names]
    
    pred = model.predict(input_df)[0]
    
    # Logic Overrides
    current_absences = input_data['absences']
    if current_absences > 15:
        penalty = (current_absences - 15) * 0.3  
        pred = pred - penalty
    elif current_absences == 0:
        pred = pred + 3.0
    elif current_absences <= 3:
        pred = pred + 1.5 
        
    pred = max(0, min(20, pred))
    
    shap_values = shap.TreeExplainer(model).shap_values(input_df)
    importances = sorted([{
        'feature': feature,
        'importance': shap_values[0][i],
        'value': input_df.iloc[0][i]
    } for i, feature in enumerate(feature_names)], key=lambda x: abs(x['importance']), reverse=True)
    
    factors = []
    for item in importances[:3]: 
        feat = item['feature']
        imp = item['importance']
        val = item['value']
        if feat == 'absences' and imp < 0 and val < 5: continue 
        direction = "Positive" if imp > 0 else "Negative"
        readable_name = FEATURE_MAP.get(feat, feat)
        factors.append(f"{readable_name} ({direction})")
    
    if current_absences > 15: factors.insert(0, "Extreme Class Absences (Negative)")
    return pred, ", ".join(factors)

def generate_report(name, score, factors):
    # Convert to Indian metrics
    pct = (score / 20) * 100
    cgpa = score / 2
    
    # STRICTER PROMPT
    prompt = f"""
    Act as a senior academic counselor at an Indian University.
    
    Student: {name}
    Predicted Score: {pct:.1f}% ({cgpa:.1f} CGPA)
    Influencing Factors: {factors}
    
    Task: Write a concise performance review.
    
    RULES:
    1. DIRECTNESS: Start immediately with the status (e.g., "Aswin is currently At Risk..."). Remove introductions like "This summary outlines...".
    2. CONTENT: specific advice based on the factors provided.
    3. FORBIDDEN: Do NOT mention "Age" or "demographics" as a factor. Ignore it if present.
    4. FORMAT: 
       - 1 Sentence Summary of current status.
       - 3 Bullet points for improvement (Actionable & Strict).
    """
    try:
        return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
    except:
        return "AI Service Unavailable."
        
def generate_timetable(student_data):
    prompt = f"Create a detailed 3-day study table (Markdown) for Internal 1 ({student_data['internal1']*5:.0f}), Internal 2 ({student_data['internal2']*5:.0f}). Study Level {student_data['study_time']}/4."
    try: return genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
    except: return "AI Service Unavailable."

# --- SESSION STATE ---
if 'user_role' not in st.session_state: st.session_state['user_role'] = None; st.session_state['user_data'] = None
if 'pred_result' not in st.session_state: st.session_state['pred_result'] = None
if 'study_plan' not in st.session_state: st.session_state['study_plan'] = None

# ==========================================
# 1. LOGIN SCREEN (Local Background Image)
# ==========================================
if st.session_state['user_role'] is None:
    
    # --- LOAD LOCAL BACKGROUND IMAGE ---
    # Make sure 'background.jpg' is in the SAME folder as app.py
    try:
        img_file = "bg.jpg"  # <--- REPLACE WITH YOUR IMAGE NAME
        bin_str = get_base64_of_bin_file(img_file)
        
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.95)), url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ Error: Could not find image file '{img_file}'. Please check the name.")

    # --- CENTERED LOGIN CARD ---
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        c_logo, c_title = st.columns([1, 4])
        with c_logo: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
        with c_title: st.markdown("<h2 style='margin:0; padding-top:10px;'>Uni. AI Portal</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#cbd5e1; margin-bottom: 20px;'>Student Performance Predictor</p>", unsafe_allow_html=True)
        
        usn = st.text_input("Student Identity (USN)", placeholder="e.g. 1MS24MC001")
        dob_input = st.date_input("Security Key (DOB)", min_value=date(1990, 1, 1), max_value=date.today())
        dob_str = dob_input.strftime('%Y-%m-%d')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Login ➔", type="primary", use_container_width=True):
            if usn == "ADMIN" and dob_str == "2026-01-01":
                st.session_state['user_role'] = "ADMIN"
                st.rerun()
            elif usn == "ADMIN":
                st.error("Admin Access: Use 2026-01-01")
            else:
                user = verify_student(usn, dob_str)
                if user is not None:
                    st.session_state['user_role'] = "STUDENT"
                    st.session_state['user_data'] = user
                    st.rerun()
                else:
                    st.error("⛔ Access Denied: Invalid Credentials")
        
        st.markdown('</div>', unsafe_allow_html=True)
# ==========================================
# 2. ADMIN DASHBOARD
# ==========================================
elif st.session_state['user_role'] == "ADMIN":
    def logout():
        st.session_state.clear() # Wipes all data (Role, Student Data, Prediction, Study Plan)
        
    with st.sidebar:
        st.markdown("### Admin Console")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.markdown("---")
        st.button("Logout", on_click=logout)
    
    st.title("Admin Dashboard")
    tab1, tab2 = st.tabs(["Add Student", "Database & Analytics"])
    
    with tab1:
        with st.container(border=True):
            st.subheader("Student Enrollment")
            with st.form("reg_form"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Academic Info (Enter Percentage 0-100)**")
                    new_usn = st.text_input("USN")
                    new_name = st.text_input("Name")
                    
                    # --- DATE PICKER UI ---
                    new_dob_date = st.date_input("Date of Birth", value=date(2002, 1, 1))
                    
                    new_sem = st.number_input("Semester", 1, 8, 4)
                    col_g1, col_g2 = st.columns(2)
                    with col_g1: raw_g1 = st.number_input("Internal 1 (%)", 0, 100)
                    with col_g2: raw_g2 = st.number_input("Internal 2 (%)", 0, 100)
                
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
                     conv_g1 = raw_g1 / 5
                     conv_g2 = raw_g2 / 5
                     # Convert Date to String
                     dob_save = new_dob_date.strftime('%Y-%m-%d')
                     
                     data = {'usn': new_usn, 'name': new_name, 'dob': dob_save, 'sem': new_sem, 
                             'g1': conv_g1, 'g2': conv_g2, 'absences': new_abs, 'failures': new_fail, 
                             'study_time': p_study, 'health': p_health, 'famrel': p_fam, 'goout': p_goout, 'freetime': p_free}
                     
                     if add_new_student(data): st.success(f"Student added! (Saved DOB: {dob_save})")
                     else: st.error("Error: USN already exists.")

    with tab2:
        st.markdown("### 📊 Class Analytics")
        all_students = get_all_students()
        
        if not all_students.empty:
            avg_g1 = all_students['internal1'].mean() * 5 
            avg_g2 = all_students['internal2'].mean() * 5
            k1, k2, k3 = st.columns(3)
            k1.metric("Class Strength", len(all_students))
            k2.metric("Avg Internal 1", f"{avg_g1:.1f}%")
            k3.metric("Avg Internal 2", f"{avg_g2:.1f}%")
            
            fig = go.Figure(data=[
                go.Bar(name='Internal 1', x=all_students['name'], y=all_students['internal1']*5),
                go.Bar(name='Internal 2', x=all_students['name'], y=all_students['internal2']*5)
            ])
            fig.update_layout(barmode='group', title="Class Performance Overview", height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🗂️ Database Management")
        st.dataframe(all_students, use_container_width=True)
        
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
                            
                            # --- DATE PICKER UI (Pre-fill with existing date) ---
                            try:
                                default_date = datetime.strptime(s_data['dob'], '%Y-%m-%d').date()
                            except:
                                default_date = date(2000, 1, 1)
                            
                            e_dob_date = st.date_input("DOB", value=default_date)
                            
                            e_sem = st.number_input("Semester", 1, 8, int(s_data['sem']))
                            current_g1_pct = float(s_data['internal1']) * 5
                            current_g2_pct = float(s_data['internal2']) * 5
                            edit_raw_g1 = st.number_input("Internal 1 (%)", 0.0, 100.0, current_g1_pct)
                            edit_raw_g2 = st.number_input("Internal 2 (%)", 0.0, 100.0, current_g2_pct)
                        
                        with ec2:
                            e_abs = st.number_input("Absences", 0, 100, int(s_data['absences']))
                            e_fail = st.number_input("Failures", 0, 10, int(s_data['failures']))
                            e_study = st.slider("Study Time", 1, 4, int(s_data['study_time']))
                            e_health = st.slider("Health", 1, 5, int(s_data['health']))
                            e_fam = st.slider("Family Rel", 1, 5, int(s_data['famrel']))
                            e_goout = st.slider("Going Out", 1, 5, int(s_data['goout']))
                            e_free = st.slider("Free Time", 1, 5, int(s_data['freetime']))

                        if st.form_submit_button("💾 Update Details"):
                             upd_g1 = edit_raw_g1 / 5
                             upd_g2 = edit_raw_g2 / 5
                             e_dob_save = e_dob_date.strftime('%Y-%m-%d')
                             
                             upd_data = {
                                 'usn': selected_usn, 'name': e_name, 'dob': e_dob_save, 'sem': e_sem,
                                 'g1': upd_g1, 'g2': upd_g2, 'absences': e_abs, 'failures': e_fail,
                                 'study_time': e_study, 'health': e_health, 'famrel': e_fam,
                                 'goout': e_goout, 'freetime': e_free
                             }
                             if update_student(upd_data):
                                 st.success("✅ Student updated successfully!")
                                 st.rerun()
                             else: st.error("❌ Update failed.")

                with col_delete:
                    st.warning("Deleting a record is permanent.")
                    if st.button("🗑️ DELETE STUDENT", type="primary"):
                        if delete_student(selected_usn):
                            st.success(f"Student {selected_usn} deleted.")
                            st.rerun()
                        else: st.error("Delete failed.")
# ==========================================
# 3. STUDENT DASHBOARD (ENHANCED SIDEBAR)
# ==========================================
elif st.session_state['user_role'] == "STUDENT":
    s = st.session_state['user_data']
    
    with st.sidebar:
        # --- ENHANCED PROFILE SECTION ---
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title(s['name'])
        st.markdown(f"**{s['usn'][0]}**")
        
        st.markdown("---")
        st.markdown("### 👤 Profile Details")
        
        # Display extra details from the database
        st.markdown(f"**📚 Semester:** {s['sem']}")
        st.markdown(f"**🎂 DOB:** {s['dob']}")
        
        # Static "Status" badge to make it look official
        st.success("✅ Status: Active Student")
        
        st.markdown("---")
        
        # LOGOUT WITH SESSION CLEAR (The Fix from before)
        def logout():
            st.session_state.clear()
        st.button("Logout", on_click=logout, type="secondary")

    # --- TOP METRICS (Standard) ---
    st.subheader("Overview")
    col_metrics, col_chart = st.columns([1, 1.5])
    
    with col_metrics:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        def metric_card(label, value):
            return f"""<div class="metric-card" style="padding:15px; margin-bottom:10px;">
                       <div class="metric-value">{value}</div>
                       <div class="metric-label">{label}</div></div>"""
        
        # Display Internals as Percentage (Score * 5)
        m1.markdown(metric_card("Internal 1", f"{s['internal1']*5:.0f}%"), unsafe_allow_html=True)
        m2.markdown(metric_card("Internal 2", f"{s['internal2']*5:.0f}%"), unsafe_allow_html=True)
        m3.markdown(metric_card("Absences", s['absences']), unsafe_allow_html=True)
        m4.markdown(metric_card("Failures", s['failures']), unsafe_allow_html=True)

    with col_chart:
        # Comparative Chart
        all_students = get_all_students()
        avg_g1 = all_students['internal1'].mean() * 5
        avg_g2 = all_students['internal2'].mean() * 5
        my_g1 = s['internal1'] * 5
        my_g2 = s['internal2'] * 5
        
        fig = go.Figure(data=[
            go.Bar(name='My Score (%)', x=['Internal 1', 'Internal 2'], y=[my_g1, my_g2], marker_color='#6366f1'), 
            go.Bar(name='Class Avg (%)', x=['Internal 1', 'Internal 2'], y=[avg_g1, avg_g2], marker_color='#94a3b8')
        ])
        fig.update_layout(
            barmode='group', title="My Performance vs. Class Average", 
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'), height=280, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    tab_pred, tab_sim, tab_plan = st.tabs(["AI Prediction", "Simulator", "Study Plan"])
    
    # --- TAB 1: AI PREDICTION (INDIAN STYLE) ---
    with tab_pred:
        if st.session_state['pred_result']:
            res = st.session_state['pred_result']
            raw_score = res['score'] # 0-20 scale
            
            # --- CONVERSION LOGIC ---
            final_pct = (raw_score / 20) * 100
            final_cgpa = raw_score / 2
            
            color_class = "pred-good" if final_pct > 70 else "pred-bad"
            status_text = "Distinction" if final_pct > 75 else ("First Class" if final_pct > 60 else "Risk")
            forecast_color = '#34d399' if final_pct > 70 else '#f87171'

            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.markdown(f"""
                <div class="pred-box">
                    <h1 class="{color_class}" style="font-size:4rem; margin:0;">{final_pct:.1f}%</h1>
                    <div style="color:#94a3b8; font-size:1.4rem; font-weight:600;">{final_cgpa:.2f} CGPA | {status_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📈 Academic Trajectory")
                y_past = [s['internal1']*5, s['internal2']*5]
                y_future = [s['internal2']*5, final_pct]
                x_past = ["Internal 1", "Internal 2"]
                x_future = ["Internal 2", "Final (Predicted)"]
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=x_past, y=y_past, mode='lines+markers', name='History', line=dict(color='#6366f1', width=3), marker=dict(size=8)))
                fig_trend.add_trace(go.Scatter(x=x_future, y=y_future, mode='lines+markers', name='Forecast', line=dict(color=forecast_color, width=3, dash='dot'), marker=dict(size=8, symbol='star')))
                fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=True, yaxis=dict(range=[0, 100], title="Percentage (%)", gridcolor='#334155'), xaxis=dict(showgrid=False))
                st.plotly_chart(fig_trend, use_container_width=True)

            with c2:
                with st.container(border=True):
                    st.markdown("### 🤖 AI Counselor")
                    st.info("Based on your pattern, here is my assessment:")
                    st.write(res['advice'])
                    st.markdown("---")
                    
                    report_text = f"""
                    UNIVERSITY STUDENT REPORT
                    ----------------------------------
                    Name: {s['name']} (USN: {s['usn']})
                    
                    ACADEMIC PERFORMANCE:
                    - Internal 1: {s['internal1']*5:.1f}%
                    - Internal 2: {s['internal2']*5:.1f}%
                    
                    AI FORECAST:
                    - Predicted Percentage: {final_pct:.2f}%
                    - Predicted CGPA:       {final_cgpa:.2f}
                    - Status:               {status_text}
                    
                    COUNSELOR ADVICE:
                    {res['advice']}
                    ----------------------------------
                    Generated by Uni. AI Portal
                    """
                    st.download_button(label="📄 Download Report", data=report_text, file_name=f"{s['name']}_Report.txt", mime="text/plain", type="primary", use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state['pred_result'] = None
                st.rerun()

        else:
            col_hero_text, col_hero_img = st.columns([1.5, 1])
            with col_hero_text:
                st.markdown(f"""<div style="padding-top: 10px;"><h2 style="font-size: 2.2rem; margin-bottom: 10px; color: #f8fafc;">Ready to Forecast, {s['name']}?</h2><p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.6;">Our Hybrid AI is ready to analyze your academic profile. We will evaluate <b>33 data points</b> to predict your final CGPA/Percentage.</p></div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                def status_badge(title, status, icon, color):
                    return f"""<div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center;"><div style="font-size: 24px; margin-bottom: 5px;">{icon}</div><div style="font-weight: 600; color: #f8fafc; font-size: 14px;">{title}</div><div style="color: {color}; font-size: 12px; font-weight: bold; margin-top: 5px;">{status}</div></div>"""
                k1.markdown(status_badge("Academics", "DATA FOUND", "📚", "#34d399"), unsafe_allow_html=True)
                k2.markdown(status_badge("Attendance", "LOGGED", "📅", "#60a5fa"), unsafe_allow_html=True)
                k3.markdown(status_badge("Lifestyle", "CONNECTED", "🧘", "#a78bfa"), unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🚀 Launch AI Analysis", type="primary", use_container_width=True):
                    with st.spinner("🔄 Crunching numbers & generating insights..."):
                        score, factors = run_prediction(s)
                        advice = generate_report(s['name'], score, factors)
                        st.session_state['pred_result'] = {'score': score, 'factors': factors, 'advice': advice}
                        st.rerun()

            with col_hero_img:
                lottie_url = "https://assets8.lottiefiles.com/packages/lf20_qp1q7mct.json" 
                animation = load_lottieurl(lottie_url)
                if animation: st_lottie(animation, height=350, key="analysis_anim")

    # --- TAB 2: SIMULATOR ---
    with tab_sim:
        st.markdown("### What-If Analysis")
        st.caption("Adjust sliders to see the impact on your final grade.")
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
        
        base_pct = (base_score/20)*100
        new_pct = (new_score/20)*100
        diff = new_pct - base_pct
        
        st.metric("Projected Percentage", f"{new_pct:.2f}%", delta=f"{diff:.2f}%")

    # --- TAB 3: STUDY PLAN ---
    with tab_plan:
        c_head, c_btn = st.columns([3, 1])
        with c_head:
            st.markdown("### Smart Study Planner")
        
        if st.button("Generate Schedule"):
            with st.spinner("Generating..."):
                plan = generate_timetable(s)
                st.session_state['study_plan'] = plan
        
        if st.session_state['study_plan']:
            st.markdown(st.session_state['study_plan'])
            st.download_button(
                label="📥 Download Study Plan",
                data=st.session_state['study_plan'],
                file_name=f"{s['name']}_Study_Plan.md",
                mime="text/markdown",
                type="secondary"
            )
            