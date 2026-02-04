# 🎓 Student Performance Predictor (Hybrid AI Framework)

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI](https://img.shields.io/badge/AI-RandomForest%20%2B%20Gemini-orange)

## 📌 Project Overview

This project is an advanced **Student Performance Prediction System** that moves beyond simple grade forecasting. It implements a **Hybrid AI Architecture** combining:

1.  **Discriminative AI (Random Forest):** For high-accuracy quantitative grade prediction (R² Score: 0.83).
2.  **Explainable AI (SHAP):** To mathematically identify the "Why" behind a student's performance.
3.  **Generative AI (Google Gemini):** To act as an automated academic counselor, generating personalized study plans and intervention strategies.

The system is designed as a full-stack **University Management Portal** with Role-Based Access Control (RBAC) for Admins and Students.

---

## 🚀 Key Features

### 🔐 Secure Portal Architecture

- **Role-Based Access Control:** Distinct dashboards for **Admins** and **Students**.
- **Two-Factor Authentication (Simulated):** Login requires a valid **USN** (University Seat Number) and **DOB** (Date of Birth).
- **Database Management:** Powered by **SQLite** to store Academic records (Internals, Attendance) separate from Proctorial data (Health, Family Support).

### 🤖 The AI Core

- **Performance Forecasting:** Predicts End-Semester grades based on Internal Assessments (G1, G2) and Lifestyle factors.
- **XAI (Explainable AI):** Uses **SHAP values** to visualize which factors (e.g., _High Absences_ or _Low Study Time_) are pulling the grade down.
- **Generative Reports:** Integrated **Google Gemini API** to write detailed, human-like strategy emails and reports for students.

### 🧪 Student Innovation Modules

- **"What-If" Simulator:** An interactive lab where students can tweak their habits (e.g., _reduce social hours_) to see the real-time impact on their predicted grade.
- **AI Study Planner:** Generates a custom markdown study timetable based on the student's weak subjects and health status.

---

## 🛠️ Tech Stack

| Component            | Technology                               |
| :------------------- | :--------------------------------------- |
| **Frontend**         | Streamlit (Python) with Custom CSS       |
| **Machine Learning** | Scikit-Learn (Random Forest Regressor)   |
| **Explainability**   | SHAP (SHapley Additive exPlanations)     |
| **Generative AI**    | Google Gemini API (gemini-2.5-flash/pro) |
| **Database**         | SQLite3                                  |
| **Data Processing**  | Pandas, NumPy                            |
| **Visualization**    | Plotly, Matplotlib                       |

---

## 📂 Project Structure

```text
├── app.py                 # The Main Application (Frontend + Logic)
├── setup_database.py      # Script to initialize/reset the SQLite Database
├── model_pipeline.py      # Script to Train the ML Model
├── evaluate_model.py      # Script to Generate Accuracy Graphs for PPT
├── college_data.db        # The Database file (Created after running setup)
├── student_grade_model.pkl # Saved ML Model
├── feature_names.pkl      # Saved Feature List
├── requirements.txt       # List of dependencies
└── README.md              # Documentation
```
## 📸 Screenshots
### Admin Dashboard
![Admin Dsahboard](https://i.postimg.cc/NjwDY2Wq/admindashboard.png)
### User Dashboard
![User Dashboard](https://i.postimg.cc/2SFw58DC/userdashboard.png)
### Login Page
![Login](https://i.postimg.cc/MpbDGKSv/login.png)
### Prediction Result
![Prediction](https://i.postimg.cc/htbrGPBS/prediction.png)
