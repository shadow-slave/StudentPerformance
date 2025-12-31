import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    # 1. Load Data
    data = pd.read_csv("student-mat.csv", sep=";") 

    # 2. Preprocessing (Convert text categories to numbers)
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    # 3. Features and Target
    # G3 is the final grade. G1 and G2 are period grades.
    X = data.drop(columns=['G3']) 
    y = data['G3']

    # 4. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Save Model and Column names for later use
    joblib.dump(model, "student_grade_model.pkl")
    joblib.dump(X.columns.tolist(), "feature_names.pkl")
    print("Model Trained and Saved!")

if __name__ == "__main__":
    train_model()