import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. RELOAD AND PREPARE DATA ---
# We must replicate the exact training steps to test accurately
print("Loading data...")
data = pd.read_csv("student-mat.csv", sep=";")

# Convert categories to numbers (Same as before)
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

X = data.drop(columns=['G3'])
y = data['G3']

# Split data (Must use same random_state=42 as training to see 'new' data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a fresh model to ensure we have the object ready for plotting
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 2. CALCULATE METRICS ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*30)
print("   MODEL PERFORMANCE METRICS   ")
print("="*30)
print(f"R² Score (Accuracy): {r2:.4f} (Closer to 1.0 is better)")
print(f"MAE (Mean Error):    {mae:.2f} (Average error in marks)")
print(f"RMSE:                {rmse:.2f}")
print("="*30 + "\n")

# --- 3. GENERATE GRAPHS FOR PPT ---

# Graph A: Actual vs Predicted (The "Accuracy Line")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([0, 20], [0, 20], 'r--', lw=2) # The perfect prediction line
plt.xlabel("Actual Grade (from Dataset)")
plt.ylabel("Predicted Grade (by AI)")
plt.title("Actual vs Predicted Grades")
plt.grid(True)
plt.savefig("graph_accuracy.png")
print("Saved: graph_accuracy.png")

# Graph B: Feature Importance (The "Explainability" chart)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10 # Top 10 factors

plt.figure(figsize=(10, 6))
plt.title("Top 10 Factors Influencing Student Grades")
plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='green')
plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45)
plt.tight_layout()
plt.savefig("graph_feature_importance.png")
print("Saved: graph_feature_importance.png")

plt.show()