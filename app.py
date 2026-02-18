import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Title
st.title("🎓 Student Performance Prediction App")

# Load dataset
df = pd.read_csv("dataset.csv")

# Create separate encoders
le_grade = LabelEncoder()
le_result = LabelEncoder()

df['previous_grade'] = le_grade.fit_transform(df['previous_grade'])
df['result'] = le_result.fit_transform(df['result'])

# Split features and target
X = df[['attendance', 'study_hours', 'internal_marks', 'previous_grade']]
y = df['result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ---- USER INPUT SECTION ----
st.header("Enter Student Details")

attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
study_hours = st.number_input("Study Hours", min_value=0, max_value=24)
internal_marks = st.number_input("Internal Marks", min_value=0, max_value=100)
previous_grade = st.selectbox("Previous Grade", le_grade.classes_)

# Predict button
if st.button("Predict Result"):

    previous_grade_encoded = le_grade.transform([previous_grade])[0]

    new_student = [[attendance, study_hours, internal_marks, previous_grade_encoded]]

    prediction = model.predict(new_student)

    result_label = le_result.inverse_transform(prediction)[0]

    if result_label.lower() == "pass":
        st.success("🎉 Student will PASS")
    else:
        st.error("❌ Student will FAIL")

