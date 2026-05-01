import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/regression_model.pkl")


def show_regression_page():
    st.title("Student Performance Regression")
    st.subheader("Predicting Final Score")

    st.markdown("""
    Această pagină prezice scorul final al unui student pe baza factorilor academici,
    personali și comportamentali.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=30, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        hours_studied = st.number_input("Hours_Studied", min_value=0.0, max_value=15.0, value=8.0)
        attendance = st.number_input("Attendance", min_value=0.0, max_value=100.0, value=95.0)
        sleep_hours = st.number_input("Sleep_Hours", min_value=0.0, max_value=12.0, value=7.5)

    with col2:
        stress_level = st.slider("Stress_Level", 0.0, 10.0, 3.0)
        screen_time = st.number_input("Screen_Time", min_value=0.0, max_value=15.0, value=2.0)
        previous_gpa = st.number_input("Previous_GPA", min_value=0.0, max_value=4.0, value=3.8)
        part_time_job = st.selectbox("Part_Time_Job", ["No", "Yes"])
        study_method = st.selectbox("Study_Method", ["Online", "Offline", "Hybrid"])

    with col3:
        diet_quality = st.selectbox("Diet_Quality", ["Poor", "Average", "Good"])
        internet_quality = st.selectbox("Internet_Quality", ["Poor", "Average", "Good", "Excellent"])
        extracurricular = st.selectbox("Extracurricular", ["Yes", "No"])
        tutoring = st.number_input("Tutoring_Sessions_Per_Week", min_value=0, max_value=10, value=3)
        family_income = st.selectbox("Family_Income_Level", ["Low", "Middle", "High"])
        exam_anxiety = st.slider("Exam_Anxiety_Score", 0.0, 10.0, 2.0)

    X = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Stress_Level": stress_level,
        "Screen_Time": screen_time,
        "Previous_GPA": previous_gpa,
        "Part_Time_Job": part_time_job,
        "Study_Method": study_method,
        "Diet_Quality": diet_quality,
        "Internet_Quality": internet_quality,
        "Extracurricular": extracurricular,
        "Tutoring_Sessions_Per_Week": tutoring,
        "Family_Income_Level": family_income,
        "Exam_Anxiety_Score": exam_anxiety
    }])

    st.write("Input data:")
    st.dataframe(X)

    if st.button("Predict Final Score"):
        pred = model.predict(X)[0]

        st.success(f"Predicted Final Score: {pred:.2f}")

        if pred >= 90:
            st.info("Interpretare: scor estimat foarte bun.")
        elif pred >= 75:
            st.info("Interpretare: scor estimat bun.")
        elif pred >= 50:
            st.warning("Interpretare: scor estimat mediu.")
        else:
            st.error("Interpretare: scor estimat scăzut.")

    st.divider()