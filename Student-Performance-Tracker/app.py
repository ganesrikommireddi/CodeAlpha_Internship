"""
app.py
-------
Streamlit web app for the Student Performance Tracker.

Loads the pickled best model (models/best_model.pkl) and lets a user
enter a student's details in order to predict their final exam score.

Run with:
    streamlit run app.py
"""

import os
import pickle

import pandas as pd
import streamlit as st

from src.data_preprocessing import clean_data, engineer_features

MODEL_PATH = os.path.join("models", "best_model.pkl")

st.set_page_config(page_title="Student Performance Tracker", page_icon="🎓", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["model_name"], artifact["feature_columns"]


def build_input_row(inputs: dict) -> pd.DataFrame:
    """Turn raw form inputs into a single-row dataframe matching the training pipeline."""
    raw_df = pd.DataFrame([inputs])
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    return engineered


def align_columns(row_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Ensure the single prediction row has exactly the same columns
    (and order) as the data the model was trained on. Missing dummy
    columns (e.g. a category not present in this single row) are
    filled with 0.
    """
    for col in feature_columns:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[feature_columns]


def main():
    st.title("🎓 Student Performance Tracker")
    st.write(
        "Predict a student's likely final exam score from study habits, "
        "attendance, and other lifestyle factors."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "No trained model found. Please run `python src/train_models.py` "
            "first to generate `models/best_model.pkl`."
        )
        return

    model, model_name, feature_columns = load_model()
    st.caption(f"Using model: **{model_name}**")

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours_per_week = st.slider("Study hours per week", 0.0, 40.0, 15.0, 0.5)
            attendance_percent = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
            previous_grade = st.slider("Previous grade (%)", 0.0, 100.0, 70.0, 1.0)
            sleep_hours = st.slider("Average sleep hours", 3.0, 10.0, 7.0, 0.5)

        with col2:
            extracurricular_hours = st.slider("Extracurricular hours/week", 0.0, 20.0, 4.0, 0.5)
            parental_support = st.selectbox("Parental support", ["Low", "Medium", "High"], index=1)
            internet_access = st.selectbox("Internet access at home", ["Yes", "No"], index=0)
            part_time_job = st.selectbox("Has a part-time job", ["Yes", "No"], index=1)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)

        submitted = st.form_submit_button("Predict Final Score")

    if submitted:
        inputs = {
            "gender": gender,
            "study_hours_per_week": study_hours_per_week,
            "attendance_percent": attendance_percent,
            "previous_grade": previous_grade,
            "sleep_hours": sleep_hours,
            "extracurricular_hours": extracurricular_hours,
            "parental_support": parental_support,
            "internet_access": internet_access,
            "part_time_job": part_time_job,
        }

        row_df = build_input_row(inputs)
        row_df = align_columns(row_df, feature_columns)

        prediction = model.predict(row_df)[0]
        prediction = max(0, min(100, prediction))

        st.success(f"### Predicted Final Score: {prediction:.1f} / 100")

        if prediction >= 85:
            st.info("Outstanding performance predicted. Keep up the great habits!")
        elif prediction >= 70:
            st.info("Good performance predicted. Small improvements in study time or attendance could help.")
        elif prediction >= 50:
            st.warning("Average performance predicted. Consider increasing study hours and attendance.")
        else:
            st.error("At-risk performance predicted. Early intervention and support are recommended.")


if __name__ == "__main__":
    main()
