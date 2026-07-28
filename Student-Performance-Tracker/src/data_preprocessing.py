"""
data_preprocessing.py
----------------------
Reusable data cleaning and feature engineering functions for the
Student Performance Tracker project.

These functions are imported by both the Jupyter notebook and the
Streamlit app, so cleaning logic only lives in one place.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load the raw student performance CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - Fill missing numeric values with column median
      - Strip whitespace from categorical columns
      - Drop exact duplicate rows
    """
    df = df.copy()

    # Drop full duplicates, if any
    df = df.drop_duplicates()

    # Fill missing numeric values with the median of that column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Clean up categorical text columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col != "student_id":
            df[col] = df[col].astype(str).str.strip()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that help the models capture non-linear
    relationships, and one-hot encode categorical variables.
    """
    df = df.copy()

    # Engineered numeric features
    df["study_attendance_ratio"] = df["study_hours_per_week"] / (
        df["attendance_percent"] + 1
    )
    df["engagement_score"] = (
        df["study_hours_per_week"] + df["attendance_percent"] / 10
    )
    df["rest_balance"] = df["sleep_hours"] - df["extracurricular_hours"] / 5

    # Ordinal encode parental_support (ordered category)
    support_order = {"Low": 0, "Medium": 1, "High": 2}
    df["parental_support_encoded"] = df["parental_support"].map(support_order)

    # One-hot encode remaining nominal categoricals
    df = pd.get_dummies(
        df,
        columns=["gender", "internet_access", "part_time_job"],
        drop_first=True,
    )

    # Drop columns not used for modeling
    drop_cols = ["student_id", "parental_support"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_feature_target_split(df: pd.DataFrame, target: str = "final_score"):
    """Split a fully-processed dataframe into features (X) and target (y)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
