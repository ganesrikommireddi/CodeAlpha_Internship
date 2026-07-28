"""
generate_dataset.py
--------------------
Generates a synthetic but realistic "Student Performance" dataset and
saves it to data/student_performance.csv.

This script is provided for transparency/reproducibility. The dataset
that ships with the repo (data/student_performance.csv) was created by
running this script once with a fixed random seed.
"""

import numpy as np
import pandas as pd
import os

# Fixed seed for reproducibility
np.random.seed(42)

N = 500  # number of students

# ---- Base feature generation -------------------------------------------------
study_hours_per_week = np.round(np.random.normal(15, 6, N).clip(1, 40), 1)
attendance_percent = np.round(np.random.normal(80, 12, N).clip(30, 100), 1)
previous_grade = np.round(np.random.normal(70, 12, N).clip(35, 100), 1)
sleep_hours = np.round(np.random.normal(7, 1.3, N).clip(3, 10), 1)
extracurricular_hours = np.round(np.random.gamma(2, 2, N).clip(0, 20), 1)
parental_support = np.random.choice(
    ["Low", "Medium", "High"], size=N, p=[0.25, 0.45, 0.30]
)
internet_access = np.random.choice(["Yes", "No"], size=N, p=[0.85, 0.15])
gender = np.random.choice(["Male", "Female"], size=N)
part_time_job = np.random.choice(["Yes", "No"], size=N, p=[0.3, 0.7])

support_map = {"Low": 0, "Medium": 5, "High": 10}
internet_map = {"Yes": 3, "No": 0}
job_map = {"Yes": -3, "No": 0}

# ---- Target: final exam score --------------------------------------------
# A weighted combination of the features + random noise, clipped to 0-100.
final_score = (
    0.35 * study_hours_per_week
    + 0.30 * attendance_percent
    + 0.25 * previous_grade
    + 1.2 * sleep_hours
    + 0.4 * extracurricular_hours
    + np.vectorize(support_map.get)(parental_support)
    + np.vectorize(internet_map.get)(internet_access)
    + np.vectorize(job_map.get)(part_time_job)
    + np.random.normal(0, 6, N)  # noise
)

final_score = np.round(final_score.clip(0, 100), 1)

df = pd.DataFrame(
    {
        "student_id": [f"S{1000 + i}" for i in range(N)],
        "gender": gender,
        "study_hours_per_week": study_hours_per_week,
        "attendance_percent": attendance_percent,
        "previous_grade": previous_grade,
        "sleep_hours": sleep_hours,
        "extracurricular_hours": extracurricular_hours,
        "parental_support": parental_support,
        "internet_access": internet_access,
        "part_time_job": part_time_job,
        "final_score": final_score,
    }
)

# Introduce a few missing values on purpose, to demonstrate data cleaning
for col in ["attendance_percent", "sleep_hours", "previous_grade"]:
    idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[idx, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)
print("Dataset saved to data/student_performance.csv")
print(df.head())
