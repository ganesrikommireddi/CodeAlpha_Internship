"""
train.py
--------
End-to-end training pipeline for the Real Estate Appraisal System.

Steps:
    1. Load raw data
    2. Clean data (missing values, duplicates, outliers)
    3. Engineer features
    4. Train/test split
    5. Train Linear Regression, Random Forest, and Gradient Boosting models
    6. Evaluate with MAE, RMSE, R2
    7. Save the best-performing model (and preprocessing pipeline) to models/

Run from the project root with:
    python src/train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    return df


# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Impute missing numeric values with the column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Remove extreme price outliers using the IQR method
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    before = len(df)
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]
    print(f"Removed {before - len(df)} price outlier rows")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price per square foot style ratios help tree models find structure faster
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    df["lot_to_house_ratio"] = df["lot_size"] / df["square_footage"]
    df["amenity_score"] = df["has_pool"] + df["has_garden"] + df["renovated"] + (df["garage_spaces"] > 0).astype(int)
    df["is_new"] = (df["house_age"] <= 5).astype(int)

    return df


NUMERIC_FEATURES = [
    "square_footage", "bedrooms", "bathrooms", "lot_size", "house_age",
    "distance_to_city_center", "garage_spaces", "has_pool", "has_garden",
    "school_rating", "crime_rate", "renovated", "rooms_total",
    "lot_to_house_ratio", "amenity_score", "is_new",
]
CATEGORICAL_FEATURES = ["neighborhood_quality", "condition", "property_type"]
TARGET = "price"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# 4-6. Train, evaluate, compare
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        scores = evaluate(y_test, preds)
        results[name] = scores
        fitted_pipelines[name] = pipeline
        print(f"\n{name}")
        print(f"  MAE : {scores['MAE']:,.2f}")
        print(f"  RMSE: {scores['RMSE']:,.2f}")
        print(f"  R2  : {scores['R2']:.4f}")

    # Pick best model by R2
    best_name = max(results, key=lambda n: results[n]["R2"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (R2={results[best_name]['R2']:.4f})")

    # Save best model pipeline (includes preprocessing) with pickle/joblib
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model to {model_path}")

    # Save metadata used by the Streamlit app and for documentation
    metadata = {
        "best_model": best_name,
        "results": results,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": {
            col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {MODELS_DIR / 'metadata.json'}")

    # Save a small results table for the README / notebook
    results_df = pd.DataFrame(results).T
    results_df.to_csv(MODELS_DIR / "model_comparison.csv")
    print("\nModel comparison table:")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
