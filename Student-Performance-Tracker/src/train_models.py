"""
train_models.py
-----------------
Trains and evaluates multiple regression models on the Student
Performance dataset, compares their performance, and pickles the
best-performing model to models/best_model.pkl.

Run directly with:
    python src/train_models.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    clean_data,
    engineer_features,
    get_feature_target_split,
    load_data,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate_model(name, model, X_test, y_test):
    """Compute MAE, RMSE, and R^2 for a fitted model."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    # 1. Load and clean data
    raw_df = load_data(DATA_PATH)
    clean_df = clean_data(raw_df)
    processed_df = engineer_features(clean_df)

    # 2. Train / test split
    X, y = get_feature_target_split(processed_df, target="final_score")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define candidate models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    # 4. Train and evaluate each model
    results = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\nModel comparison (sorted by R2):")
    print(results_df.to_string(index=False))

    # 5. Select and save the best model (highest R2)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "model_name": best_model_name,
                "feature_columns": list(X.columns),
            },
            f,
        )
    print(f"Best model saved to {model_path}")

    # Also save the comparison table for the README / notebook
    results_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Model comparison table saved to {results_path}")


if __name__ == "__main__":
    main()
